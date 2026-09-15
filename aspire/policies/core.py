from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aspire.policies.base import BatchView, BoolArray, IntArray, PolicyDecision
from aspire.policies.timing import LatencyModel


def expected_token_targets(
    alphas: ArrayLike,
    cost_ratios: ArrayLike,
    min_draft: int,
    max_draft: int,
) -> IntArray:
    """Per-request optimal draft length: the gamma maximizing expected
    committed tokens per unit time, ((1-a^(g+1))/(1-a)) / (1 + c*g), given
    acceptance estimate a and draft/verify cost ratio c."""
    alpha = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 1.0)
    cost = np.broadcast_to(np.asarray(cost_ratios, dtype=np.float64), alpha.shape)
    if max_draft < 1:
        targets = np.full(alpha.shape, min_draft, dtype=np.int32)
        targets[alpha >= 1.0 - 1e-7] = max_draft
        return np.clip(targets, min_draft, max_draft).astype(np.int32)
    # Vectorized ascending scan over gamma candidates; the result is the
    # largest gamma whose margin condition holds. Integer-typed exponents keep
    # np.power on the integer-power path.
    gammas = np.arange(1, max_draft + 1, dtype=np.int64)[:, np.newaxis]
    lhs = alpha[np.newaxis, :] ** gammas * (
        (1.0 - alpha)[np.newaxis, :] * cost[np.newaxis, :] * (gammas - 1).astype(np.float64)
        + cost[np.newaxis, :]
        + (1.0 - alpha)[np.newaxis, :]
    )
    hold = lhs > cost[np.newaxis, :]
    any_hold = hold.any(axis=0)
    last_hold = (max_draft - 1) - np.argmax(hold[::-1], axis=0)
    targets = np.where(any_hold, (last_hold + 1).astype(np.int32), np.int32(min_draft))
    targets = np.where(alpha >= 1.0 - 1e-7, np.int32(max_draft), targets)
    return np.clip(targets, min_draft, max_draft).astype(np.int32)


class FixedDraftPolicy:
    """Every request verifies at the same fixed draft length (gamma=0 is
    the autoregressive baseline)."""

    def __init__(self, batch_size: int, gamma: int):
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.max_draft = gamma
        self.targets = np.full(batch_size, gamma, dtype=np.int32)

    def decide(self, batch: BatchView) -> PolicyDecision:
        # Verify any active request that has reached its fixed target.
        verify = (batch.draft_counts >= self.targets) & batch.active
        return PolicyDecision(verify, self.targets.copy(), {})

    def observe_verify(self, verify_mask: BoolArray, accepted: IntArray, draft_counts: IntArray) -> None:
        return None  # no state to adapt

    def reset_slots(self, slot_ids: NDArray[np.integer]) -> None:
        return None  # no per-slot state


class FeedbackPolicy:
    """Per-request target with a +1/-1 feedback rule: a fully accepted
    verification raises the request's draft target, any rejection lowers it."""

    def __init__(self, batch_size: int, min_draft: int, max_draft: int, initial_draft: int):
        self.min_draft = min_draft
        self.max_draft = max_draft
        self.initial_draft = initial_draft
        self.targets = np.full(batch_size, initial_draft, dtype=np.int32)

    def decide(self, batch: BatchView) -> PolicyDecision:
        # Verify any active request that has reached its current target.
        verify = (batch.draft_counts >= self.targets) & batch.active
        return PolicyDecision(verify, self.targets.copy(), {"target_mean": float(self.targets.mean())})

    def observe_verify(self, verify_mask: BoolArray, accepted: IntArray, draft_counts: IntArray) -> None:
        # Fully accepted -> target+1 (capped); any rejection -> target-1 (floored).
        ids = np.flatnonzero(verify_mask)
        if ids.size == 0:
            return
        current = self.targets[ids]
        all_accepted = accepted[ids] >= current
        self.targets[ids] = np.where(all_accepted, np.minimum(current + 1, self.max_draft), np.maximum(current - 1, self.min_draft))

    def reset_slots(self, slot_ids: NDArray[np.integer]) -> None:
        # A refilled slot starts over from the initial target.
        self.targets[slot_ids] = self.initial_draft


class AspirePolicy:
    """Per-request scheduler: each request selects the draft-length target
    maximizing expected committed tokens per unit time, with the cost ratio
    predicted online by the fitted batch-aware forward-time model."""

    def __init__(
        self,
        batch_size: int,
        min_draft: int,
        max_draft: int,
        initial_draft: int,
        latency: LatencyModel,
        omega: float = 0.8,
        alpha_init: float = 0.9,
    ):
        self.batch_size = batch_size
        self.min_draft = min_draft
        self.max_draft = max_draft
        self.initial_draft = initial_draft
        self.latency = latency
        self.omega = omega
        self.alpha_init = alpha_init
        self.alpha = np.full(batch_size, alpha_init, dtype=np.float64)
        self.fresh = np.ones(batch_size, dtype=np.bool_)
        self.targets = np.full(batch_size, initial_draft, dtype=np.int32)

    def decide(self, batch: BatchView) -> PolicyDecision:
        # Each request's target comes from the expected-token rule under the
        # homogeneous-batch cost ratio (a batch of copies of that request);
        # it verifies when its draft length reaches the target or d_max.
        active = batch.active
        drafts = batch.draft_counts.astype(np.float64, copy=False)
        seq_lens = batch.seq_lens.astype(np.float64, copy=False)
        minimum = np.where(self.fresh, max(self.min_draft, self.initial_draft), self.min_draft).astype(np.int32)
        can_verify = batch.draft_counts >= minimum
        must_verify = (batch.draft_counts >= self.max_draft) & active

        ratio = self.latency.homogeneous_ratio(drafts, seq_lens, self.batch_size)
        targets = expected_token_targets(self.alpha, ratio, self.min_draft, self.max_draft)
        self.targets = np.minimum(np.maximum(targets, minimum), self.max_draft).astype(np.int32)
        verify = (((batch.draft_counts >= self.targets) & can_verify) | must_verify) & active

        draft_mask = active & ~verify
        reward = float(np.sum(self.alpha ** (drafts + 1.0) * draft_mask) + verify.sum())
        predicted_ms = self.latency.step_cost(int(active.sum()), float(np.dot(drafts, verify)), int(draft_mask.sum()), float(np.dot(seq_lens, verify)))
        metrics: dict[str, float | int] = {
            "target_mean": float(self.targets.mean()),
            "alpha_mean": float(self.alpha.mean()),
            "verify_requests": int(verify.sum()),
            "draft_requests": int(draft_mask.sum()),
            "predicted_reward": reward,
            "predicted_forward_ms": float(predicted_ms),
        }
        return PolicyDecision(verify.copy(), self.targets.copy(), metrics)

    def observe_verify(self, verify_mask: BoolArray, accepted: IntArray, draft_counts: IntArray) -> None:
        # EMA acceptance update: a_hat = accepted / min(accepted+1, drafted)
        # treats a partial acceptance as `accepted` successes then one failure
        # and a full acceptance as `drafted` successes.
        if np.any(accepted[verify_mask] < 0) or np.any(accepted[verify_mask] > draft_counts[verify_mask]):
            raise ValueError("accepted tokens must be between zero and the drafted-token count")
        valid = verify_mask & (draft_counts > 0)
        if not valid.any():
            return
        accepted_f = accepted[valid].astype(np.float64)
        drafts_f = draft_counts[valid].astype(np.float64)
        instantaneous = accepted_f / np.minimum(accepted_f + 1.0, drafts_f)
        self.alpha[valid] = self.omega * self.alpha[valid] + (1.0 - self.omega) * instantaneous
        self.fresh[valid] = False

    def reset_slots(self, slot_ids: NDArray[np.integer]) -> None:
        # A refilled slot restarts from the prior: it must draft at least
        # initial_draft tokens before its first verification.
        self.alpha[slot_ids] = self.alpha_init
        self.fresh[slot_ids] = True
        self.targets[slot_ids] = self.initial_draft
