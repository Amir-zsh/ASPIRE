from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Sequence

import numpy as np

from aspire.policies import BatchView, Policy, PolicyDecision


@dataclass(slots=True)
class SchedulerStats:
    decisions: int = 0
    device_to_host_syncs: int = 0
    decision_uploads: int = 0
    broadcasts: int = 0
    cpu_broadcasts: int = 0
    transferred_bytes: int = 0
    decision_seconds: float = 0.0


_SHARED_GLOO_GROUP = None


def _shared_gloo_group():
    """Process-wide gloo group, used only at init and by the
    ``ASPIRE_SCHED_CROSSCHECK=1`` per-step decision compare. Created once
    because repeated ``new_group`` calls leak communicators.
    """
    global _SHARED_GLOO_GROUP
    import torch.distributed as dist

    if _SHARED_GLOO_GROUP is None:
        _SHARED_GLOO_GROUP = dist.new_group(backend="gloo")
    return _SHARED_GLOO_GROUP


class ReplicatedPlanner:
    """Deterministic scheduler replicated on every rank.

    Every rank holds the same NumPy state and policy and sees the same verify
    outcomes, so it reaches the same decision each step with no communication.
    The device receives the decision through one async pinned upload; the host
    receives the verify outcome through one deferred pinned readback consumed
    at the top of the next iteration.

    Cross-rank divergence is guarded by an init-time fingerprint comparison, a
    per-step device-side check of the uploaded finished state, and the
    ``ASPIRE_SCHED_CROSSCHECK=1`` mode which gloo-compares every decision.
    """

    ACCEPTED = 0
    COMMITTED = 1
    EOS = 2
    OBSERVATION_COLUMNS = 3
    # Ring depth bounds host run-ahead (slot reuse waits on the slot's
    # event); four slots keep those waits rare under deep pipelining.
    RING = 4

    def __init__(
        self,
        policy: Policy,
        seq_lens: Sequence[int],
        max_new_tokens: int | Sequence[int],
        device,
        initial_finished: Sequence[bool] | None = None,
        page_size: int | None = None,
    ) -> None:
        import torch

        seq_lens_array = np.asarray(seq_lens, dtype=np.int64)
        if seq_lens_array.ndim != 1 or seq_lens_array.size < 1:
            raise ValueError("seq_lens must be a nonempty one-dimensional sequence")
        if np.any(seq_lens_array < 1):
            raise ValueError("sequence lengths must be positive")
        max_new_tokens_array = np.asarray(max_new_tokens, dtype=np.int64)
        if max_new_tokens_array.ndim == 0:
            max_new_tokens_array = np.full(
                seq_lens_array.shape, int(max_new_tokens_array), dtype=np.int64
            )
        if max_new_tokens_array.shape != seq_lens_array.shape:
            raise ValueError(
                f"max_new_tokens must be scalar or have shape ({seq_lens_array.size},)"
            )
        if np.any(max_new_tokens_array < 1):
            raise ValueError("max_new_tokens must be positive")
        if policy is None:
            raise ValueError("the replicated planner requires a policy on every rank")

        self.policy = policy
        self.batch_size = int(seq_lens_array.size)
        self.max_new_tokens = max_new_tokens_array.copy()
        self.device = torch.device(device)
        self.stats = SchedulerStats()
        self.epoch = 0
        self.draft_counts = np.zeros(self.batch_size, dtype=np.int32)
        self.finished = np.zeros(self.batch_size, dtype=np.bool_)
        if initial_finished is not None:
            initial = np.asarray(initial_finished, dtype=np.bool_)
            if initial.shape != self.finished.shape:
                raise ValueError(
                    f"initial_finished must have shape ({self.batch_size},)"
                )
            self.finished[:] = initial
        self.seq_lens = seq_lens_array.copy()
        self.generated = np.ones(self.batch_size, dtype=np.int32)
        self._last_verify: np.ndarray | None = None
        self._pending_slot: int | None = None

        pin = self.device.type == "cuda"
        # Decision packet: [verify bool | finished bool | epoch byte], uploaded
        # asynchronously each step from an event-guarded pinned ring.
        packet_len = 2 * self.batch_size + 1
        self.device_decision_packet = torch.empty(
            packet_len, dtype=torch.uint8, device=self.device
        )
        self.device_decision = self.device_decision_packet[: self.batch_size].view(
            torch.bool
        )
        self.device_finished = self.device_decision_packet[
            self.batch_size : 2 * self.batch_size
        ].view(torch.bool)
        self._host_packets = [
            torch.empty(packet_len, dtype=torch.uint8, pin_memory=pin)
            for _ in range(self.RING)
        ]
        self._packet_events: list = [None] * self.RING
        self._packet_slot = 0

        # Draft/verify row-index upload buffers for the sampling paths.
        self._host_rows = torch.zeros(
            2 * self.batch_size, dtype=torch.int64, pin_memory=pin
        )
        self.device_rows = torch.zeros(
            2 * self.batch_size, dtype=torch.int64, device=self.device
        )
        # Per-step decode lengths (row 0) and mode flags (row 1, 1 = verify).
        self._host_step_ints = [
            torch.zeros((2, self.batch_size), dtype=torch.int32, pin_memory=pin)
            for _ in range(self.RING)
        ]
        self.device_step_ints = torch.zeros(
            (2, self.batch_size), dtype=torch.int32, device=self.device
        )

        # Deferred observation channel: device packet written after the verify
        # tail, copied on a private stream into a pinned ring slot, consumed
        # by the runtime one iteration later.
        self.device_observation = torch.empty(
            (self.batch_size, self.OBSERVATION_COLUMNS),
            dtype=torch.int32,
            device=self.device,
        )
        self._host_observations = [
            torch.empty(
                (self.batch_size, self.OBSERVATION_COLUMNS),
                dtype=torch.int32,
                pin_memory=pin,
            )
            for _ in range(self.RING)
        ]
        self._observation_events = [
            torch.cuda.Event() if pin else None for _ in range(self.RING)
        ]
        self._slot_in_flight = [False] * self.RING
        self._observation_slot = 0
        self._pending_needs_data = False
        self._zero_observation = np.zeros(
            (self.batch_size, self.OBSERVATION_COLUMNS), dtype=np.int32
        )
        self.copy_stream = torch.cuda.Stream(device=self.device) if pin else None

        self.page_size = page_size
        self.step_shape: dict[str, int] = {}
        self.step_rows: dict[str, np.ndarray] | None = None
        self.step_result: dict[str, int] = {}
        self._crosscheck = os.environ.get("ASPIRE_SCHED_CROSSCHECK", "0") == "1"
        self._assert_planner_replicated()

    # ------------------------------------------------------------------
    # guards
    # ------------------------------------------------------------------

    def _assert_planner_replicated(self) -> None:
        """One-time check that every rank constructed an identical planner."""
        import torch.distributed as dist

        distributed, _, world_size = self._distributed_context()
        if not distributed or world_size < 2:
            return
        fingerprint = [
            type(self.policy).__name__,
            self.batch_size,
            self.seq_lens.tolist(),
            self.max_new_tokens.tolist(),
            self.finished.tolist(),
            self.page_size,
        ]
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, fingerprint, group=_shared_gloo_group())
        if any(item != fingerprint for item in gathered):
            raise RuntimeError(
                f"replicated planner state diverged at init: {gathered}"
            )

    def _crosscheck_decision(self, verify: np.ndarray) -> None:
        import torch.distributed as dist

        distributed, _, world_size = self._distributed_context()
        if not distributed or world_size < 2:
            return
        digest = [
            self.epoch,
            verify.tolist(),
            self.finished.tolist(),
            self.draft_counts.tolist(),
            self.seq_lens.tolist(),
        ]
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, digest, group=_shared_gloo_group())
        if any(item != digest for item in gathered):
            raise RuntimeError(
                f"replicated scheduler decisions diverged at epoch {self.epoch}: "
                f"{gathered}"
            )

    def _distributed_context(self) -> tuple[bool, int, int]:
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        return distributed, rank, world_size

    def _validate_device_vector(self, name: str, tensor) -> None:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
        if tensor.ndim != 1 or tensor.numel() != self.batch_size:
            raise ValueError(f"{name} must have shape ({self.batch_size},)")

    def buffer_pointers(self) -> dict[str, int]:
        return {
            "device_observation": self.device_observation.data_ptr(),
            "device_decision": self.device_decision.data_ptr(),
            "device_decision_packet": self.device_decision_packet.data_ptr(),
            "device_rows": self.device_rows.data_ptr(),
        }

    def reset_stats(self) -> None:
        if self._last_verify is not None or self._pending_slot is not None:
            raise RuntimeError("cannot reset scheduler stats while a step is pending")
        self.stats = SchedulerStats()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def initialize_finished(self, finished) -> None:
        """Import initial EOS results when they cannot be known on the CPU.

        Every rank reads its own copy; the device data is identical across
        ranks by construction of the in-graph collectives.
        """
        import torch

        self._validate_device_vector("finished", finished)
        host = self._host_observations[0]
        host[:, self.EOS].copy_(finished, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
            self.stats.device_to_host_syncs += 1
            self.stats.transferred_bytes += self.batch_size * host.element_size()
        self.finished[:] = host[:, self.EOS].numpy().astype(np.bool_, copy=False)

    def reset_slots(
        self,
        slot_ids: Sequence[int],
        seq_lens: Sequence[int],
        max_new_tokens: Sequence[int] | None = None,
        initial_finished: Sequence[bool] | None = None,
    ) -> None:
        """Reset the replicated mirrors when fixed batch slots are refilled."""

        ids = np.asarray(slot_ids, dtype=np.int64)
        lengths = np.asarray(seq_lens, dtype=np.int64)
        if ids.ndim != 1 or ids.size < 1:
            raise ValueError("slot_ids must be a nonempty one-dimensional sequence")
        if np.unique(ids).size != ids.size:
            raise ValueError("slot_ids must be unique")
        if np.any(ids < 0) or np.any(ids >= self.batch_size):
            raise ValueError("slot_ids contain an out-of-range slot")
        if lengths.shape != ids.shape or np.any(lengths < 1):
            raise ValueError("seq_lens must match slot_ids and be positive")
        limits = None
        if max_new_tokens is not None:
            limits = np.asarray(max_new_tokens, dtype=np.int64)
            if limits.shape != ids.shape or np.any(limits < 1):
                raise ValueError("max_new_tokens must match slot_ids and be positive")
        if self._last_verify is not None or self._pending_slot is not None:
            raise RuntimeError("cannot reset slots while a scheduler step is pending")
        finished = np.zeros(ids.shape, dtype=np.bool_)
        if initial_finished is not None:
            finished = np.asarray(initial_finished, dtype=np.bool_)
            if finished.shape != ids.shape:
                raise ValueError("initial_finished must match slot_ids")

        self.draft_counts[ids] = 0
        self.finished[ids] = finished
        self.seq_lens[ids] = lengths
        if limits is not None:
            self.max_new_tokens[ids] = limits
        self.generated[ids] = 1
        self.policy.reset_slots(ids)

    # ------------------------------------------------------------------
    # per-step: decide -> (issue | complete) -> consume
    # ------------------------------------------------------------------

    def decide(self, local_finished):
        import torch

        self._validate_device_vector("local_finished", local_finished)
        if self._pending_slot is not None:
            raise RuntimeError("decide() called with an unconsumed observation")

        batch = BatchView(self.draft_counts, self.finished, self.seq_lens)
        started = time.perf_counter()
        decision: PolicyDecision = self.policy.decide(batch)
        self.stats.decision_seconds += time.perf_counter() - started
        self.stats.decisions += 1
        self._last_verify = decision.verify.copy()
        if self._crosscheck:
            self._crosscheck_decision(decision.verify)

        active = ~self.finished
        verify_mask = self._last_verify & active
        draft_mask = active & ~verify_mask
        dec = np.where(
            verify_mask,
            self.draft_counts + 1,
            np.where(draft_mask, 1, 0),
        ).astype(np.int64)
        spill_candidate = False
        if self.page_size is not None and draft_mask.any():
            boundary = (self.seq_lens + self.draft_counts) % self.page_size == 0
            spill_candidate = bool(np.any(draft_mask & boundary))
        verify_drafts = self.draft_counts[verify_mask]
        self.step_shape = {
            "num_draft": int(draft_mask.sum()),
            "num_verify": int(verify_mask.sum()),
            "step_width": int(dec.max()) if dec.size else 0,
            "total_tokens": int(dec.sum()),
            "verify_kv_tokens": int(self.seq_lens[verify_mask].sum()),
            "max_verify_draft": int(verify_drafts.max()) if verify_drafts.size else 0,
            "spill_candidate": spill_candidate,
        }
        self.step_rows = {
            "verify": self._last_verify.copy(),
            "finished": self.finished.copy(),
            "draft_counts": self.draft_counts.astype(np.int64),
            "seq_lens": self.seq_lens.copy(),
        }

        # Upload the decision packet from an event-guarded pinned ring slot.
        slot = self._packet_slot
        event = self._packet_events[slot]
        if event is not None:
            event.synchronize()
        packet = self._host_packets[slot]
        packet[: self.batch_size].view(torch.bool).numpy()[:] = self._last_verify
        packet[self.batch_size : 2 * self.batch_size].view(torch.bool).numpy()[:] = (
            self.finished
        )
        packet[-1] = self.epoch % 251
        self.device_decision_packet.copy_(
            packet, non_blocking=self.device.type == "cuda"
        )
        # Row-index lists for the sampling paths (ascending — the same values
        # torch.where(mask)[0] would produce). They share the packet slot's
        # reuse fence via the event recorded below.
        draft_rows = np.flatnonzero(draft_mask)
        verify_rows = np.flatnonzero(verify_mask)
        rows = self._host_rows.numpy()
        rows[: draft_rows.size] = draft_rows
        rows[self.batch_size : self.batch_size + verify_rows.size] = verify_rows
        self.device_rows.copy_(self._host_rows, non_blocking=self.device.type == "cuda")
        step_ints = self._host_step_ints[slot]
        ints = step_ints.numpy()
        ints[0, :] = dec
        # Device modes are where(raw_decision, 1, 0).
        ints[1, :] = self._last_verify.astype(np.int32)
        self.device_step_ints.copy_(step_ints, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            if event is None:
                event = torch.cuda.Event()
                self._packet_events[slot] = event
            event.record()
            self.stats.decision_uploads += 1
            self.stats.transferred_bytes += (
                packet.numel() * packet.element_size()
                + self._host_rows.numel() * self._host_rows.element_size()
            )
        self._packet_slot = (slot + 1) % self.RING

        expected_epoch = self.epoch % 251
        if self.device.type == "cuda":
            torch._assert_async(
                self.device_decision_packet[-1] == expected_epoch,
                "scheduler decision epoch diverged from device state",
            )
            torch._assert_async(
                torch.all(self.device_finished == local_finished),
                "scheduler finished mirror diverged from device state",
            )
        else:
            if int(self.device_decision_packet[-1]) != expected_epoch:
                raise RuntimeError("scheduler decision epoch diverged from device state")
            if not torch.equal(self.device_finished, local_finished):
                raise RuntimeError("scheduler finished mirror diverged from device state")
        self.epoch += 1
        return self.device_decision, decision.metrics

    def device_draft_rows(self, count: int):
        """Device view of this step's draft row indices (ascending)."""
        return self.device_rows[:count]

    def device_verify_rows(self, count: int):
        """Device view of this step's verify row indices (ascending)."""
        return self.device_rows[self.batch_size : self.batch_size + count]

    def _advance_draft_rows(self) -> None:
        active = ~self.finished
        draft_only = active & ~(self._last_verify & active)
        self.draft_counts[draft_only] += 1

    def complete(self, accepted, committed, hit_eos) -> None:
        """Synchronous convenience wrapper: issue and immediately consume.

        The runtime uses the split issue/consume path so the readback overlaps
        the next iteration's work; tests and simple callers keep the one-call
        form.
        """
        self.issue_observation(accepted, committed, hit_eos)
        self.consume_observation()

    def issue_observation(self, accepted, committed, hit_eos) -> None:
        """Start the deferred readback of this step's outcome.

        Every step retires through this one path regardless of its
        draft/verify composition: the ``[accepted, committed, hit_eos]``
        packet is copied on the private copy stream into a pinned ring slot
        and an event is recorded — no waiting here. The runtime calls
        :meth:`consume_observation` at the top of the next iteration, by
        which time the copy has completed behind the in-flight forward.
        """
        import torch

        for name, tensor in (
            ("accepted", accepted),
            ("committed", committed),
            ("hit_eos", hit_eos),
        ):
            self._validate_device_vector(name, tensor)
        if self._last_verify is None:
            raise RuntimeError("issue_observation() requires a preceding decide()")
        if self._pending_slot is not None:
            raise RuntimeError("issue_observation() called with one already pending")

        self._advance_draft_rows()
        packet = self.device_observation
        packet[:, self.ACCEPTED].copy_(accepted)
        packet[:, self.COMMITTED].copy_(committed)
        packet[:, self.EOS].copy_(hit_eos)
        slot = self._observation_slot
        if self.device.type == "cuda":
            if self._slot_in_flight[slot]:
                # The slot's previous copy was never waited on (its step had
                # no verify rows); fence before overwriting the pinned buffer.
                self._observation_events[slot].synchronize()
            current_stream = torch.cuda.current_stream(self.device)
            self.copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.copy_stream):
                self._host_observations[slot].copy_(packet, non_blocking=True)
                self._observation_events[slot].record(self.copy_stream)
            packet.record_stream(self.copy_stream)
            self._slot_in_flight[slot] = True
            self.stats.transferred_bytes += (
                self._host_observations[slot].numel()
                * self._host_observations[slot].element_size()
            )
        else:
            self._host_observations[slot].copy_(packet)
        self._pending_slot = slot
        # A step with no active verify rows produces an all-zero observation
        # by construction, so consuming it never needs the copied bytes.
        self._pending_needs_data = bool(np.any(self._last_verify & ~self.finished))
        self._observation_slot = (slot + 1) % self.RING

    def consume_observation(self) -> dict[str, int]:
        """Apply the pending step outcome to the mirrors.

        One path for every step composition: when the issued step carried
        verify rows the host waits on the copy event and reads the packet;
        otherwise the observation is all-zero by construction and the wait is
        skipped (the slot's event fences its eventual reuse).
        """
        if self._pending_slot is None:
            raise RuntimeError("consume_observation() called with none pending")
        if self._last_verify is None:
            raise RuntimeError("consume_observation() requires the issuing decide()")
        slot = self._pending_slot
        if self._pending_needs_data:
            if self.device.type == "cuda":
                self._observation_events[slot].synchronize()
                self._slot_in_flight[slot] = False
                self.stats.device_to_host_syncs += 1
            observation = self._host_observations[slot].numpy()
        else:
            observation = self._zero_observation
        accepted_cpu = observation[:, self.ACCEPTED]
        committed_cpu = observation[:, self.COMMITTED]
        eos_cpu = observation[:, self.EOS].astype(np.bool_, copy=False)

        active = ~self.finished
        verify = self._last_verify & active
        self.policy.observe_verify(verify, accepted_cpu, self.draft_counts)
        self.seq_lens[verify] += committed_cpu[verify]
        self.generated[verify] += committed_cpu[verify]
        self.finished[verify] |= (
            eos_cpu[verify] | (self.generated[verify] >= self.max_new_tokens[verify])
        )
        self.draft_counts[verify] = 0
        self._pending_slot = None
        self._last_verify = None
        self.step_result = {
            "accepted_tokens": int(accepted_cpu[verify].sum()),
            "committed_tokens": int(committed_cpu[verify].sum()),
            "any_finished": bool(self.finished.any()),
        }
        return self.step_result


SchedulerBridge = ReplicatedPlanner
