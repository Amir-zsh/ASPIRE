from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class BatchView:
    """Read-only per-request state a policy decides from."""

    draft_counts: IntArray
    finished: BoolArray
    seq_lens: NDArray[np.int64]

    def __post_init__(self) -> None:
        size = self.draft_counts.size
        if self.finished.size != size or self.seq_lens.size != size:
            raise ValueError("scheduler arrays must have identical lengths")

    @property
    def active(self) -> BoolArray:
        return ~self.finished


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Which requests verify this step, the per-request targets, and
    diagnostic metrics for logging."""

    verify: BoolArray
    targets: IntArray
    metrics: dict[str, float | int]


class Policy(Protocol):
    """Draft-length policy: decide() picks this step's verify set,
    observe_verify() feeds back acceptance outcomes, reset_slots() clears
    state when batch slots are refilled with new requests."""

    max_draft: int

    def decide(self, batch: BatchView) -> PolicyDecision: ...

    def observe_verify(
        self,
        verify_mask: BoolArray,
        accepted: IntArray,
        draft_counts: IntArray,
    ) -> None: ...

    def reset_slots(self, slot_ids: NDArray[np.integer]) -> None: ...

