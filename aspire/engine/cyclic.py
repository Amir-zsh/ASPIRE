from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CyclicUpdate:
    warmup_crossed: bool
    evaluated: int
    stable: int
    converged: bool
    stop: bool
    stop_reason: str | None


class CyclicController:
    """Deterministic request cycling and cumulative-throughput convergence."""

    def __init__(
        self,
        *,
        pool_size: int,
        warmup_samples: int,
        max_evaluation_samples: int,
        converge: bool,
        converge_threshold: float,
        converge_window: int,
        converge_min_samples: int,
    ) -> None:
        if pool_size < 1:
            raise ValueError("cyclic prompt pool must be nonempty")
        if warmup_samples < 0 or max_evaluation_samples < 1:
            raise ValueError("cyclic sample budgets are invalid")
        if not 0.0 < converge_threshold < 1.0:
            raise ValueError("converge_threshold must be in (0, 1)")
        if converge_window < 1 or converge_min_samples < 1:
            raise ValueError("convergence sample counts must be positive")
        self.pool_size = pool_size
        self.warmup_samples = warmup_samples
        self.max_evaluation_samples = max_evaluation_samples
        self.converge = converge
        self.converge_threshold = converge_threshold
        self.converge_window = converge_window
        self.converge_min_samples = converge_min_samples
        self.cursor = 0
        self.total_completed = 0
        self.evaluated = 0
        self.warmup_done = warmup_samples == 0
        self.anchor_throughput: float | None = None
        self.stable = 0

    def next_indices(self, count: int) -> list[int]:
        if count < 1:
            raise ValueError("cyclic refill count must be positive")
        result = [
            (self.cursor + offset) % self.pool_size for offset in range(count)
        ]
        self.cursor = (self.cursor + count) % self.pool_size
        return result

    def complete(
        self, count: int, *, cumulative_forward_throughput: float | None
    ) -> CyclicUpdate:
        if count < 1:
            raise ValueError("completed sample count must be positive")
        was_warmup = not self.warmup_done
        self.total_completed += count
        warmup_crossed = was_warmup and self.total_completed >= self.warmup_samples
        if warmup_crossed:
            self.warmup_done = True
        elif not was_warmup:
            self.evaluated += count

        converged = False
        if (
            not was_warmup
            and self.converge
            and self.evaluated >= self.converge_min_samples
        ):
            if cumulative_forward_throughput is None or cumulative_forward_throughput <= 0:
                raise ValueError("convergence requires positive cumulative throughput")
            if self.anchor_throughput is None:
                self.anchor_throughput = cumulative_forward_throughput
            else:
                relative = abs(
                    cumulative_forward_throughput - self.anchor_throughput
                ) / self.anchor_throughput
                if relative < self.converge_threshold:
                    self.stable += count
                else:
                    self.stable = 0
                    self.anchor_throughput = cumulative_forward_throughput
                converged = self.stable >= self.converge_window

        budget_reached = self.evaluated >= self.max_evaluation_samples
        stop = converged or budget_reached
        reason = "converged" if converged else "max_evaluation_samples" if stop else None
        return CyclicUpdate(
            warmup_crossed=warmup_crossed,
            evaluated=self.evaluated,
            stable=self.stable,
            converged=converged,
            stop=stop,
            stop_reason=reason,
        )
