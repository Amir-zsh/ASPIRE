from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from aspire.config import BenchmarkConfig, EngineConfig, GenerationConfig, Method, SchedulerConfig


@dataclass(frozen=True, slots=True)
class GenerationResult:
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int


class Backend(Protocol):
    def generate(self, prompts: Sequence[Sequence[int]], config: GenerationConfig) -> list[GenerationResult]: ...
    def generate_cyclic(
        self,
        prompts: Sequence[Sequence[int]],
        config: GenerationConfig,
        benchmark: BenchmarkConfig,
    ) -> list[GenerationResult]: ...
    def graph_buffer_pointers(self) -> dict[str, int]: ...
    def close(self) -> None: ...


class AspireEngine:
    """Stable public facade around the CUDA backend."""

    def __init__(
        self,
        config: EngineConfig,
        method: Method = Method.ASPIRE,
        scheduler: SchedulerConfig | None = None,
        backend: Backend | None = None,
    ):
        self.config = config
        if backend is None:
            from aspire.engine.runtime import CudaBackend

            backend = CudaBackend(config, method, scheduler or SchedulerConfig())
        self._backend = backend

    def generate(self, prompts: Sequence[Sequence[int]], generation: GenerationConfig | None = None) -> list[GenerationResult]:
        if not prompts:
            return []
        if len(prompts) > self.config.max_batch_size:
            raise ValueError("prompt count exceeds max_batch_size")
        return self._backend.generate(prompts, generation or GenerationConfig())

    def generate_cyclic(
        self,
        prompts: Sequence[Sequence[int]],
        generation: GenerationConfig,
        benchmark: BenchmarkConfig,
    ) -> list[GenerationResult]:
        if not prompts:
            return []
        if len(prompts) > self.config.max_batch_size:
            raise ValueError("prompt pool exceeds max_batch_size")
        return self._backend.generate_cyclic(prompts, generation, benchmark)

    def close(self) -> None:
        self._backend.close()

    @property
    def metrics(self):
        """Metrics from the most recent generation, or ``None`` before a run."""

        return getattr(self._backend, "last_metrics", None)

    def calibration_key(self, max_draft: int):
        """Return the hardware/runtime key required by a latency calibration."""

        return self._backend.calibration_key(max_draft)

    def graph_buffer_pointers(self) -> dict[str, int]:
        """Return validated addresses for the captured CUDA graph buffers."""

        return self._backend.graph_buffer_pointers()

    def __enter__(self) -> "AspireEngine":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
