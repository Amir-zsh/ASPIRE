from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Method(StrEnum):
    AUTOREGRESSIVE = "autoregressive"
    ASPIRE_FIXED = "aspire-fixed"
    ASPIRE_FSM = "aspire-fsm"
    ASPIRE = "aspire"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model: str
    checkpoint: Path
    tp_size: int = 1
    dtype: str = "bf16"
    max_batch_size: int = 8
    max_seq_len: int = 8192
    max_prefill_tokens: int = 8192
    page_size: int = 16
    top_k_pages: int = 32
    recent_pages: int = 8
    draft_percent: float | None = 7.0
    refresh_layer: int = -2
    deterministic_attention: bool = False
    holistic_attention: bool = True
    compile: bool = True
    cuda_graphs: bool = True
    graph_buckets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "graph_buckets", tuple(self.graph_buckets))
        if self.tp_size < 1:
            raise ValueError("tp_size must be positive")
        if self.max_batch_size < 1 or self.max_seq_len < 1 or self.max_prefill_tokens < 1:
            raise ValueError(
                "max_batch_size, max_seq_len, and max_prefill_tokens must be positive"
            )
        if self.page_size < 1 or self.top_k_pages < 1:
            raise ValueError("page_size and top_k_pages must be positive")
        if not 0 <= self.recent_pages <= self.top_k_pages:
            raise ValueError("recent_pages must be between zero and top_k_pages")
        if self.dtype not in {"bf16", "fp16"}:
            raise ValueError("dtype must be bf16 or fp16")
        if self.cuda_graphs and not self.compile:
            raise ValueError("cuda_graphs requires compile=true")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    min_draft: int = 1
    max_draft: int = 16
    initial_draft: int = 3
    fixed_draft: int = 5
    omega: float = 0.8
    alpha_init: float = 0.9
    calibration: Path | None = None

    def __post_init__(self) -> None:
        if self.calibration is not None:
            object.__setattr__(self, "calibration", Path(self.calibration))
        if not 0 <= self.min_draft <= self.initial_draft <= self.max_draft:
            raise ValueError("require min_draft <= initial_draft <= max_draft")
        if not self.min_draft <= self.fixed_draft <= self.max_draft:
            raise ValueError("fixed_draft must be within scheduler bounds")
        if not 0.0 <= self.omega < 1.0:
            raise ValueError("omega must be in [0, 1)")
        if not 0.0 < self.alpha_init <= 1.0:
            raise ValueError("alpha_init must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    seed: int = 123
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "eos_token_ids", tuple(self.eos_token_ids))
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")

    @property
    def sampling(self) -> bool:
        return self.temperature > 0.0


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str = "synthetic"
    subset: str | None = None
    revision: str | None = None
    split: str = "test"
    limit: int | None = None
    min_prompt_tokens: int | None = None
    max_prompt_tokens: int | None = None
    prompt_lengths: tuple[int, ...] = ()
    apply_chat_template: bool = True
    enable_thinking: bool = False
    domain: str | None = None
    difficulty: str | None = None
    length_category: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_lengths", tuple(self.prompt_lengths))
        if self.limit is not None and self.limit < 1:
            raise ValueError("dataset limit must be positive")
        if any(length < 1 for length in self.prompt_lengths):
            raise ValueError("synthetic prompt lengths must be positive")
        if (
            self.min_prompt_tokens is not None
            and self.max_prompt_tokens is not None
            and self.min_prompt_tokens > self.max_prompt_tokens
        ):
            raise ValueError("min_prompt_tokens cannot exceed max_prompt_tokens")
        if self.difficulty not in {None, "easy", "hard"}:
            raise ValueError("LongBench-v2 difficulty must be easy or hard")
        if self.length_category not in {None, "short", "medium", "long"}:
            raise ValueError(
                "LongBench-v2 length_category must be short, medium, or long"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    warmup_samples: int = 8
    repetitions: int = 5
    output_dir: Path = Path("results")
    mode: str = "static"
    converge: bool = False
    converge_threshold: float = 0.01
    converge_window: int = 10
    converge_min_samples: int = 20
    max_evaluation_samples: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.warmup_samples < 0 or self.repetitions < 1:
            raise ValueError("warmup_samples must be nonnegative and repetitions must be positive")
        if self.mode not in {"static", "cyclic"}:
            raise ValueError("benchmark mode must be static or cyclic")
        if self.converge and self.mode != "cyclic":
            raise ValueError("benchmark convergence requires mode=cyclic")
        if not 0.0 < self.converge_threshold < 1.0:
            raise ValueError("converge_threshold must be in (0, 1)")
        if self.converge_window < 1 or self.converge_min_samples < 1:
            raise ValueError("convergence sample counts must be positive")
        if self.max_evaluation_samples is not None and self.max_evaluation_samples < 1:
            raise ValueError("max_evaluation_samples must be positive")


@dataclass(frozen=True, slots=True)
class RunConfig:
    engine: EngineConfig
    scheduler: SchedulerConfig = SchedulerConfig()
    generation: GenerationConfig = GenerationConfig()
    dataset: DatasetConfig = DatasetConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    method: Method = Method.ASPIRE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _serialize(payload)


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, (Path, StrEnum)):
        return str(value)
    return value
