from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bucket_size(
    tokens: int | ArrayLike,
    buckets: tuple[int, ...] | None = None,
) -> int | NDArray[np.int64]:
    """Round a token count up to its CUDA-graph bucket (padded shape)."""
    values = np.asarray(tokens, dtype=np.int64)
    if buckets is None:
        rounded = np.where(
            values <= 0,
            0,
            np.where(
                values <= 4,
                1 << np.ceil(np.log2(np.maximum(values, 1))).astype(np.int64),
                ((values + 7) // 8) * 8,
            ),
        )
    else:
        if not buckets or tuple(sorted(set(buckets))) != buckets or buckets[0] < 1:
            raise ValueError("graph buckets must be nonempty, unique, sorted, and positive")
        choices = np.asarray(buckets, dtype=np.int64)
        indices = np.searchsorted(choices, np.maximum(values, 1), side="left")
        if np.any((values > 0) & (indices >= choices.size)):
            largest = int(values[(values > 0) & (indices >= choices.size)].max())
            raise ValueError(f"{largest} tokens exceed largest graph bucket {buckets[-1]}")
        rounded = np.where(values <= 0, 0, choices[np.minimum(indices, choices.size - 1)])
    if rounded.ndim == 0:
        return int(rounded)
    return rounded


@dataclass(frozen=True, slots=True)
class TimingSample:
    """One measured forward step: its time and the shape that produced it."""

    forward_ms: float
    active_requests: int
    extra_decode_tokens: int
    draft_requests: int
    verify_kv_tokens: int

    @property
    def graph_bucket(self) -> int:
        return int(bucket_size(self.active_requests + self.extra_decode_tokens))


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    """Configuration a calibration is valid for; must match at load time."""

    model: str
    gpu_name: str
    tp_size: int
    dtype: str
    max_batch_size: int
    page_size: int
    top_k_pages: int
    graph_buckets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Fitted 3-parameter forward-time model:
    time = base + bucket_ms * graph_bucket(tokens) + attn_ms * total_KV_read."""

    base_ms: float
    bucket_ms: float
    attention_ms_per_token: float
    draft_cache_tokens: int
    key: CalibrationKey
    samples: int
    r2: float
    rmse_ms: float

    def step_cost(
        self,
        active_requests: float | ArrayLike,
        extra_decode_tokens: float | ArrayLike,
        draft_requests: float | ArrayLike,
        verify_kv_tokens: float | ArrayLike,
    ) -> float | NDArray[np.float64]:
        # Predicted step time for a mixed batch; draft rows read
        # draft_cache_tokens of KV each, verify rows their full context.
        active = np.asarray(active_requests)
        extra = np.asarray(extra_decode_tokens)
        total_kv = np.asarray(verify_kv_tokens) + np.asarray(draft_requests) * self.draft_cache_tokens
        result = (
            self.base_ms
            + self.bucket_ms * bucket_size(active + extra, self.key.graph_buckets)
            + self.attention_ms_per_token * total_kv
        )
        if np.ndim(result) == 0:
            return float(result)
        return np.asarray(result, dtype=np.float64)

    def homogeneous_ratio(self, draft_counts: ArrayLike, seq_lens: ArrayLike, batch_size: int) -> NDArray[np.float64]:
        # Per-request draft/verify cost ratio c_i assuming a batch of
        # batch_size copies of request i: one all-draft step vs one
        # all-verify step at its draft depth and context length.
        drafts = np.asarray(draft_counts, dtype=np.float64)
        seqs = np.asarray(seq_lens, dtype=np.float64)
        draft_ms = self.step_cost(batch_size, 0, batch_size, 0)
        verify_ms = (
            self.base_ms
            + self.bucket_ms
            * bucket_size(batch_size + batch_size * drafts, self.key.graph_buckets)
            + self.attention_ms_per_token * batch_size * seqs
        )
        return np.clip(float(draft_ms) / np.maximum(verify_ms, 1e-6), 1e-4, 1.0)

    def validate_key(self, expected: CalibrationKey) -> None:
        # Fail loudly on any GPU/model/TP/geometry mismatch instead of
        # silently scheduling with a wrong-machine fit.
        if self.key != expected:
            actual = asdict(self.key)
            wanted = asdict(expected)
            mismatches = {name: (actual[name], wanted[name]) for name in wanted if actual[name] != wanted[name]}
            raise ValueError(f"calibration does not match run configuration: {mismatches}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"]["graph_buckets"] = list(self.key.graph_buckets)
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "LatencyModel":
        payload = json.loads(Path(path).read_text())
        payload["key"]["graph_buckets"] = tuple(payload["key"]["graph_buckets"])
        payload["key"] = CalibrationKey(**payload["key"])
        return cls(**payload)


def fit_latency_model(samples: Iterable[TimingSample], key: CalibrationKey, draft_cache_tokens: int) -> LatencyModel:
    """Least-squares fit of LatencyModel from measured step timings."""
    rows = list(samples)
    if len(rows) < 3:
        raise ValueError("at least three timing samples are required")
    x = np.asarray(
        [
            [
                1.0,
                bucket_size(
                    row.active_requests + row.extra_decode_tokens,
                    key.graph_buckets,
                ),
                row.verify_kv_tokens + row.draft_requests * draft_cache_tokens,
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([row.forward_ms for row in rows], dtype=np.float64)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < 3:
        raise ValueError("timing samples do not span all three model features")
    if np.any(beta < 0):
        raise ValueError(f"latency fit produced a negative cost coefficient: {beta.tolist()}")
    prediction = x @ beta
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return LatencyModel(
        base_ms=float(beta[0]),
        bucket_ms=float(beta[1]),
        attention_ms_per_token=float(beta[2]),
        draft_cache_tokens=draft_cache_tokens,
        key=key,
        samples=len(rows),
        r2=1.0 - residual / total if total > 0 else 1.0,
        rmse_ms=float(np.sqrt(np.mean((y - prediction) ** 2))),
    )
