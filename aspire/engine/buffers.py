from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def graph_buckets(max_tokens: int) -> tuple[int, ...]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    values = [size for size in (1, 2, 4) if size <= max_tokens]
    values.extend(range(8, max_tokens + 1, 8))
    if max_tokens not in values:
        values.append(max_tokens)
    return tuple(sorted(set(values)))


def select_bucket(tokens: int, buckets: tuple[int, ...]) -> int:
    if tokens < 1:
        raise ValueError("tokens must be positive")
    for bucket in buckets:
        if tokens <= bucket:
            return bucket
    raise ValueError(f"{tokens} tokens exceed largest graph bucket {buckets[-1]}")


@dataclass(slots=True)
class _Entry:
    tensor: Any
    data_ptr: int
    shape: tuple[int, ...]
    dtype: Any
    device: Any


class StaticBufferRegistry:
    """Owns tensors whose addresses are captured by CUDA graphs."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._sealed = False

    def register(self, name: str, tensor: Any) -> Any:
        if self._sealed:
            raise RuntimeError("cannot register buffers after graph capture is sealed")
        if name in self._entries:
            raise ValueError(f"duplicate static buffer: {name}")
        if not hasattr(tensor, "data_ptr"):
            raise TypeError(f"buffer {name} is not tensor-like")
        self._entries[name] = _Entry(tensor, int(tensor.data_ptr()), tuple(tensor.shape), tensor.dtype, tensor.device)
        return tensor

    def get(self, name: str) -> Any:
        return self._entries[name].tensor

    def seal(self) -> None:
        self.assert_stable()
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def assert_stable(self) -> None:
        for name, entry in self._entries.items():
            tensor = entry.tensor
            actual = (int(tensor.data_ptr()), tuple(tensor.shape), tensor.dtype, tensor.device)
            expected = (entry.data_ptr, entry.shape, entry.dtype, entry.device)
            if actual != expected:
                raise RuntimeError(f"static buffer changed after registration: {name}")

    def pointers(self) -> dict[str, int]:
        self.assert_stable()
        return {name: entry.data_ptr for name, entry in self._entries.items()}

    def schema(self) -> tuple[tuple[str, tuple[int, ...], str, str], ...]:
        """Return a rank-comparable description without process-local pointers."""
        self.assert_stable()
        return tuple(
            (name, entry.shape, str(entry.dtype), entry.device.type)
            for name, entry in sorted(self._entries.items())
        )
