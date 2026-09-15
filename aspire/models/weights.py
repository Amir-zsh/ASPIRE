from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int

    @classmethod
    def from_internal(cls, config: Any) -> "ModelGeometry":
        return cls(
            hidden_size=config.dim,
            intermediate_size=config.intermediate_size,
            num_layers=config.n_layer,
            num_heads=config.n_head,
            num_kv_heads=config.n_local_heads,
            head_dim=config.head_dim,
            vocab_size=config.vocab_size,
        )


class SafeTensorStore:
    """Keep safetensor shards memory-mapped while parameters are assigned."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        single_path = self.checkpoint_dir / "model.safetensors"
        if index_path.is_file():
            payload = json.loads(index_path.read_text())
            self.weight_map = dict(payload["weight_map"])
        elif single_path.is_file():
            from safetensors import safe_open

            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self.weight_map = {key: single_path.name for key in handle.keys()}
        else:
            raise FileNotFoundError(
                f"no model.safetensors or model.safetensors.index.json in {self.checkpoint_dir}"
            )
        self._stack: ExitStack | None = None
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> "SafeTensorStore":
        from safetensors import safe_open

        self._stack = ExitStack()
        for filename in sorted(set(self.weight_map.values())):
            path = self.checkpoint_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            self._handles[filename] = self._stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._stack is not None
        self._stack.close()
        self._handles.clear()

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str):
        try:
            filename = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint is missing required tensor {name!r}") from exc
        return self._handles[filename].get_tensor(name)


def _validate_hf_geometry(checkpoint_dir: Path, expected: ModelGeometry) -> None:
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    actual = ModelGeometry(
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        num_layers=int(config["num_hidden_layers"]),
        num_heads=int(config["num_attention_heads"]),
        num_kv_heads=int(config.get("num_key_value_heads", config["num_attention_heads"])),
        head_dim=int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])),
        vocab_size=int(config["vocab_size"]),
    )
    if actual != expected:
        mismatches = {
            name: (getattr(actual, name), getattr(expected, name))
            for name in ModelGeometry.__dataclass_fields__
            if getattr(actual, name) != getattr(expected, name)
        }
        raise ValueError(f"Hugging Face config does not match engine geometry: {mismatches}")


def _kv_head_range(num_kv_heads: int, rank: int, world_size: int) -> tuple[int, int]:
    base, remainder = divmod(num_kv_heads, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + (1 if rank < remainder else 0)


def _chunk(tensor, rank: int, world_size: int, dim: int):
    chunks = tensor.chunk(world_size, dim=dim)
    if len(chunks) != world_size:
        raise ValueError(
            f"cannot shard shape {tuple(tensor.shape)} into {world_size} chunks on dim {dim}"
        )
    return chunks[rank].contiguous()


def _permute_rope(tensor, num_heads: int, head_dim: int):
    if tensor.ndim == 2:
        return (
            tensor.view(num_heads, 2, head_dim // 2, tensor.shape[1])
            .transpose(1, 2)
            .reshape(num_heads * head_dim, tensor.shape[1])
        )
    return (
        tensor.view(num_heads, 2, head_dim // 2)
        .transpose(1, 2)
        .reshape(num_heads * head_dim)
    )


def _layer_source_name(layer: int, suffix: str) -> str:
    return f"model.layers.{layer}.{suffix}"


def _load_parameter(
    name: str,
    store: SafeTensorStore,
    geometry: ModelGeometry,
    model_name: str,
    rank: int,
    world_size: int,
):
    import torch

    if name == "tok_embeddings.weight":
        return store.get("model.embed_tokens.weight")
    if name == "norm.weight":
        return store.get("model.norm.weight")
    if name == "output.weight":
        source = "lm_head.weight" if store.has("lm_head.weight") else "model.embed_tokens.weight"
        tensor = store.get(source)
        return _chunk(tensor, rank, world_size, 0) if world_size > 1 else tensor

    match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
    if match is None:
        raise KeyError(f"unsupported engine parameter {name!r}")
    layer = int(match.group(1))
    suffix = match.group(2)
    direct = {
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
    }
    if suffix in direct:
        return store.get(_layer_source_name(layer, direct[suffix]))

    if suffix.startswith("feed_forward."):
        projection, kind = suffix.split(".")[1:]
        source_projection = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[projection]
        tensor = store.get(
            _layer_source_name(layer, f"mlp.{source_projection}.{kind}")
        )
        if world_size == 1:
            return tensor
        return _chunk(tensor, rank, world_size, 0 if projection in {"w1", "w3"} else 1)

    if suffix == "attention.wo.weight":
        tensor = store.get(_layer_source_name(layer, "self_attn.o_proj.weight"))
        if world_size == 1:
            return tensor
        kv_start, kv_end = _kv_head_range(geometry.num_kv_heads, rank, world_size)
        groups = geometry.num_heads // geometry.num_kv_heads
        q_start = kv_start * groups * geometry.head_dim
        q_end = kv_end * groups * geometry.head_dim
        return tensor[:, q_start:q_end].contiguous()

    if suffix in {"attention.wqkv.weight", "attention.wqkv.bias"}:
        kind = suffix.rsplit(".", 1)[1]
        q = store.get(_layer_source_name(layer, f"self_attn.q_proj.{kind}"))
        k = store.get(_layer_source_name(layer, f"self_attn.k_proj.{kind}"))
        v = store.get(_layer_source_name(layer, f"self_attn.v_proj.{kind}"))
        if "qwen3" not in model_name.lower():
            q = _permute_rope(q, geometry.num_heads, geometry.head_dim)
            k = _permute_rope(k, geometry.num_kv_heads, geometry.head_dim)
        if world_size > 1:
            kv_start, kv_end = _kv_head_range(geometry.num_kv_heads, rank, world_size)
            groups = geometry.num_heads // geometry.num_kv_heads
            q_start = kv_start * groups * geometry.head_dim
            q_end = kv_end * groups * geometry.head_dim
            k_start = kv_start * geometry.head_dim
            k_end = kv_end * geometry.head_dim
            q = q[q_start:q_end]
            k = k[k_start:k_end]
            v = v[k_start:k_end]
        return torch.cat((q, k, v), dim=0).contiguous()

    raise KeyError(f"unsupported engine parameter {name!r}")


def _assign_parameter(model, name: str, tensor) -> None:
    import torch

    module_name, parameter_name = name.rsplit(".", 1)
    module = model.get_submodule(module_name)
    module._parameters[parameter_name] = torch.nn.Parameter(tensor, requires_grad=False)


def load_hf_transformer(
    checkpoint_dir: str | Path,
    model_name: str,
    *,
    device,
    dtype,
    use_tp: bool,
    rank_group: list[int] | None,
    group,
):
    """Load standard HF safetensors directly into a rank-local engine model."""

    import torch

    from aspire.models.sparse_draft_transformer import TransformerSparseDraft

    checkpoint_dir = Path(checkpoint_dir)
    with torch.device("meta"):
        model = TransformerSparseDraft.from_name(model_name)
    geometry = ModelGeometry.from_internal(model.config)
    _validate_hf_geometry(checkpoint_dir, geometry)

    world_size = len(rank_group) if use_tp and rank_group is not None else 1
    if use_tp:
        if rank_group is None or group is None:
            raise ValueError("TP loading requires rank_group and process group")
        from aspire.tp import apply_tp

        apply_tp(model, rank_group, group=group)
        global_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = rank_group.index(global_rank)
    else:
        rank = 0

    expected_parameters = dict(model.named_parameters())
    with SafeTensorStore(checkpoint_dir) as store:
        for name, parameter in expected_parameters.items():
            tensor = _load_parameter(
                name,
                store,
                geometry,
                model_name,
                rank,
                world_size,
            )
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, "
                    f"engine {tuple(parameter.shape)}"
                )
            _assign_parameter(
                model,
                name,
                tensor.to(device=device, dtype=dtype, non_blocking=False),
            )
            del tensor
    return model.eval()
