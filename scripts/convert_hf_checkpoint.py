# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Derived from the gpt-fast checkpoint converter, including model-family
# extensions developed in MagicDec. See NOTICE and LICENSES/.

"""Convert supported Hugging Face weights to the gpt-fast ``model.pth`` format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file


WEIGHT_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


def _weight_files(checkpoint_dir: Path) -> list[Path]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_dir / index_name
        if index_path.is_file():
            with index_path.open() as handle:
                index = json.load(handle)
            return sorted({checkpoint_dir / name for name in index["weight_map"].values()})
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint_dir / name
        if path.is_file():
            return [path]
    raise FileNotFoundError(f"no Hugging Face weight files found in {checkpoint_dir}")


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_safetensors_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def _mapped_name(name: str) -> str | None:
    if name == "model.rotary_emb.inv_freq" or name.endswith(".rotary_emb.inv_freq"):
        return None
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match:
        layer = match.group(1)
        abstract = name.replace(f".layers.{layer}.", ".layers.{}.")
        mapped = WEIGHT_MAP.get(abstract)
        return mapped.format(layer) if mapped else None
    return WEIGHT_MAP.get(name)


def _permute(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    if weight.ndim == 2:
        return (
            weight.view(heads, 2, head_dim // 2, weight.shape[1])
            .transpose(1, 2)
            .reshape(heads * head_dim, weight.shape[1])
        )
    return weight.view(heads, 2, head_dim // 2).transpose(1, 2).reshape(heads * head_dim)


@torch.inference_mode()
def convert(checkpoint_dir: Path) -> Path:
    with (checkpoint_dir / "config.json").open() as handle:
        config = json.load(handle)
    model_type = str(config.get("model_type", "")).lower()
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim", config["hidden_size"] // heads))

    converted: dict[str, torch.Tensor] = {}
    for weight_file in _weight_files(checkpoint_dir):
        for name, value in _load_weights(weight_file).items():
            mapped = _mapped_name(name)
            if mapped is not None:
                converted[mapped] = value

    if "output.weight" not in converted:
        converted["output.weight"] = converted["tok_embeddings.weight"]

    for q_name in [name for name in converted if name.endswith("attention.wq.weight")]:
        k_name = q_name.replace("wq.weight", "wk.weight")
        v_name = q_name.replace("wq.weight", "wv.weight")
        q, k, v = converted.pop(q_name), converted.pop(k_name), converted.pop(v_name)
        if model_type != "qwen3":
            q = _permute(q, heads, head_dim)
            k = _permute(k, kv_heads, head_dim)
        converted[q_name.replace("wq.weight", "wqkv.weight")] = torch.cat((q, k, v))

    for q_name in [name for name in converted if name.endswith("attention.wq.bias")]:
        k_name = q_name.replace("wq.bias", "wk.bias")
        v_name = q_name.replace("wq.bias", "wv.bias")
        q, k, v = converted.pop(q_name), converted.pop(k_name), converted.pop(v_name)
        if model_type != "qwen3":
            q = _permute(q, heads, head_dim)
            k = _permute(k, kv_heads, head_dim)
        converted[q_name.replace("wq.bias", "wqkv.bias")] = torch.cat((q, k, v))

    output = checkpoint_dir / "model.pth"
    torch.save(converted, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    args = parser.parse_args()
    output = convert(args.checkpoint_dir)
    print(f"Saved converted checkpoint to {output}")


if __name__ == "__main__":
    main()
