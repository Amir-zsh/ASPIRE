import functools
import hashlib
from pathlib import Path

import torch


FLASHINFER_COMMIT = "56ed54098411946525da17eca2bcfd7ec72a515e"
FLASHINFER_PATCH_SHA256 = (
    "ed1b3cc512a4003e49a909c5cd6bad487e21ad4731ecdc006908fc2b3babc71f"
)


def validate_flashinfer_patch(
    csrc_dir: str | Path | None = None,
    include_dir: str | Path | None = None,
) -> dict[str, str]:
    """Fail early when the required persistent-attention patch is absent."""

    if csrc_dir is None or include_dir is None:
        from flashinfer.jit import env

        csrc_dir = env.FLASHINFER_CSRC_DIR
        include_dir = env.FLASHINFER_INCLUDE_DIR
    roots = {"csrc": Path(csrc_dir), "include": Path(include_dir)}
    required = {
        ("csrc", "batch_attention.cu"): (
            "params[i].batch_idx_arr",
            "BatchPagedAttentionPersistent<64, 16",
        ),
        ("include", "flashinfer/attention/persistent.cuh"): (
            "AttentionVariant variant(params, work_idx, nullptr)",
        ),
        ("include", "flashinfer/attention/scheduler.cuh"): (
            "batch_idx_offset",
            "CTA_TILE_Q_SIZES[NUM_TASKS] = {64, 16}",
        ),
    }
    digest = hashlib.sha256()
    missing = []
    for (root_name, relative), patterns in required.items():
        path = roots[root_name] / relative
        if not path.is_file():
            missing.append(str(path))
            continue
        content = path.read_bytes()
        digest.update(relative.encode())
        digest.update(content)
        text = content.decode(errors="replace")
        missing.extend(pattern for pattern in patterns if pattern not in text)
    if missing:
        raise RuntimeError(
            "FlashInfer is missing the ASPIRE persistent-attention patch "
            f"({', '.join(missing)}). Run scripts/install_flashinfer.sh."
        )
    return {
        "expected_commit": FLASHINFER_COMMIT,
        "patch_sha256": FLASHINFER_PATCH_SHA256,
        "installed_sources_sha256": digest.hexdigest(),
    }


def compute_max_cta_tile_kv(
    head_dim_qk: int = 128,
    head_dim_vo: int = 128,
    dtype_bytes: int = 2,
    device: int = 0,
) -> int:
    """Max CTA_TILE_KV that FlashInfer FA2/persistent kernels may use.

    Mirrors the dispatch logic in ``flashinfer/attention/prefill.cuh``.
    Use as the pad multiple for dump-buffer kv_stride allocation so that
    the ``kv_idx < kv_len`` guard can be safely removed from JIT
    logits-transform callbacks.

    Why this matters
    ----------------
    Without the ``kv_idx < kv_len`` guard the JIT callback writes to every
    kv_idx in the last CTA tile, including up to CTA_TILE_KV − 1 positions
    past the true kv_len.  If the dump buffer's stride is not padded to at
    least this value those writes go out of bounds (silent memory corruption).
    Padding the stride alone is not sufficient — see the full safety contract
    documented at the top of the logits-transform bodies in
    ``aspire.backends.jit_kernels`` for the remaining invariants (-inf fill,
    narrowing on read, per-sample num_pages masking).
    """
    smem_per_sm = torch.cuda.get_device_properties(device).shared_memory_per_multiprocessor
    best = 0
    for cta_tile_q in (128, 64, 16):
        warps_q = 4 if cta_tile_q > 16 else 1
        warps_kv = 4 // warps_q
        mma_q = 2 if cta_tile_q > 64 else 1
        smem_min = (
            cta_tile_q * head_dim_qk
            + (head_dim_qk + head_dim_vo) * 16 * warps_kv
        ) * dtype_bytes
        smem_per_block = smem_per_sm // (2 if smem_per_sm >= 2 * smem_min else 1)
        mma_kv_reg = 8 // mma_q
        mma_kv_smem = (smem_per_block - cta_tile_q * head_dim_qk * dtype_bytes) // (
            (head_dim_qk + head_dim_vo) * 16 * warps_kv * dtype_bytes
        )
        raw = min(mma_kv_smem, mma_kv_reg)
        mma_kv = next(t for t in (8, 4, 2, 1) if raw >= t)
        best = max(best, mma_kv * warps_kv * 16)
    return best


@functools.lru_cache(maxsize=None)
def _resolve_flashinfer_attention_backend(device_index: int, dtype: torch.dtype) -> str:
    return "fa2"


def resolve_flashinfer_attention_backend(dtype: torch.dtype) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("FlashInfer attention backend resolution requires CUDA.")

    return _resolve_flashinfer_attention_backend(torch.cuda.current_device(), dtype)
