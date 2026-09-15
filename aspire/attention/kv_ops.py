"""Torch custom operators for paged KV-cache writes."""

from __future__ import annotations

import flashinfer
import flashinfer.triton.page as flashinfer_page
import torch
import triton
import triton.language as tl


torch.library.define(
    "mylib::update_kv",
    "(Tensor k, Tensor v, Tensor kv_append_indptr, Tensor(a!) kv_cache, "
    "Tensor kv_page_indices, Tensor kv_page_indptr, Tensor kv_last_page_len, "
    "int page_size) -> ()",
)


@torch.library.impl("mylib::update_kv", "cuda")
def update_kv(
    k,
    v,
    kv_append_indptr,
    kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    page_size,
):
    seq_lens = flashinfer.get_seq_lens(
        kv_page_indptr, kv_last_page_len, page_size
    )
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr, seq_lens, k.shape[0]
    )
    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )


@torch.library.register_fake("mylib::update_kv")
def update_kv_abstract(
    k,
    v,
    kv_append_indptr,
    kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    page_size,
):
    return None


@triton.jit
def _append_paged_kv_cache_padded_kernel(
    append_key_ptr,
    append_value_ptr,
    batch_indices_ptr,
    positions_ptr,
    kv_cache_ptr,
    kv_page_indices_ptr,
    kv_page_indptr_ptr,
    kv_last_page_len_ptr,
    stride_k_n,
    stride_k_h,
    stride_v_n,
    stride_v_h,
    stride_cache_p,
    stride_cache_kv,
    stride_cache_t,
    stride_cache_h,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.load(batch_indices_ptr + token_idx)
    pos = tl.load(positions_ptr + token_idx)
    start_page = tl.load(kv_page_indptr_ptr + batch_idx)
    end_page = tl.load(kv_page_indptr_ptr + batch_idx + 1)
    last_len = tl.load(kv_last_page_len_ptr + batch_idx)
    page_count = end_page - start_page
    if pos < 0:
        return
    if page_count == 0:
        return
    length = (page_count - 1) * PAGE_SIZE + last_len
    if pos >= length:
        return

    abs_pos = start_page * PAGE_SIZE + pos
    page_iter = abs_pos // PAGE_SIZE
    entry_idx = abs_pos - page_iter * PAGE_SIZE
    phys_page = tl.load(kv_page_indices_ptr + page_iter)
    offsets = tl.arange(0, HEAD_DIM)
    k = tl.load(
        append_key_ptr
        + token_idx * stride_k_n
        + head_idx * stride_k_h
        + offsets
    )
    v = tl.load(
        append_value_ptr
        + token_idx * stride_v_n
        + head_idx * stride_v_h
        + offsets
    )
    base = (
        phys_page * stride_cache_p
        + entry_idx * stride_cache_t
        + head_idx * stride_cache_h
        + offsets
    )
    tl.store(kv_cache_ptr + base, k)
    tl.store(kv_cache_ptr + base + stride_cache_kv, v)


@torch.library.triton_op("mylib::update_kv_padded", mutates_args={"kv_cache"})
def update_kv_padded(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_append_indptr: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
) -> None:
    total_tokens = k.shape[0]
    if total_tokens == 0:
        return
    seq_lens = flashinfer.get_seq_lens(
        kv_page_indptr, kv_last_page_len, page_size
    )
    batch_indices = torch.zeros(total_tokens, device=k.device, dtype=torch.int32)
    positions = torch.full(
        (total_tokens,), -1, device=k.device, dtype=torch.int32
    )
    torch.library.wrap_triton(
        flashinfer_page.get_batch_indices_positions_kernel
    )[(seq_lens.shape[0],)](
        kv_append_indptr,
        seq_lens,
        batch_indices,
        positions,
        num_stages=2,
    )

    head_dim = k.shape[2]
    num_heads = k.shape[1]
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("update_kv_padded requires contiguous k/v head dimensions")
    if k.stride(1) != head_dim or v.stride(1) != head_dim:
        raise ValueError("update_kv_padded requires contiguous k/v heads")
    if kv_cache.stride(-1) != 1:
        raise ValueError("update_kv_padded requires a contiguous cache head dimension")
    torch.library.wrap_triton(_append_paged_kv_cache_padded_kernel)[
        (total_tokens, num_heads)
    ](
        k,
        v,
        batch_indices,
        positions,
        kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
    )
