"""Sparse-draft model: the draft cache holds only the top-k compressed
tokens, chosen once at prefill; the recent window is read from the target
cache, so decode steps write to the target cache only.
"""
from dataclasses import dataclass
from typing import Optional
import os
from contextlib import nullcontext

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
try:
    from torch.profiler import record_function
except Exception:  # pragma: no cover - profiler may be unavailable in minimal builds
    record_function = None

from aspire.models.mixed_transformer import (
    Attention, TransformerBlock, Transformer, KVCache,
    ModelArgs, RMSNorm, FeedForward, find_multiple
)
from aspire.engine.sampling import local_topk_candidates, merge_topk_candidates


class _NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)
        return None

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        return False


def _profile_scope(name: str):
    if os.getenv("ASPIRE_NVTX") == "1":
        return _NvtxRange(name)
    if os.getenv("ASPIRE_PROFILE") == "1" and record_function is not None:
        return record_function(name)
    return nullcontext()


class AttentionSparseDraft(Attention):
    """Attention whose draft cache stores only the top-k compressed tokens;
    draft attention reads the recent window from the target cache."""

    @property
    def topk(self):
        """Compressed tokens kept in the draft cache (draft_budget - window_size)."""
        return self.draft_budget - self.window_size

    def prefill(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor,
                kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor,
                is_last=False, draft_paged_kv_indptr=None, draft_paged_kv_indices=None,
                draft_paged_kv_last_page_len=None) -> Tensor:
        """Prefill the target cache and, on the last chunk, build the top-k draft cache."""
        bsz, seqlen, _ = x.shape
        q, k, v = self._qkv_proj(x, bsz, seqlen)
        self.rope(q, k, kv_append_indptr, offsets)
        kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        y = torch.empty_like(q)
        self.attn_prefill(q, kv_cache, y)
        if is_last and self.is_spec:
            # Draft-cache append indptr is scaled by topk (window is not stored).
            self.gen_draft_kv(
                q, kv_cache[:, 0], kv_cache[:, 1], bsz, seqlen, offsets[0] + seqlen,
                (kv_append_indptr / seqlen * self.topk).to(torch.int32),
                draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len
            )
        y = y.contiguous().view(bsz, seqlen, self.q_proj_size)
        y = self.wo(y)
        if self.process_group is not None:
            dist.all_reduce(y, group=self.process_group)
        return y

    def gen_draft_kv(self, q, k, v, bsz, seqlen, context_len, kv_append_indptr,
                     draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len):
        """Select the top-k context tokens by pooled attention weight of the
        window queries and write their K/V into the draft cache."""
        query_states = q.reshape(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        key_states = k.reshape(bsz, -1, self.n_local_heads, self.head_dim)[:, :context_len].transpose(1, 2).contiguous()
        value_states = v.reshape(bsz, -1, self.n_local_heads, self.head_dim)[:, :context_len].transpose(1, 2).contiguous()

        nrepeat = self.n_head // self.n_local_heads
        query_states = rearrange(query_states, 'b (h r) l d -> b h (r l) d', r=nrepeat).contiguous()

        B, H, L, D = query_states.shape
        topk = self.draft_budget - self.window_size

        mask = torch.full((self.window_size, self.window_size), torch.finfo(query_states.dtype).min, device=query_states.device)
        mask_cond = torch.arange(mask.size(-1), device=query_states.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(query_states.device)

        window_size = self.window_size
        chunk_size = nrepeat * 8  # to support qwen
        assert chunk_size % nrepeat == 0
        num_chunks = (L + chunk_size - 1) // chunk_size
        attn_weights_sum = torch.zeros(bsz, self.n_head, context_len - window_size, device=query_states.device, dtype=query_states.dtype)

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, L)
            chunk_query_states = query_states[:, :, start_idx:end_idx, :]
            attn_weights = torch.einsum('b h i d, b h j d -> b h i j', chunk_query_states, key_states)
            attn_weights[:, :, -window_size:, -window_size:] += mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = rearrange(attn_weights, 'b h (r l) s -> b (h r) l s', r=nrepeat)
            attn_weights_sum += attn_weights[..., :-window_size].sum(dim=-2)

        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        # avg across each kv group
        attn_cache = rearrange(attn_cache, 'b (h r) s -> b h r s', h=self.n_local_heads)
        attn_cache = attn_cache.sum(dim=2)

        indices = attn_cache.topk(topk, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

        new_key_states = k_past_compress.transpose(1, 2).contiguous().view(-1, self.n_local_heads, self.head_dim)
        new_value_states = v_past_compress.transpose(1, 2).contiguous().view(-1, self.n_local_heads, self.head_dim)

        self.kv_cache.update_draft(new_key_states, new_value_states, kv_append_indptr,
                                   draft_paged_kv_indices, draft_paged_kv_indptr, draft_paged_kv_last_page_len)

    def unified_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor,
                        target_page_indices: Tensor, target_page_indptr: Tensor, target_page_lastlen: Tensor) -> Tensor:
        """Mixed draft/verify forward over a flattened token batch.

        New K/V is written to the target cache; the pre-planned page indices
        decide what each row attends to (draft rows: draft pages + recent
        target pages, verify rows: target pages only).
        """
        total_tokens = x.shape[0]
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.q_proj_size, kv_size, kv_size], dim=-1)
        if self.use_qk_norm:
            q = q.view(1, total_tokens, self.n_head, self.head_dim)
            k = k.view(1, total_tokens, self.n_local_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = self._permute_for_rope(q)
            k = self._permute_for_rope(k)
        q = q.reshape(total_tokens, self.n_head, self.head_dim)
        k = k.reshape(total_tokens, self.n_local_heads, self.head_dim)
        v = v.contiguous().view(total_tokens, self.n_local_heads, self.head_dim)
        self.rope(q, k, kv_append_indptr, offsets)

        # Write and attend through the same unified_cache tensor (CUDA-graph safe).
        unified_cache = self.kv_cache.unified_cache
        torch.ops.mylib.update_kv_padded(
            k, v, kv_append_indptr,
            unified_cache,
            target_page_indices, target_page_indptr, target_page_lastlen,
            self.kv_cache.page_size,
        )
        y = torch.empty_like(q)
        self.attn_unified(q, unified_cache, y)
        y = y.contiguous().view(total_tokens, self.q_proj_size)
        y = self.wo(y)
        if self.process_group is not None:
            dist.all_reduce(y, group=self.process_group)
        return y



class TransformerBlockSparseDraft(TransformerBlock):
    """TransformerBlock with AttentionSparseDraft."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.attention = AttentionSparseDraft(config)

    def unified_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor,
                        target_page_indices: Tensor, target_page_indptr: Tensor, target_page_lastlen: Tensor) -> Tensor:
        with _profile_scope("ATTN_NORM"):
            attn_in = self.attention_norm(x)
        with _profile_scope("ATTENTION"):
            attn_out = self.attention.unified_forward(
                attn_in, offsets, kv_append_indptr,
                target_page_indices, target_page_indptr, target_page_lastlen,
            )
        h = x + attn_out
        with _profile_scope("FFN_NORM"):
            ffn_in = self.ffn_norm(h)
        with _profile_scope("FFN"):
            ffn_out = self.feed_forward(ffn_in)
        out = h + ffn_out
        return out


class TransformerSparseDraft(Transformer):
    """Transformer built from TransformerBlockSparseDraft layers."""

    def __init__(self, config: ModelArgs) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlockSparseDraft(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.world_size = None
        self.rank = None
        self.process_group = None
        self.banned_token_ids = None
        self.top_k = None  # set by backend.setup_caches(top_k=...)
        self.logit_process = None  # set by backend.set_logit_processing()

    def unified_forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor,
                        target_page_indices: Tensor, target_page_indptr: Tensor, target_page_lastlen: Tensor) -> Tensor:
        """Mixed draft/verify forward; returns (probs, ids) of the top-k
        candidates, or argmax ids when top_k == 0."""
        with _profile_scope("EMBED"):
            x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer.unified_forward(
                x, input_pos, kv_append_indptr,
                target_page_indices, target_page_indptr, target_page_lastlen,
            )
        with _profile_scope("FINAL_NORM"):
            x = self.norm(x)
        with _profile_scope("OUTPUT"):
            logits = self.output(x)
        with _profile_scope("TOPK"):
            if self.top_k > 0:
                if self.process_group is not None:
                    # Local top-k per rank, then all_gather and merge.
                    B = logits.shape[0]
                    local_vals, local_ids = local_topk_candidates(
                        logits,
                        self.top_k,
                        vocab_start=self.rank * logits.shape[-1],
                    )
                    gathered_values = [tensor[:B] for tensor in self._gather_vals]
                    gathered_indices = [tensor[:B] for tensor in self._gather_ids]
                    dist.all_gather(gathered_values, local_vals, group=self.process_group)
                    dist.all_gather(gathered_indices, local_ids, group=self.process_group)
                    all_vals = self._gather_merged_vals[:B]
                    all_ids = self._gather_merged_ids[:B]
                    for rank in range(self.world_size):
                        start = rank * self.top_k
                        end = start + self.top_k
                        all_vals[:, start:end].copy_(gathered_values[rank])
                        all_ids[:, start:end].copy_(gathered_indices[rank])
                    vals, ids = merge_topk_candidates(all_vals, all_ids, self.top_k)
                else:
                    vals, ids = local_topk_candidates(logits, self.top_k)
                probs = self.logit_process(vals)
                return probs, ids
            else:
                if self.process_group is not None:
                    # Local argmax per rank, then all_reduce.
                    B = logits.shape[0]
                    all_max_value = torch.zeros(
                        (B, self.world_size), dtype=logits.dtype, device=logits.device)
                    all_max_indices = torch.zeros(
                        (B, self.world_size), dtype=torch.long, device=logits.device)
                    all_max_value[:, self.rank], all_max_indices[:, self.rank] = torch.max(logits, dim=-1)
                    all_max_indices[:, self.rank] += self.rank * logits.shape[-1]
                    dist.all_reduce(all_max_value, group=self.process_group)
                    dist.all_reduce(all_max_indices, group=self.process_group)
                    global_select = torch.argmax(all_max_value, dim=-1)
                    global_ids = torch.gather(all_max_indices, -1, global_select.unsqueeze(-1))
                    return global_ids.squeeze(-1)
                else:
                    return torch.argmax(logits, dim=-1)


    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))
