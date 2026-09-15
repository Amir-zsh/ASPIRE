"""Transformer with a unified forward pass over mixed draft/verify batches.

The KV cache is one buffer holding target pages followed by draft pages, so a
single attention call can read either cache per request.
"""
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist

from aspire.models.transformer import (
    Transformer as BaseTransformer,
    TransformerBlock as BaseTransformerBlock,
    Attention as BaseAttention,
    KVCache as BaseKVCache,
    ModelArgs, FeedForward, RMSNorm, transformer_configs, find_multiple
)


class KVCache(BaseKVCache):
    """One buffer for target and draft pages; kv_cache/draft_cache are views into it."""
    def __init__(self, max_num_pages, page_size, n_heads, head_dim, dtype=torch.bfloat16, spec=False, draft_max_num_pages=0):
        nn.Module.__init__(self)
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.draft_max_num_pages = draft_max_num_pages
        self.is_spec = spec
        import torch._dynamo as dynamo

        if spec:
            # Layout: [target pages 0..max_num_pages-1][draft pages max_num_pages..total-1]
            total_pages = max_num_pages + draft_max_num_pages
            unified_shape = (total_pages, 2, page_size, n_heads, head_dim)
            self.register_buffer('_unified_kv_cache', torch.zeros(unified_shape, dtype=dtype))
            dynamo.mark_static_address(self._unified_kv_cache)

            self.kv_cache = self._unified_kv_cache[:max_num_pages]
            self.draft_cache = self._unified_kv_cache[max_num_pages:]
            dynamo.mark_static_address(self.kv_cache)
            dynamo.mark_static_address(self.draft_cache)
        else:
            cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
            self.register_buffer('kv_cache', torch.zeros(cache_shape, dtype=dtype))
            dynamo.mark_static_address(self.kv_cache)

    @property
    def unified_cache(self):
        """The full buffer (target + draft pages); a view, never a copy."""
        if self.is_spec:
            return self._unified_kv_cache
        return self.kv_cache


class Attention(BaseAttention):
    """Attention with unified_forward() for mixed draft/verify batches."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.unified_out = None

    def unified_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor,
                        target_page_indices: Tensor, target_page_indptr: Tensor, target_page_lastlen: Tensor,
                        draft_page_indices: Tensor, draft_page_indptr: Tensor, draft_page_lastlen: Tensor) -> Tensor:
        """
        Unified forward pass supporting mixed draft/verify batches.

        All tokens are written to BOTH caches. For draft samples, the target cache write
        is temporary and will be overwritten on next verify (since cachelens isn't incremented).
        This avoids conditionals and enables torch.compile compatibility.

        Args:
            x: Input tensor of shape (total_tokens, dim)
            offsets: Starting position for each sample (batch_size,)
            kv_append_indptr: Token count indptr (batch_size + 1,) - same for both caches
            target_page_*: Page table for target cache
            draft_page_*: Page table for draft cache
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

        # Write ALL tokens to BOTH caches (no conditionals)
        # Draft samples' writes to target cache are temporary garbage (overwritten on next verify)
        self.kv_cache.update_target(k, v, kv_append_indptr,
                                    target_page_indices, target_page_indptr, target_page_lastlen)
        self.kv_cache.update_draft(k, v, kv_append_indptr,
                                   draft_page_indices, draft_page_indptr, draft_page_lastlen)

        # Attention uses unified cache, writes into preallocated output buffer.
        attn_out = self.unified_out
        if attn_out is None or attn_out.shape[0] < total_tokens:
            attn_out = torch.empty_like(q)
        else:
            attn_out = attn_out[:total_tokens]
        self.attn_unified(q, self.kv_cache.unified_cache, attn_out)
        y = attn_out
        y = y.contiguous().view(total_tokens, self.q_proj_size)
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y


class TransformerBlock(BaseTransformerBlock):
    """TransformerBlock with unified_forward()."""

    def __init__(self, config: ModelArgs) -> None:
        nn.Module.__init__(self)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def unified_forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor,
                        target_page_indices: Tensor, target_page_indptr: Tensor, target_page_lastlen: Tensor,
                        draft_page_indices: Tensor, draft_page_indptr: Tensor, draft_page_lastlen: Tensor) -> Tensor:
        h = x + self.attention.unified_forward(
            self.attention_norm(x), offsets, kv_append_indptr,
            target_page_indices, target_page_indptr, target_page_lastlen,
            draft_page_indices, draft_page_indptr, draft_page_lastlen,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(BaseTransformer):
    """Transformer with unified_forward()."""

    def __init__(self, config: ModelArgs) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.world_size = None
        self.rank = None
        self.process_group = None
        self.banned_token_ids = None

    def setup_caches(self, num_pages, page_size, spec=False, draft_num_pages = 0, draft_budget = 0, window_size = 32, max_unified_tokens: int = None):
        """Allocate the unified KV cache and register the attention ops."""
        head_dim = self.config.dim // self.config.n_head
        dtype = self.output.weight.dtype if self.output.weight.dtype == torch.float16 else torch.bfloat16

        # Define rope operations (in-place)
        if (self.config.high_freq_factor is not None) and (self.config.low_freq_factor is not None):
            torch.library.define(
                "mylib::rope",
                "(Tensor(a!) q, Tensor(b!) k, Tensor indptr, Tensor offsets) -> ()",
            )
            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                import flashinfer.rope
                if q.numel() == 0:
                    return
                flashinfer.rope.apply_llama31_rope_inplace(
                    q, k, indptr, offsets, interleave=True, rope_scale=self.config.scaling_factor,
                    rope_theta=self.config.rope_base, low_freq_factor=self.config.low_freq_factor,
                    high_freq_factor=self.config.high_freq_factor,
                    old_context_len=self.config.original_max_position_embeddings
                )

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return None
        else:
            torch.library.define(
                "mylib::rope",
                "(Tensor(a!) q, Tensor(b!) k, Tensor indptr, Tensor offsets) -> ()",
            )
            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                import flashinfer.rope
                if q.numel() == 0:
                    return
                flashinfer.rope.apply_rope_inplace(
                    q, k, indptr, offsets, interleave=True,
                    rope_scale=self.config.scaling_factor, rope_theta=self.config.rope_base
                )

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return None

        for b in self.layers:
            b.attention.kv_cache = KVCache(num_pages, page_size, self.config.n_local_heads, head_dim, dtype, spec, draft_num_pages)
            b.attention.attn_decode = torch.ops.mylib.target_decode
            b.attention.attn_prefill = torch.ops.mylib.target_prefill
            b.attention.rope = torch.ops.mylib.rope
            if spec:
                b.attention.attn_draft = torch.ops.mylib.draft_decode
                b.attention.attn_unified = torch.ops.mylib.unified_decode
                if max_unified_tokens is not None:
                    b.attention.unified_out = torch.empty(
                        max_unified_tokens, self.config.n_head, head_dim,
                        dtype=dtype, device=self.output.weight.device
                    )
                b.attention.is_spec = True
                b.attention.draft_budget = draft_budget
                b.attention.window_size = window_size
                b.attention.pooling = 'avgpool'
                b.attention.kernel_size = 5

    def unified_forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor,
                        target_page_indices: Tensor, target_page_indptr: Tensor, target_page_lastlen: Tensor,
                        draft_page_indices: Tensor, draft_page_indptr: Tensor, draft_page_lastlen: Tensor) -> Tensor:
        """
        Unified forward pass supporting mixed draft/verify batches.

        All tokens are written to BOTH caches. For draft samples, the target cache write
        is temporary and will be overwritten on next verify (since cachelens isn't incremented).

        Args:
            idx: Token ids of shape (total_tokens,)
            input_pos: Starting position for each sample (batch_size,)
            kv_append_indptr: Token count indptr (batch_size + 1,)
            target_page_*: Page table for target cache
            draft_page_*: Page table for draft cache
        """
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.unified_forward(
                x, input_pos, kv_append_indptr,
                target_page_indices, target_page_indptr, target_page_lastlen,
                draft_page_indices, draft_page_indptr, draft_page_lastlen,
            )
        x = self.norm(x)
        logits = self.output(x)
        if self.process_group != None:
            total_tokens = x.shape[0]
            all_max_value = torch.zeros((total_tokens, self.world_size), dtype=logits.dtype, device=logits.device)
            all_max_indices = torch.zeros((total_tokens, self.world_size), dtype=torch.long, device=logits.device)
            all_max_value[:, self.rank], all_max_indices[:, self.rank] = torch.max(logits, dim=-1)
            all_max_indices[:, self.rank] += self.rank * logits.shape[-1]
            dist.all_reduce(all_max_value, group=self.process_group)
            dist.all_reduce(all_max_indices, group=self.process_group)
            global_select_indices = torch.argmax(all_max_value, dim=-1)
            global_indices = torch.gather(all_max_indices, dim=-1, index=global_select_indices.unsqueeze(-1))
            return global_indices.squeeze(-1)
        return torch.argmax(logits, dim=-1)
