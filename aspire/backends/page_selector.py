import math

import torch
import torch.nn.functional as F


class PageSelector:

    def __init__(self, max_batch_size, pages_per_slot, subset_cache_size, L,
                 n_heads, page_size, max_seq_length, device, bsz, dec_len,
                 head_dim=128, kv_stride=None, draft_percent=None,
                 min_draft_pages=None):
        self.subset_cache_size = subset_cache_size  # max draft pages (allocation cap)
        self.L = L                                   # last pages always kept
        self._min_draft_pages = min_draft_pages      # runtime floor (top_k_pages from CLI)
        self.n_heads = n_heads
        self.page_size = page_size
        self.max_seq_length = max_seq_length
        self._kv_stride = kv_stride or max_seq_length
        self.pages_per_slot = pages_per_slot
        self.device = device
        self._draft_percent = draft_percent

        self._bsz = bsz
        self._dec_len = dec_len
        sm_scale = 1.0 / math.sqrt(head_dim)
        self._sm_scale_log2e = sm_scale * math.log2(math.e)
        self._seq_pad = pages_per_slot * page_size - max_seq_length

        self._j = torch.arange(pages_per_slot, device=device)
        # Pre-allocate reusable buffers — reused every call.
        self._scores = torch.empty(max_batch_size, pages_per_slot,
                                   dtype=torch.float32, device=device)
        self._pos_bias = -torch.arange(
            pages_per_slot, device=device, dtype=torch.float32
        ) * 1e-7

        # Pre-allocate finalize buffers (for B <= max_batch_size; larger B falls back)
        k = subset_cache_size
        self._max_finalize_B = max_batch_size
        if k > 0:
            self._selected_buf = torch.empty(max_batch_size, k, dtype=torch.int32, device=device)
            self._col_ids = torch.arange(k, device=device, dtype=torch.int32)
        self._valid_counts_buf = torch.empty(max_batch_size, dtype=torch.int32, device=device)
        self._sentinel = torch.iinfo(torch.int32).max

        # Pre-allocate head_max buffer for _select_impl (zero-filled so padding region is 0)
        padded_seq = pages_per_slot * page_size
        self._head_max_buf = torch.zeros(max_batch_size, padded_seq,
                                         dtype=torch.float32, device=device)

    def compute_valid_counts(self, num_pages: torch.Tensor) -> torch.Tensor:
        """Per-request draft page count."""
        hi = num_pages.clamp(max=self.subset_cache_size).to(torch.int32)
        if self._draft_percent is None:
            if self._min_draft_pages is not None:
                return num_pages.clamp(max=self._min_draft_pages).to(torch.int32)
            return hi
        floor = self._min_draft_pages if self._min_draft_pages is not None else self.L
        lo = num_pages.clamp(max=floor).to(torch.int32)
        target = (num_pages.float() * self._draft_percent / 100).ceil_().to(torch.int32)
        return torch.clamp(target, lo, hi)

    def _get_scores_buffer(self, batch_size):
        if batch_size <= self._scores.shape[0]:
            return self._scores[:batch_size]
        return torch.empty(batch_size, self.pages_per_slot, dtype=torch.float32, device=self.device)

    def _get_selected_buffer(self, B):
        if B <= self._max_finalize_B:
            return self._selected_buf[:B]
        return torch.empty(B, self.subset_cache_size, dtype=torch.int32, device=self.device)

    def _get_valid_counts_buffer(self, B):
        if B <= self._max_finalize_B:
            return self._valid_counts_buf[:B]
        return torch.empty(B, dtype=torch.int32, device=self.device)

    def _finalize_scores(self, scores, num_pages, block_table):
        B = scores.shape[0]
        np_long = num_pages.long()
        j = self._j[None]

        # Inplace on scores (we own this buffer): force last L pages to inf, invalid to -inf
        last_start = (np_long - self.L).clamp_(min=0)
        last_mask = (j >= last_start[:, None]) & (j < np_long[:, None])
        scores[last_mask] = float("inf")
        scores[j >= np_long[:, None]] = float("-inf")

        # Top-k selection
        _, topk_idx = scores.topk(self.subset_cache_size, dim=1)

        # Gather page IDs into pre-allocated buffer
        selected = self._get_selected_buffer(B)
        topk_idx.clamp_(max=block_table.shape[1] - 1)
        torch.gather(block_table.to(torch.int32), 1, topk_idx, out=selected)

        # Compute valid counts into pre-allocated buffer
        valid_counts = self._get_valid_counts_buffer(B)
        valid_counts[:] = self.compute_valid_counts(num_pages)

        # Sentinel sort: set invalid entries to sentinel, sort, then zero them
        valid_mask = self._col_ids[None, :] < valid_counts[:, None]
        selected[~valid_mask] = self._sentinel
        selected[:], _ = selected.sort(dim=1)
        selected[~valid_mask] = 0

        return selected, valid_counts

    def _select_impl(self, logits, lse, num_pages, block_table):
        B = logits.shape[0]
        Mmax = self.pages_per_slot
        seq_len = logits.shape[2]

        # Normalize and reduce: logits * scale - lse → exp2 → max over heads
        # NOT inplace — logits may be a view of the shared attention_weights_buffer
        normalized = logits * self._sm_scale_log2e - lse[:, :, None]
        normalized.nan_to_num_(nan=float("-inf"))
        # Max over heads → write into pre-allocated buffer (avoids F.pad allocation)
        head_max = self._head_max_buf[:B, :seq_len]
        torch.amax(normalized, dim=1, out=head_max)
        head_max.exp2_()
        # Use full padded width (padding region is 0 from init, never overwritten)
        head_max_padded = self._head_max_buf[:B]

        # Reshape to pages and sum tokens per page → scores
        scores = self._get_scores_buffer(B)
        scores[:] = head_max_padded.view(B, Mmax, self.page_size).sum(dim=-1)
        scores.add_(self._pos_bias)
        return self._finalize_scores(scores, num_pages, block_table)

    def _view_buffer(self, attention_weights_buffer, B, *extra_dims):
        """View the flat dump buffer with optional kv_stride padding, then narrow."""
        shape = (B, self.n_heads, *extra_dims, self._kv_stride)
        buf = attention_weights_buffer.view(*shape)
        if self._kv_stride != self.max_seq_length:
            buf = buf[..., :self.max_seq_length]
        return buf

    def select(self, attention_weights_buffer, lse_buffer,
               num_pages, block_table, qo_indptr):
        B = self._bsz
        logits = self._view_buffer(attention_weights_buffer, B)
        last_q_flat = (qo_indptr[1:B+1].long() - 1).clamp(min=0)
        lse = lse_buffer[last_q_flat]
        return self._select_impl(logits, lse, num_pages, block_table)

    def select_subset(self, slot_ids, attention_weights_buffer, lse_buffer,
                      num_pages, block_table, qo_indptr):
        if slot_ids.numel() == 0:
            empty = block_table.new_empty((0, self.subset_cache_size), dtype=torch.int32)
            return empty, torch.zeros(0, dtype=torch.int32, device=self.device)
        full_logits = self._view_buffer(attention_weights_buffer, self._bsz)
        logits = full_logits[slot_ids]
        last_q_flat = (qo_indptr[slot_ids.long() + 1].long() - 1).clamp(min=0)
        lse = lse_buffer[last_q_flat]
        return self._select_impl(logits, lse, num_pages[slot_ids], block_table[slot_ids])


class PageSelectorFirstLastAvg(PageSelector):
    """Page selector that averages compact first/last verify-token dump scores."""

    def _select_first_last_impl(self, logits, first_lse, last_lse, num_pages, block_table):
        B = logits.shape[0]
        Mmax = self.pages_per_slot
        # logits: [B, H, 2, seq_len], lse: [B, H] each

        # Normalize — NOT inplace (logits may be a view of a shared buffer)
        lse = torch.stack([first_lse, last_lse], dim=2)  # [B, H, 2]
        norm = logits * self._sm_scale_log2e - lse[:, :, :, None]  # [B, H, 2, seq_len]
        norm.nan_to_num_(nan=float("-inf"))
        # Max over heads, sum over slots → [B, seq_len]
        token_scores = torch.exp2(norm).amax(dim=1).sum(dim=1)

        if self._seq_pad > 0:
            token_scores = F.pad(token_scores, (0, self._seq_pad))

        scores = self._get_scores_buffer(B)
        scores[:] = token_scores.view(B, Mmax, self.page_size).sum(dim=-1)
        scores.add_(self._pos_bias)
        return self._finalize_scores(scores, num_pages, block_table)

    def select_from_compact_first_last(self, logits, first_lse, last_lse, num_pages, block_table):
        return self._select_first_last_impl(logits, first_lse, last_lse, num_pages, block_table)

    def select_layers_from_compact_first_last(self, logits, first_lse, last_lse, num_pages, block_table):
        n_layers, batch_size = logits.shape[:2]
        flat_logits = logits.reshape(n_layers * batch_size, self.n_heads, 2, self._kv_stride)
        if self._kv_stride != self.max_seq_length:
            flat_logits = flat_logits[..., :self.max_seq_length]
        flat_first_lse = first_lse.reshape(n_layers * batch_size, self.n_heads)
        flat_last_lse = last_lse.reshape(n_layers * batch_size, self.n_heads)
        flat_num_pages = num_pages.repeat(n_layers)
        flat_block_table = block_table.repeat(n_layers, 1)
        selected, flat_valid_counts = self._select_first_last_impl(
            flat_logits,
            flat_first_lse,
            flat_last_lse,
            flat_num_pages,
            flat_block_table,
        )
        # valid_counts is the same for all layers (same num_pages), take first layer's
        valid_counts = flat_valid_counts[:batch_size]
        return selected.view(n_layers, batch_size, self.subset_cache_size), valid_counts

    def select(self, attention_weights_buffer, lse_buffer, num_pages, block_table, qo_indptr):
        B = self._bsz
        logits = self._view_buffer(attention_weights_buffer, B, 2)
        max_q_idx = max(lse_buffer.shape[0] - 1, 0)
        first_q_flat = qo_indptr[:B].long().clamp(min=0, max=max_q_idx)
        last_q_flat = (qo_indptr[1 : B + 1].long() - 1).clamp(min=0, max=max_q_idx)
        first_lse = lse_buffer[first_q_flat]
        last_lse = lse_buffer[last_q_flat]
        return self._select_first_last_impl(logits, first_lse, last_lse, num_pages, block_table)

    def select_subset(self, slot_ids, attention_weights_buffer, lse_buffer, num_pages, block_table, qo_indptr):
        if slot_ids.numel() == 0:
            empty = block_table.new_empty((0, self.subset_cache_size), dtype=torch.int32)
            return empty, torch.zeros(0, dtype=torch.int32, device=self.device)
        full_logits = self._view_buffer(attention_weights_buffer, self._bsz, 2)
        logits = full_logits[slot_ids]
        max_q_idx = max(lse_buffer.shape[0] - 1, 0)
        first_q_flat = qo_indptr[slot_ids.long()].long().clamp(min=0, max=max_q_idx)
        last_q_flat = (qo_indptr[slot_ids.long() + 1].long() - 1).clamp(min=0, max=max_q_idx)
        first_lse = lse_buffer[first_q_flat]
        last_lse = lse_buffer[last_q_flat]
        return self._select_first_last_impl(
            logits,
            first_lse,
            last_lse,
            num_pages[slot_ids],
            block_table[slot_ids],
        )
