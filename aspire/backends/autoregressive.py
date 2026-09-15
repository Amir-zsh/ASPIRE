"""Autoregressive backend: dense decoding, one token per request per step.

Every step decodes with full-context attention (the unified wrapper's verify
path); there is no drafting, no sparse context, and no attention-dump
wrapper or JIT kernel, which also saves the dump-buffer memory.

The shared base class maintains per-request draft page tables as part of its
step preparation, so this backend supplies a trivial position-only selector
(first sink pages + most recent pages). With draft length fixed at zero those
tables are never attended; the selector only keeps table maintenance cheap
and free of attention-score plumbing.
"""

from aspire.backends.base import LMBackendBase
from aspire.backends.page_selector import PageSelector
from aspire.models.sparse_draft_runner import _profile_scope

import torch


class _SinkRecentPageSelector(PageSelector):
    """Position-only page selector: first S sink pages + last L recent pages."""

    def __init__(self, *args, num_sink_pages=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_sink_pages = num_sink_pages
        # Pre-allocate gather indices [max_batch_size, subset_cache_size]
        self._gather_idx = torch.empty(
            self._scores.shape[0], self.subset_cache_size,
            dtype=torch.int64, device=self.device,
        )
        # Sink part is constant: [0, 1, ..., num_sink_pages-1]
        sink = min(self._num_sink_pages, self.subset_cache_size)
        self._gather_idx[:, :sink] = torch.arange(sink, device=self.device)
        self._sink_count = sink
        self._recent_offsets = torch.arange(
            self.subset_cache_size - sink, device=self.device,
        )

    def _select_pages(self, num_pages, block_table):
        B = num_pages.shape[0]
        k = self.subset_cache_size
        sink = self._sink_count
        recent = k - sink  # = L

        np_long = num_pages.long()
        bt = block_table.to(torch.int32)

        # Build gather indices: [sink_indices | recent_indices]
        idx = self._gather_idx[:B]
        # Sink part already filled at init; recent part varies per request
        if recent > 0:
            recent_starts = (np_long - recent).clamp_(min=sink)
            idx[:, sink:k] = recent_starts[:, None] + self._recent_offsets[None, :]
        idx.clamp_(min=0, max=bt.shape[1] - 1)

        # Gather page IDs
        selected = self._get_selected_buffer(B)
        torch.gather(bt, 1, idx, out=selected)

        # Valid counts: min(num_pages, k)
        valid_counts = self._get_valid_counts_buffer(B)
        valid_counts[:] = np_long.clamp(max=k).to(torch.int32)

        # Zero invalid entries for short sequences
        col_ids = self._col_ids[None, :]
        selected[col_ids >= valid_counts[:, None]] = 0

        return selected, valid_counts

    def select(self, attention_weights_buffer, lse_buffer,
               num_pages, block_table, qo_indptr):
        return self._select_pages(num_pages, block_table)

    def select_subset(self, slot_ids, attention_weights_buffer, lse_buffer,
                      num_pages, block_table, qo_indptr):
        if slot_ids.numel() == 0:
            empty = block_table.new_empty((0, self.subset_cache_size), dtype=torch.int32)
            return empty, torch.zeros(0, dtype=torch.int32, device=self.device)
        return self._select_pages(num_pages[slot_ids], block_table[slot_ids])


class LMBackendAutoregressive(LMBackendBase):
    """Dense autoregressive decoding on the unified paged-KV runtime."""

    _host_plan_draft_kind = "sink_recent"
    _host_plan_full_kind = None

    def setup_caches(self, *, top_k_pages: int = 16, page_size: int = 1, **kwargs):
        # Derive the recent-page count from top_k_pages; ignore caller's L
        kwargs.pop("L", None)
        # Hardcode sink size: at least 4 tokens worth of sink pages
        self._num_sink_pages = max(4, page_size) // page_size
        L = top_k_pages - self._num_sink_pages
        assert L > 0, (
            f"AR backend requires top_k_pages ({top_k_pages}) > "
            f"num_sink_pages ({self._num_sink_pages}) so that L = {L} > 0"
        )
        # Call base which calls our overridden hooks (all no-ops for dump)
        super().setup_caches(top_k_pages=top_k_pages, L=L, page_size=page_size, **kwargs)

    def _make_page_selector(self, *, max_batch_size, max_seq_length, page_size):
        return _SinkRecentPageSelector(
            max_batch_size=max_batch_size,
            pages_per_slot=self.max_num_pages_per_request,
            subset_cache_size=self.max_draft_pages,
            L=self.L,
            n_heads=self.model.config.n_head,
            page_size=page_size,
            max_seq_length=max_seq_length,
            device=self.device,
            bsz=max_batch_size,
            dec_len=self.dec_len,
            head_dim=self.model.config.head_dim,
            kv_stride=getattr(self, '_kv_stride_alloc', None),
            draft_percent=self._draft_percent,
            min_draft_pages=self._top_k_pages,
            num_sink_pages=self._num_sink_pages,
        )

    # -- Skip all dump infrastructure --

    def _allocate_dump_state(self, *, max_batch_size, max_seq_length):
        """Minimal — only set _kv_stride_alloc (used by page selector)."""
        from aspire.attention.flashinfer import compute_max_cta_tile_kv
        head_dim = self.model.config.head_dim
        pad = compute_max_cta_tile_kv(
            head_dim_qk=head_dim, head_dim_vo=head_dim,
            dtype_bytes=self.dtype.itemsize,
        )
        self._kv_stride_alloc = ((max_seq_length + pad - 1) // pad) * pad

    def _setup_unified_full_wrapper(self, *, max_batch_size, max_seq_length,
                                     page_size, use_cuda_graph, deterministic_attention):
        """No-op: no dump wrapper. Create dummy attrs for dynamo.mark_static_address."""
        # Base setup_caches calls dynamo.mark_static_address on these after this method
        self.unified_full_qo_indptr = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.unified_full_paged_kv_indices = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.unified_full_paged_kv_indptr = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.unified_full_paged_kv_last_page_len = torch.zeros(1, dtype=torch.int32, device=self.device)

    def _bind_dump_attention_ops(self, weight_dump_layer):
        """No-op: all layers use unified_wrapper."""
        pass

    def _plan_step_dump_wrapper(self, *, bsz, verify_num_pages, last_page_len, mode_flags):
        """No-op."""
        pass

    def reset_attention_weights_buffer(self):
        """No-op."""
        pass

    def _after_prepare_step(self, mode_flags, bsz):
        """No-op: no dump mode flags to update."""
        pass

    # -- Override replan to not reference dump buffers --

    def _whole_batch_replan_impl(self):
        """Position-only replan body: no attention-weight inputs. The base
        dispatcher (and its CUDA-graph capture) is shared unchanged."""
        selected, valid_counts = self.page_selector.select(
            None,  # attention_weights_buffer (ignored by position-only selector)
            None,  # lse_buffer (ignored)
            self.num_pages_per_request,
            self.target_block_table,
            None,  # qo_indptr (ignored)
        )
        self._store_draft_rows(
            None,
            selected,
            valid_counts,
            count_buffer=self._draft_num_pages_per_request,
            full_prefix=self.page_selector._bsz,
        )

    def _replan_draft_indices(self):
        """Replan via the shared base dispatcher, except the partial-batch
        fallback, which must pass the position-only selector's argument shape."""
        bsz = int(getattr(self, "_current_replan_bsz", self.batch_size))
        if bsz == self.page_selector._bsz:
            super()._replan_draft_indices()
            return
        slot_ids = torch.arange(bsz, device=self.device)
        with _profile_scope("REPLAN_SELECT"):
            selected, valid_counts = self.page_selector.select_subset(
                slot_ids,
                None,  # attention_weights_buffer (ignored by position-only selector)
                None,  # lse_buffer (ignored)
                self.num_pages_per_request,
                self.target_block_table,
                None,  # qo_indptr (ignored)
            )
        self._store_draft_rows(
            slot_ids,
            selected,
            valid_counts,
            count_buffer=self._draft_num_pages_per_request,
        )


__all__ = ["LMBackendAutoregressive"]
