"""ASPIRE backend: sparse-context drafting with intra-draft refresh. Page
selection scores come from the first and last verify-token attention dumps.
"""

from aspire.backends.base import LMBackendBase
from aspire.backends.page_selector import PageSelectorFirstLastAvg


class LMBackendAspire(LMBackendBase):
    """Speculative backend using first/last verify-token dumps for page selection."""

    def _dump_query_slots(self) -> int:
        """Override to allocate two dump slots per head: first token and last token."""
        return 2

    def _dump_kind(self) -> str:
        """Override to select the compact first/last JIT kernel variant."""
        return "first_last_compact"

    def _dump_jit_name_suffix(self) -> str:
        """Override so the first/last JIT kernel is compiled under its own symbol."""
        return "FirstLastCompact"

    def _dump_torch_op_name(self) -> str:
        """Override to bind the dump layer to the first/last-specific Torch op name."""
        return "mylib::unified_full_decode_first_last"

    def _make_page_selector(self, *, max_batch_size: int, max_seq_length: int, page_size: int):
        """Override to score pages from averaged first/last verify-token dumps."""
        return PageSelectorFirstLastAvg(
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
        )


