"""Runner for mixed draft/verify batches (unified forward)."""
import torch
import triton
import triton.language as tl
import flashinfer

from aspire.models.runner import LMBackend as BaseLMBackend

DRAFT_MODE = 0
VERIFY_MODE = 1


@triton.jit
def _build_kv_indices_kernel(
    paged_kv_indices,    # Output: [total_pages] flattened page indices
    block_table,         # Input: [batch_size, max_pages_per_req] page mapping
    block_table_stride,  # block_table.stride(0)
    cu_num_pages,        # Input: [batch_size + 1] cumulative page counts (indptr)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Build paged_kv_indices from block_table based on per-request page counts.

    Each Triton program handles one request:
    - Reads block_table[req_idx, 0:num_pages]
    - Writes to paged_kv_indices[start_idx:end_idx]

    Supports variable page counts per request via cu_num_pages (indptr).
    """
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_pages + req_idx)
    end_idx = tl.load(cu_num_pages + req_idx + 1)
    num_pages = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_pages, BLOCK_SIZE):
        mask = (i + offset) < num_pages
        block_ids = tl.load(row_ptr + i + offset, mask=mask)
        tl.store(paged_kv_indices + start_idx + i + offset, block_ids, mask=mask)


def build_kv_indices_from_block_table(
    block_table: torch.Tensor,
    num_pages_per_request: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
) -> None:
    """
    Build paged_kv_indices and paged_kv_indptr from block_table.

    Args:
        block_table: [batch_size, max_pages] - static page mapping
        num_pages_per_request: [batch_size] - current page counts per request
        paged_kv_indices: [total_max_pages] - output buffer (modified in-place)
        paged_kv_indptr: [batch_size+1] - output buffer (modified in-place)
    """
    batch_size = block_table.shape[0]

    paged_kv_indptr[0] = 0
    paged_kv_indptr[1:batch_size+1] = torch.cumsum(num_pages_per_request, dim=0)

    if batch_size > 0:
        _build_kv_indices_kernel[(batch_size,)](
            paged_kv_indices,
            block_table,
            block_table.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=128,
        )


class LMBackend(BaseLMBackend):
    """LMBackend with unified_inference() for mixed draft/verify batches."""

    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_len: int = 1, draft_dec_len: int = None) -> None:
        super().__init__(dtype, device, dec_len, draft_dec_len)
        if self.is_spec:
            self.unified_forward = lambda model, x, input_pos, kv_append_indptr, target_page_indices, target_page_indptr, target_page_lastlen, draft_page_indices, draft_page_indptr, draft_page_lastlen: model.unified_forward(x, input_pos, kv_append_indptr, target_page_indices, target_page_indptr, target_page_lastlen, draft_page_indices, draft_page_indptr, draft_page_lastlen)

    def load_model(self, checkpoints, model_name: str, use_tp: bool, rank_group=None, group=None):
        """Load the mixed-forward Transformer from a converted model.pth."""
        from aspire.models.mixed_transformer import Transformer

        with torch.device('meta'):
            self.model = Transformer.from_name(model_name)

        checkpoint = torch.load(str(checkpoints), mmap=True, weights_only=True)
        self.model.load_state_dict(checkpoint, assign=True)

        if use_tp:
            from aspire.tp import apply_tp
            apply_tp(self.model, rank_group, group=group)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        return self.model.eval()

    def _get_last_page_len(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute paged_kv_last_page_len from sequence lengths.

        Formula: (seq_len - 1) % page_size + 1
        This gives the number of tokens in the last page:
        - seq_len=1 → 1, seq_len=128 → 128, seq_len=129 → 1

        Special case: seq_len=0 returns 0 (no tokens, no pages)
        """
        return torch.where(
            seq_lens > 0,
            (seq_lens - 1) % self.page_size + 1,
            torch.zeros_like(seq_lens)
        )

    def _get_num_pages(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute number of pages needed for given sequence lengths.

        Formula: (seq_len + page_size - 1) // page_size = ceil(seq_len / page_size)
        """
        return (seq_lens + self.page_size - 1) // self.page_size

    def _update_target_page_table(self, cachelens: torch.Tensor):
        """
        Update TARGET cache page table state from cachelens.

        Uses block_table + Triton kernel for efficient index building.
        No Python loops, handles variable page counts per sample.

        Args:
            cachelens: Current sequence lengths per sample (target cache)
        """
        num_pages = self._get_num_pages(cachelens)
        last_page_len = self._get_last_page_len(cachelens)

        # Update the tracking tensor in place: the replan CUDA graph holds
        # this buffer's address, so rebinding the attribute to a fresh tensor
        # would leave the graph reading a dead copy.
        self.num_pages_per_request.copy_(num_pages.to(torch.int32))

        build_kv_indices_from_block_table(
            self.target_block_table,
            self.num_pages_per_request,
            self.paged_kv_indices,
            self.paged_kv_indptr,
        )

        self.paged_kv_last_page_len.copy_(last_page_len)

    def _update_draft_page_table(self, actual_tokens: torch.Tensor):
        """
        Update DRAFT cache page table state from actual token counts.

        Uses block_table + Triton kernel for efficient index building.
        Uses actual_tokens (not draft_cachelens) because draft cache stores
        draft_budget + N tokens, not prefix_len + N.

        Args:
            actual_tokens: Actual number of tokens in draft cache per sample
        """
        num_pages = self._get_num_pages(actual_tokens)
        last_page_len = self._get_last_page_len(actual_tokens)

        self.draft_num_pages_per_request = num_pages.to(torch.int32)

        build_kv_indices_from_block_table(
            self.draft_block_table,
            self.draft_num_pages_per_request,
            self.draft_paged_kv_indices,
            self.draft_paged_kv_indptr,
        )

        self.draft_paged_kv_last_page_len.copy_(last_page_len)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, draft_budget = 0, window_size = 32, page_size: int = 128):
        """Parent setup plus block tables and the unified attention wrapper."""
        super().setup_caches(max_batch_size, max_seq_length, draft_budget, window_size)
        self.page_size = page_size
        if self.is_spec:
            self.max_unified_tokens = max_batch_size * self.dec_len

        self.target_block_table = torch.arange(
            self.max_num_pages, dtype=torch.int32, device=self.device
        ).reshape(max_batch_size, self.max_num_pages_per_request)

        if self.is_spec:
            min_gen_tokens = 255
            draft_max_tokens = draft_budget + min_gen_tokens
            self.draft_max_num_pages_per_request = (draft_max_tokens + page_size - 1) // page_size
            self.draft_num_pages = self.draft_max_num_pages_per_request * max_batch_size

            self.draft_num_pages_per_request = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
            self.draft_actual_tokens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self.draft_block_table = torch.arange(
                self.draft_num_pages, dtype=torch.int32, device=self.device
            ).reshape(max_batch_size, self.draft_max_num_pages_per_request)
            self.draft_block_table_unified = self.draft_block_table + self.max_num_pages

            self.target_page_offset = 0
            self.draft_page_offset = self.max_num_pages

            max_pages_per_sample = max(self.max_num_pages_per_request, self.draft_max_num_pages_per_request)
            self.unified_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self.unified_paged_kv_indices = torch.empty(max_batch_size * max_pages_per_sample, dtype=torch.int32, device=self.device)
            self.unified_paged_kv_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
            self.unified_paged_kv_last_page_len = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self.mode_flags = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self.max_dec_len = self.dec_len
            self.unified_qo_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=self.device) * self.max_dec_len
            self._sample_idx_buffer = torch.arange(max_batch_size + 1, dtype=torch.int32, device=self.device)

            self._unified_block_table = torch.zeros(max_batch_size, max_pages_per_sample, dtype=torch.int32, device=self.device)
            self._unified_num_pages = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self._target_page_indices = torch.empty(max_batch_size * self.max_num_pages_per_request, dtype=torch.int32, device=self.device)
            self._target_page_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
            self._target_page_lastlen = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
            self._draft_page_indices = torch.empty(max_batch_size * self.draft_max_num_pages_per_request, dtype=torch.int32, device=self.device)
            self._draft_page_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
            self._draft_page_lastlen = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self.unified_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.unified_buffer, "NHD", use_cuda_graph=True,
                qo_indptr_buf=self.unified_qo_indptr,
                paged_kv_indptr_buf=self.unified_paged_kv_indptr,
                paged_kv_indices_buf=self.unified_paged_kv_indices,
                paged_kv_last_page_len_buf=self.unified_paged_kv_last_page_len,
                backend=self._jit_backend
            )

            torch.library.define(
                "mylib::unified_decode",
                "(Tensor q, Tensor kv_cache, Tensor(a!) out) -> ()",
            )
            @torch.library.impl("mylib::unified_decode", "cuda")
            def unified_decode(q, kv_cache, out):
                self.unified_wrapper.run(q, kv_cache, out=out)

            @torch.library.register_fake("mylib::unified_decode")
            def unified_decode_abstract(q, kv_cache, out):
                return None

    def pre_encode(self, dec_len):
        """Plan the prefill wrapper for the next chunk of dec_len tokens."""
        new_cachelens = self.cachelens + dec_len
        # In place: the replan CUDA graph reads this buffer by address.
        self.num_pages_per_request.copy_(
            self._get_num_pages(new_cachelens).to(torch.int32)
        )

        qo_indptr = self.qo_indptr * dec_len
        new_indices = torch.cat([
            torch.arange(i * self.max_num_pages_per_request,
                        i * self.max_num_pages_per_request + self.num_pages_per_request[i],
                        dtype=torch.int32, device=self.device)
            for i in range(self.batch_size)
        ])
        self.paged_kv_indices[:len(new_indices)] = new_indices
        self.paged_kv_indptr[1:] = torch.cumsum(self.num_pages_per_request, dim=0, dtype=torch.int32)
        self.paged_kv_last_page_len.copy_(self._get_last_page_len(new_cachelens))

        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
            causal=True
        )

    def compile(self):
        """Compile unified_forward in addition to the parent paths."""
        super().compile()
        if self.is_spec:
            self.unified_forward = torch.compile(self.unified_forward, mode="max-autotune", fullgraph=True)

    @torch.inference_mode()
    def unified_inference(self, input_ids: torch.LongTensor, mode_flags: torch.Tensor,
                          verify_dec_len: int, benchmark=False):
        """
        Unified inference for mixed draft/verify batches.

        All tokens are written to BOTH caches. For draft samples, the target cache write
        is temporary - it will be overwritten on next verify since cachelens isn't persisted.

        Args:
            input_ids: (batch_size, seq_len) tokens - draft samples have seq_len=1, verify have seq_len=verify_dec_len
            mode_flags: (batch_size,) DRAFT_MODE=0 or VERIFY_MODE=1 per sample
            verify_dec_len: Decode length for verify samples (gamma+1)
            benchmark: If True, rollback cache state after forward
        """
        bsz = mode_flags.shape[0]

        # Identify draft and verify samples
        draft_mask = (mode_flags == DRAFT_MODE)
        verify_mask = ~draft_mask
        draft_samples = draft_mask.nonzero(as_tuple=True)[0]
        verify_samples = verify_mask.nonzero(as_tuple=True)[0]
        n_draft = draft_samples.shape[0]
        n_verify = verify_samples.shape[0]

        # Update cache lengths
        self.cachelens[draft_samples] += 1  # Temporary
        self.cachelens[verify_samples] += verify_dec_len
        self.draft_cachelens += 1
        self.draft_actual_tokens += 1

        # Update page tables
        self._update_target_page_table(self.cachelens)
        self._update_draft_page_table(self.draft_actual_tokens)

        # Build unified attention block table
        if n_draft > 0:
            self._unified_block_table[:n_draft] = self.draft_block_table_unified[draft_samples]
            self._unified_num_pages[:n_draft] = self.draft_num_pages_per_request[draft_samples]
        if n_verify > 0:
            self._unified_block_table[n_draft:bsz] = self.target_block_table[verify_samples]
            self._unified_num_pages[n_draft:bsz] = self.num_pages_per_request[verify_samples]

        build_kv_indices_from_block_table(
            self._unified_block_table[:bsz],
            self._unified_num_pages[:bsz],
            self.unified_paged_kv_indices,
            self.unified_paged_kv_indptr,
        )

        if n_draft > 0:
            self.unified_paged_kv_last_page_len[:n_draft] = self.draft_paged_kv_last_page_len[draft_samples]
        if n_verify > 0:
            self.unified_paged_kv_last_page_len[n_draft:bsz] = self.paged_kv_last_page_len[verify_samples]

        # Build qo_indptr
        self.unified_qo_indptr[0] = 0
        self.unified_qo_indptr[1:n_draft+1] = self._sample_idx_buffer[1:n_draft+1]
        self.unified_qo_indptr[n_draft+1:bsz+1] = n_draft + self._sample_idx_buffer[1:n_verify+1] * verify_dec_len

        # Plan attention
        self.unified_wrapper.plan(
            qo_indptr=self.unified_qo_indptr,
            paged_kv_indptr=self.unified_paged_kv_indptr,
            paged_kv_indices=self.unified_paged_kv_indices,
            paged_kv_last_page_len=self.unified_paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
            causal=True,
        )

        # Build input_pos and rearrange tokens
        input_pos = torch.empty(bsz, dtype=torch.int32, device=self.device)
        if n_draft > 0:
            input_pos[:n_draft] = self.draft_cachelens[draft_samples]
        if n_verify > 0:
            input_pos[n_draft:] = self.cachelens[verify_samples] - verify_dec_len

        if n_draft > 0 and n_verify > 0:
            rearranged_input = torch.cat([input_ids[draft_samples, :1], input_ids[verify_samples]], dim=0).view(-1)
        elif n_draft > 0:
            rearranged_input = input_ids[draft_samples, :1].view(-1)
        else:
            rearranged_input = input_ids[verify_samples].view(-1)

        # Build KV append page tables
        # Target cache
        if n_draft > 0:
            self._unified_block_table[:n_draft] = self.target_block_table[draft_samples]
            self._unified_num_pages[:n_draft] = self.num_pages_per_request[draft_samples]
        if n_verify > 0:
            self._unified_block_table[n_draft:bsz] = self.target_block_table[verify_samples]
            self._unified_num_pages[n_draft:bsz] = self.num_pages_per_request[verify_samples]

        build_kv_indices_from_block_table(
            self._unified_block_table[:bsz, :self.max_num_pages_per_request],
            self._unified_num_pages[:bsz],
            self._target_page_indices,
            self._target_page_indptr,
        )

        if n_draft > 0:
            self._target_page_lastlen[:n_draft] = self.paged_kv_last_page_len[draft_samples]
        if n_verify > 0:
            self._target_page_lastlen[n_draft:bsz] = self.paged_kv_last_page_len[verify_samples]

        # Draft cache
        if n_draft > 0:
            self._unified_block_table[:n_draft] = self.draft_block_table[draft_samples]
            self._unified_num_pages[:n_draft] = self.draft_num_pages_per_request[draft_samples]
        if n_verify > 0:
            self._unified_block_table[n_draft:bsz] = self.draft_block_table[verify_samples]
            self._unified_num_pages[n_draft:bsz] = self.draft_num_pages_per_request[verify_samples]

        build_kv_indices_from_block_table(
            self._unified_block_table[:bsz, :self.draft_max_num_pages_per_request],
            self._unified_num_pages[:bsz],
            self._draft_page_indices,
            self._draft_page_indptr,
        )

        if n_draft > 0:
            self._draft_page_lastlen[:n_draft] = self.draft_paged_kv_last_page_len[draft_samples]
        if n_verify > 0:
            self._draft_page_lastlen[n_draft:bsz] = self.draft_paged_kv_last_page_len[verify_samples]

        # Forward pass
        logits = self.unified_forward(
            model=self.model,
            x=rearranged_input,
            input_pos=input_pos,
            kv_append_indptr=self.unified_qo_indptr,
            target_page_indices=self._target_page_indices,
            target_page_indptr=self._target_page_indptr,
            target_page_lastlen=self._target_page_lastlen,
            draft_page_indices=self._draft_page_indices,
            draft_page_indptr=self._draft_page_indptr,
            draft_page_lastlen=self._draft_page_lastlen,
        )

        # Revert draft samples' target cache length
        self.cachelens[draft_samples] -= 1

        # Rearrange output
        output = torch.zeros(bsz, verify_dec_len, dtype=logits.dtype, device=self.device)
        if n_draft > 0:
            output[draft_samples, 0] = logits[:n_draft]
        if n_verify > 0:
            output[verify_samples] = logits[n_draft:].view(n_verify, verify_dec_len)

        if benchmark:
            self.cachelens[verify_samples] -= verify_dec_len
            self.draft_cachelens -= 1
            self.draft_actual_tokens -= 1

        return output
