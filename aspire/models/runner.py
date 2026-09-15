import torch
from aspire.models.transformer import Transformer
from aspire.attention.flashinfer import resolve_flashinfer_attention_backend
import flashinfer
import triton
import triton.language as tl


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

    # Build indptr from page counts (cumsum)
    paged_kv_indptr[0] = 0
    paged_kv_indptr[1:batch_size+1] = torch.cumsum(num_pages_per_request, dim=0)

    # Build indices via Triton kernel (one program per request)
    if batch_size > 0:
        _build_kv_indices_kernel[(batch_size,)](
            paged_kv_indices,
            block_table,
            block_table.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=128,
        )

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_len: int = 1, draft_dec_len: int = None) -> None:
        self.dtype = dtype
        self.device = device
        self.dec_len = dec_len
        self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        self.prefill = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last=None, draft_paged_kv_indptr=None, draft_paged_kv_indices=None, draft_paged_kv_last_page_len=None: model.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        self.cachelens = None
        self.is_spec = False
        if draft_dec_len != None:
            self.is_spec = True
            self.draft_cachelens = None
            self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen: model.verify(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen)
            self.draft_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model.draft_forward(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)

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

        # Update tracking tensor
        self.num_pages_per_request = num_pages.to(torch.int32)

        # Build paged_kv_indices from block_table via Triton kernel
        build_kv_indices_from_block_table(
            self.target_block_table,
            self.num_pages_per_request,
            self.paged_kv_indices,
            self.paged_kv_indptr,
        )

        # Update last page lengths
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

        # Update tracking tensor
        self.draft_num_pages_per_request = num_pages.to(torch.int32)

        # Build paged_kv_indices from block_table via Triton kernel
        # Uses draft_block_table (without offset) for draft_wrapper
        build_kv_indices_from_block_table(
            self.draft_block_table,
            self.draft_num_pages_per_request,
            self.draft_paged_kv_indices,
            self.draft_paged_kv_indptr,
        )

        # Update last page lengths
        self.draft_paged_kv_last_page_len.copy_(last_page_len)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, draft_budget = 0, window_size = 32, page_size: int = 128):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.page_size = page_size
        max_num_pages = max_batch_size * (max_seq_length // page_size)
        if max_num_pages*page_size < max_batch_size*max_seq_length:
            max_num_pages += max_batch_size
        self.max_num_pages_per_request = max_num_pages // max_batch_size
        self.num_pages_per_request = torch.zeros(max_batch_size, device=self.device, dtype=torch.int32)
        self.max_num_pages = max_num_pages

        # Static block table: physical pages of request i are
        # [i*max_pages_per_req, (i+1)*max_pages_per_req).
        self.target_block_table = torch.arange(
            max_num_pages, dtype=torch.int32, device=self.device
        ).reshape(max_batch_size, self.max_num_pages_per_request)

        self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)

        self._jit_backend = resolve_flashinfer_attention_backend(self.dtype)
        print(f"[aspire] FlashInfer attention backend: {self._jit_backend}")

        self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode_buffer, "NHD", use_cuda_graph=True,
                                                                              qo_indptr_buf=self.qo_indptr,
                                                                              paged_kv_indptr_buf=self.paged_kv_indptr,
                                                                              paged_kv_indices_buf=self.paged_kv_indices,
                                                                              paged_kv_last_page_len_buf=self.paged_kv_last_page_len,
                                                                              backend=self._jit_backend)

        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_buffer, "NHD", backend=self._jit_backend)
        torch.library.define(
            "mylib::target_decode",
            "(Tensor q, Tensor kv_cache, Tensor(a!) out) -> ()",
        )
        @torch.library.impl("mylib::target_decode", "cuda")
        def target_decode(q, kv_cache, out):
            self.decode_wrapper.run(q, kv_cache, out=out)
        @torch.library.register_fake("mylib::target_decode")
        def target_decode_abstract(q, kv_cache, out):
            return None

        torch.library.define(
            "mylib::target_prefill",
            "(Tensor q, Tensor kv_cache, Tensor(a!) out) -> ()",
        )
        @torch.library.impl("mylib::target_prefill", "cuda")
        def target_prefill(q, kv_cache, out):
            self.prefill_wrapper.run(q, kv_cache, out=out)
        @torch.library.register_fake("mylib::target_prefill")
        def target_prefill_abstract(q, kv_cache, out):
            return None

        # If using speculative decoding, init draft attention backend
        if self.is_spec:
            self.draft_budget = draft_budget
            self.draft_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
            self.draft_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

            # Draft cache holds draft_budget tokens plus room for min_gen_tokens more.
            min_gen_tokens = 255
            draft_max_tokens = draft_budget + min_gen_tokens
            self.draft_max_num_pages_per_request = (draft_max_tokens + page_size - 1) // page_size
            self.draft_num_pages = self.draft_max_num_pages_per_request * max_batch_size

            self.draft_num_pages_per_request = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
            # Physical token count of the draft cache (draft_cachelens is the logical position).
            self.draft_actual_tokens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self.draft_block_table = torch.arange(
                self.draft_num_pages, dtype=torch.int32, device=self.device
            ).reshape(max_batch_size, self.draft_max_num_pages_per_request)

            self.draft_paged_kv_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
            self.draft_paged_kv_indices = torch.empty(self.draft_num_pages, dtype=torch.int32, device=self.device)
            self.draft_paged_kv_last_page_len = torch.ones((max_batch_size), dtype=torch.int32, device=self.device)
            self.draft_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.draft_buffer, "NHD", use_cuda_graph=True,
                                                                                qo_indptr_buf=self.qo_indptr,
                                                                                paged_kv_indptr_buf=self.draft_paged_kv_indptr,
                                                                                paged_kv_indices_buf=self.draft_paged_kv_indices,
                                                                                paged_kv_last_page_len_buf=self.draft_paged_kv_last_page_len,
                                                                                backend=self._jit_backend)
            torch.library.define(
                "mylib::draft_decode",
                "(Tensor q, Tensor kv_cache, Tensor(a!) out) -> ()",
            )
            @torch.library.impl("mylib::draft_decode", "cuda")
            def draft_decode(q, kv_cache, out):
                self.draft_wrapper.run(q, kv_cache, out=out)
            @torch.library.register_fake("mylib::draft_decode")
            def draft_decode_abstract(q, kv_cache, out):
                return None

        if self.is_spec:
            with torch.device(self.device):
                self.model.setup_caches(num_pages=max_num_pages, page_size=page_size, spec=self.is_spec, draft_num_pages = self.draft_num_pages, draft_budget = draft_budget, window_size = window_size)
        else:
            with torch.device(self.device):
                self.model.setup_caches(num_pages=max_num_pages, page_size=page_size)

    def compile(self):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._functorch.config.enable_autograd_cache = True
        self.model_forward = torch.compile(self.model_forward, mode="max-autotune", fullgraph=True)
        if self.is_spec:
            self.draft_forward = torch.compile(self.draft_forward, mode="max-autotune", fullgraph=True)

    # Only used for baseline inference
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            self.pre_decode(dec_len=dec_len)
            logits = self.model_forward(
                model=self.model,
                x=input_ids,
                input_pos=self.cachelens,
                kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len)

            self.cachelens += dec_len
            if benchmark:
                # Roll back; the page table is recomputed by the next pre_decode.
                self.cachelens -= dec_len
            return logits

    def pre_decode(self, dec_len):
            # Compute page table state for the new sequence length (after adding dec_len tokens)
            new_cachelens = self.cachelens + dec_len
            self._update_target_page_table(new_cachelens)
            self.decode_wrapper.plan(
                qo_indptr=self.qo_indptr*dec_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=self.paged_kv_last_page_len,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=self.page_size,
                q_data_type=self.dtype,
                causal=True,
            )
    
    # Only used for target verification
    @torch.inference_mode()
    def verify(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            self.pre_verify(dec_len=dec_len)
            logits = self.model_forward(
                model=self.model,
                x=input_ids,
                input_pos=self.cachelens,
                kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len,
                draft_kv_page_indices=self.draft_paged_kv_indices, draft_kv_page_indptr=self.draft_paged_kv_indptr, draft_kv_page_lastlen=self.draft_paged_kv_last_page_len)

            self.cachelens += dec_len
            self.draft_cachelens += 1
            self.draft_actual_tokens += 1
            if benchmark:
                # Roll back; the page tables are recomputed by the next pre_verify.
                self.cachelens -= dec_len
                self.draft_cachelens -= 1
                self.draft_actual_tokens -= 1
            return logits

    def pre_verify(self, dec_len):
            # Compute page table state for new sequence lengths
            # Target cache: will have cachelens + dec_len tokens after forward
            new_target_cachelens = self.cachelens + dec_len
            self._update_target_page_table(new_target_cachelens)

            # Draft cache: verify writes all tokens but only increments logical length by 1
            new_draft_actual = self.draft_actual_tokens + 1
            self._update_draft_page_table(new_draft_actual)

            self.decode_wrapper.plan(
                qo_indptr=self.qo_indptr*dec_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=self.paged_kv_last_page_len,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=self.page_size,
                q_data_type=self.dtype,
                causal=True,
            )
    
    @torch.inference_mode()
    def speculate(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            self.pre_spec(dec_len=dec_len)
            logits = self.draft_forward(
                model=self.model,
                x=input_ids,
                input_pos=self.draft_cachelens,
                kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.draft_paged_kv_indices, kv_page_indptr= self.draft_paged_kv_indptr, kv_page_lastlen = self.draft_paged_kv_last_page_len)

            self.draft_cachelens += dec_len
            self.draft_actual_tokens += dec_len
            if benchmark:
                # Roll back; the page table is recomputed by the next pre_spec.
                self.draft_cachelens -= dec_len
                self.draft_actual_tokens -= dec_len
            return logits

    def pre_spec(self, dec_len):
            # Compute page table state for new draft cache length (after adding dec_len tokens)
            new_draft_actual = self.draft_actual_tokens + dec_len
            self._update_draft_page_table(new_draft_actual)
            self.draft_wrapper.plan(
                qo_indptr=self.qo_indptr*dec_len,
                paged_kv_indptr=self.draft_paged_kv_indptr,
                paged_kv_indices=self.draft_paged_kv_indices,
                paged_kv_last_page_len=self.draft_paged_kv_last_page_len,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=self.page_size,
                q_data_type=self.dtype,
                causal=True,
            )
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, benchmark = False):
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        chunk_size = 128
        num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
        is_last = False
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            dec_len = end_idx-start_idx
            if i == num_chunks - 1:
                is_last = True
            self.pre_encode(dec_len=dec_len)
            if self.is_spec and is_last:
                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=self.cachelens,
                    kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len, 
                    is_last=is_last, draft_paged_kv_indptr=self.draft_paged_kv_indptr, draft_paged_kv_indices=self.draft_paged_kv_indices, draft_paged_kv_last_page_len=self.draft_paged_kv_last_page_len
                )
            else:
                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=self.cachelens,
                    kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len, 
                )
            self.cachelens += dec_len

        if self.is_spec:
            self.draft_cachelens.copy_(self.cachelens)
            # After prefill, draft cache contains exactly draft_budget tokens (selected by gen_draft_kv)
            # Page table was already set up in clear_kv() for draft_budget tokens
            self.draft_actual_tokens.fill_(self.draft_budget)

        return logits
    
    def pre_encode(self, dec_len):
        # Compute page table state for the new sequence length (after this prefill chunk)
        new_cachelens = self.cachelens + dec_len
        self.num_pages_per_request = self._get_num_pages(new_cachelens).to(torch.int32)

        qo_indptr = self.qo_indptr * dec_len
        # Build page indices and copy IN-PLACE (wrapper holds reference to this buffer)
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
          
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.kv_cache.zero_()
            if self.is_spec:
                b.attention.kv_cache.draft_cache.zero_()

        # Reset sequence lengths - page table values will be computed on-demand from cachelens
        self.cachelens.zero_()
        self.qo_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices.zero_()  # Will be filled in pre_* methods
        self.paged_kv_last_page_len.zero_()  # Will be computed in pre_* methods
        self.num_pages_per_request = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)

        if self.is_spec:
            self.draft_cachelens.zero_()
            self.draft_actual_tokens.zero_()
            self.draft_num_pages_per_request.zero_()
            self.draft_paged_kv_indptr.zero_()
            self.draft_paged_kv_indices.zero_()  # Don't reassign - wrapper holds reference
            self.draft_paged_kv_last_page_len.zero_()

            # Pre-set draft page table for gen_draft_kv during prefill
            # gen_draft_kv writes draft_budget tokens, so we need page table ready for that
            draft_budget_tensor = torch.full((self.batch_size,), self.draft_budget, dtype=torch.int32, device=self.device)
            self._update_draft_page_table(draft_budget_tensor)
