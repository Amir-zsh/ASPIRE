"""Sparse-draft runner: the draft cache holds only the top-k compressed
tokens selected at prefill, and draft attention reads them together with
the target pages from the recent window onward. All decode KV writes go to
the target cache.

Constraints (asserted): topk % page_size == 0 and
(prefix_len - window_size) % page_size == 0, so both the draft cache and
the window start at page boundaries.
"""
import torch
from aspire.engine.buffers import StaticBufferRegistry
from aspire.models.mixed_runner import LMBackend, DRAFT_MODE, VERIFY_MODE
from aspire.models.sparse_draft_transformer import TransformerSparseDraft
from aspire.attention.flashinfer import resolve_flashinfer_attention_backend
import flashinfer
import os
from contextlib import nullcontext
try:
    from torch.profiler import record_function
except Exception:  # pragma: no cover - profiler may be unavailable in minimal builds
    record_function = None


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


import time as _time
_timer_accum: dict[str, float] = {}
_timer_counts: dict[str, int] = {}


class _CudaTimerScope:
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = _time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize()
        dt = _time.perf_counter() - self.t0
        _timer_accum[self.name] = _timer_accum.get(self.name, 0.0) + dt
        _timer_counts[self.name] = _timer_counts.get(self.name, 0) + 1
        return False


def reset_timer_stats():
    _timer_accum.clear()
    _timer_counts.clear()


def get_timer_stats() -> dict[str, tuple[float, int]]:
    """Returns {name: (total_seconds, call_count)}."""
    return {k: (_timer_accum[k], _timer_counts[k]) for k in sorted(_timer_accum)}


def _profile_scope(name: str):
    if os.getenv("ASPIRE_TIMER") == "1":
        return _CudaTimerScope(name)
    if os.getenv("ASPIRE_NVTX") == "1":
        return _NvtxRange(name)
    if os.getenv("ASPIRE_PROFILE") == "1" and record_function is not None:
        return record_function(name)
    return nullcontext()
import triton
import triton.language as tl

from aspire.engine.buffers import graph_buckets


def compute_bucket_sizes(max_len: int) -> tuple:
    """Return graph buckets that do not exceed their backing allocation."""
    return graph_buckets(max_len)


@triton.jit
def _copy_indices_kernel(
    paged_kv_indices,       # Output: [total_pages] flattened page indices
    paged_kv_indptr,        # Input: [batch_size + 1] cumulative page counts
    mode_flags,             # Input: [batch_size] DRAFT_MODE=0 or VERIFY_MODE=1
    draft_indices,          # Input: [batch_size, max_draft_pages] pre-computed
    verify_indices,         # Input: [batch_size, max_verify_pages] pre-computed
    attn_num_pages,         # Input: [batch_size] pages to copy per sample
    max_draft_pages: tl.constexpr,
    max_verify_pages: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy each row's page indices from draft_indices or verify_indices by mode."""
    sample_idx = tl.program_id(0)
    is_draft = tl.load(mode_flags + sample_idx) == 0  # DRAFT_MODE = 0
    start_idx = tl.load(paged_kv_indptr + sample_idx)
    n_pages = tl.load(attn_num_pages + sample_idx)

    offset = tl.arange(0, BLOCK_SIZE)

    if is_draft:
        src_base = sample_idx * max_draft_pages
        for i in tl.range(0, n_pages, BLOCK_SIZE):
            mask = (i + offset) < n_pages
            page_ids = tl.load(draft_indices + src_base + i + offset, mask=mask)
            tl.store(paged_kv_indices + start_idx + i + offset, page_ids, mask=mask)
    else:
        src_base = sample_idx * max_verify_pages
        for i in tl.range(0, n_pages, BLOCK_SIZE):
            mask = (i + offset) < n_pages
            page_ids = tl.load(verify_indices + src_base + i + offset, mask=mask)
            tl.store(paged_kv_indices + start_idx + i + offset, page_ids, mask=mask)


def build_unified_indices(
    mode_flags: torch.Tensor,
    draft_indices: torch.Tensor,
    verify_indices: torch.Tensor,
    draft_num_pages: torch.Tensor,
    verify_num_pages: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
) -> None:
    """Fill paged_kv_indptr/paged_kv_indices per row from the precomputed
    draft or verify index arrays, selected by mode_flags."""
    batch_size = mode_flags.shape[0]
    is_draft = (mode_flags == DRAFT_MODE)
    attn_num_pages = torch.where(is_draft, draft_num_pages, verify_num_pages)
    paged_kv_indptr[0] = 0
    paged_kv_indptr[1:batch_size + 1] = torch.cumsum(attn_num_pages, dim=0)
    if batch_size > 0:
        _copy_indices_kernel[(batch_size,)](
            paged_kv_indices,
            paged_kv_indptr,
            mode_flags,
            draft_indices,
            verify_indices,
            attn_num_pages,
            draft_indices.shape[1],
            verify_indices.shape[1],
            BLOCK_SIZE=128,
        )


def load_sparse_draft_model(checkpoint_path, model_name, device, precision, use_tp, rank_group=None, group=None):
    """Load a TransformerSparseDraft from a converted model.pth."""
    with torch.device('meta'):
        model = TransformerSparseDraft.from_name(model_name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from aspire.tp import apply_tp
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()


class LMBackendSparseDraft(LMBackend):
    """Runner with a static top-k draft cache; see the module docstring."""

    def _use_holistic_attention(self) -> bool:
        """Whether to use HolisticAttentionWrapper instead of FA2 prefill wrapper."""
        return getattr(self, '_holistic_attention', False)

    def _create_attention_wrapper(self, workspace_buffer, *, use_cuda_graph,
                                   qo_indptr_buf=None, kv_indptr_buf=None,
                                   kv_indices_buf=None, kv_last_page_len_buf=None,
                                   jit_args=None):
        """Create the FlashInfer attention wrapper (holistic or standard paged prefill)."""
        jit_tag = " (JIT)" if jit_args else ""
        if self._use_holistic_attention():
            from aspire.attention.holistic_wrapper import HolisticAttentionWrapper
            print(f"[AttentionWrapper] Using HolisticAttentionWrapper{jit_tag}")
            return HolisticAttentionWrapper(
                workspace_buffer, "NHD",
                use_cuda_graph=use_cuda_graph,
                qo_indptr_buf=qo_indptr_buf,
                paged_kv_indptr_buf=kv_indptr_buf,
                paged_kv_indices_buf=kv_indices_buf,
                paged_kv_last_page_len_buf=kv_last_page_len_buf,
                jit_args=jit_args,
                jit_kwargs={} if jit_args else None,
            )
        print(f"[AttentionWrapper] Using BatchPrefillWithPagedKVCacheWrapper (backend={self._jit_backend}){jit_tag}")
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD",
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=kv_indptr_buf,
            paged_kv_indices_buf=kv_indices_buf,
            paged_kv_last_page_len_buf=kv_last_page_len_buf,
            backend=self._jit_backend,
            jit_args=jit_args,
            jit_kwargs={} if jit_args else None,
        )

    def load_model(self, checkpoints, model_name: str, use_tp: bool, rank_group=None, group=None):
        """Load a TransformerSparseDraft from a converted model.pth."""
        self.model: TransformerSparseDraft = load_sparse_draft_model(
            checkpoint_path=checkpoints, model_name=model_name, device=self.device, precision=self.dtype,
            use_tp=use_tp, rank_group=rank_group, group=group
        )

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048,
                     draft_budget: int = 0, window_size: int = 32, page_size: int = 128,
                     use_cuda_graph: bool = True,
                     deterministic_attention: bool = False,
                     top_k: int = 50,
                     temperature: float = 0, top_p: float = 0):
        """Allocate KV caches, page tables, attention wrappers, and static buffers."""
        self.static_buffers = StaticBufferRegistry()
        self.topk = draft_budget - window_size
        assert self.topk % page_size == 0, (
            f"topk = draft_budget - window_size = {self.topk} must be divisible by page_size ({page_size})"
        )
        self.window_size = window_size
        # Deterministic FlashInfer planning/execution (disables split-KV).
        self.deterministic_attention = deterministic_attention
        self.topk_pages_per_sample = self.topk // page_size

        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.page_size = page_size

        max_num_pages = max_batch_size * (max_seq_length // page_size)
        if max_num_pages * page_size < max_batch_size * max_seq_length:
            max_num_pages += max_batch_size
        self.max_num_pages_per_request = max_num_pages // max_batch_size
        self.num_pages_per_request = torch.zeros(max_batch_size, device=self.device, dtype=torch.int32)
        self.max_num_pages = max_num_pages

        self.target_block_table = torch.arange(
            max_num_pages, dtype=torch.int32, device=self.device
        ).reshape(max_batch_size, self.max_num_pages_per_request)

        self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.qo_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)
        self._jit_backend = resolve_flashinfer_attention_backend(self.dtype)
        print(f"[sparse-draft runner] FlashInfer attention backend: {self._jit_backend}")
        self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.decode_buffer, "NHD", use_cuda_graph=True,
            qo_indptr_buf=self.qo_indptr,
            paged_kv_indptr_buf=self.paged_kv_indptr,
            paged_kv_indices_buf=self.paged_kv_indices,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len,
            backend=self._jit_backend
        )

        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.prefill_buffer, "NHD", backend=self._jit_backend)
        self._define_target_attention_ops()

        self._top_k = top_k
        self._greedy_mode = (top_k == 0)

        if self.is_spec:
            self.draft_budget = draft_budget
            self.draft_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
            self.draft_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

            # The draft cache is static after prefill: exactly topk tokens per request.
            self.draft_max_num_pages_per_request = self.topk_pages_per_sample
            self.draft_num_pages = self.draft_max_num_pages_per_request * max_batch_size
            self.draft_num_pages_per_request = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
            self.draft_actual_tokens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            self.draft_block_table = torch.arange(
                self.draft_num_pages, dtype=torch.int32, device=self.device
            ).reshape(max_batch_size, self.draft_max_num_pages_per_request)
            # Same table offset into the unified (target + draft) cache.
            self.draft_block_table_unified = self.draft_block_table + max_num_pages

            self.draft_paged_kv_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
            self.draft_paged_kv_indices = torch.empty(self.draft_num_pages, dtype=torch.int32, device=self.device)
            self.draft_paged_kv_last_page_len = torch.ones((max_batch_size), dtype=torch.int32, device=self.device)
            self.draft_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.draft_buffer, "NHD", use_cuda_graph=True,
                qo_indptr_buf=self.qo_indptr,
                paged_kv_indptr_buf=self.draft_paged_kv_indptr,
                paged_kv_indices_buf=self.draft_paged_kv_indices,
                paged_kv_last_page_len_buf=self.draft_paged_kv_last_page_len,
                backend=self._jit_backend
            )
            self._define_draft_attention_ops()

            # Unified attention buffers, sized for the worst case per row.
            max_pages_per_sample = max(self.max_num_pages_per_request,
                                       self.draft_max_num_pages_per_request + self.max_num_pages_per_request)
            self.unified_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            # One extra dummy row (0 pages) absorbs the padding tokens of a bucket.
            self.unified_paged_kv_indices = torch.empty(max_batch_size * max_pages_per_sample, dtype=torch.int32, device=self.device)
            self.unified_paged_kv_indptr = torch.zeros(max_batch_size + 2, dtype=torch.int32, device=self.device)
            self.unified_paged_kv_last_page_len = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)

            self.mode_flags = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

            # Draft attention indices: [draft topk pages, target pages from the window start]; filled in encode().
            # Verify rows use target_block_table directly.
            max_draft_attn_pages = self.topk_pages_per_sample + self.max_num_pages_per_request
            self._draft_indices = torch.zeros(
                max_batch_size, max_draft_attn_pages,
                dtype=torch.int32, device=self.device
            )

            self.max_dec_len = self.dec_len
            self.max_unified_tokens = max_batch_size * self.max_dec_len

            self.bucket_sizes = compute_bucket_sizes(self.max_unified_tokens)
            self.max_bucket_size = max(self.bucket_sizes)
            print(f"Bucket sizes: {len(self.bucket_sizes)} buckets from 1 to {self.max_bucket_size}")

            self.unified_qo_indptr = torch.zeros(max_batch_size + 2, dtype=torch.int32, device=self.device)
            self._padded_tokens = torch.zeros(self.max_unified_tokens, dtype=torch.long, device=self.device)
            # Fixed-size views per bucket keep tensor identity stable for torch.compile.
            self._bucket_views = {}
            for bucket in self.bucket_sizes:
                self._bucket_views[bucket] = self._padded_tokens[:bucket]
            self._input_pos = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device)
            # Mark static input buffers so Inductor cudagraphs don't require pool allocation.
            import torch._dynamo as dynamo
            dynamo.mark_static_address(self._padded_tokens)
            dynamo.mark_static_address(self._input_pos)
            dynamo.mark_static_address(self.unified_qo_indptr)
            dynamo.mark_static_address(self.unified_paged_kv_indices)
            dynamo.mark_static_address(self.unified_paged_kv_indptr)
            dynamo.mark_static_address(self.unified_paged_kv_last_page_len)

            if self._greedy_mode:
                self._output_buffer = torch.zeros(
                    max_batch_size, self.max_dec_len, dtype=torch.int64, device=self.device)
            else:
                self._topk_values_buffer = torch.zeros(
                    max_batch_size, self.max_dec_len, self._top_k,
                    dtype=torch.float32, device=self.device)
                self._topk_indices_buffer = torch.zeros(
                    max_batch_size, self.max_dec_len, self._top_k,
                    dtype=torch.int64, device=self.device)

            # compile() recreates the wrapper with use_cuda_graph=True if needed.
            self._flashinfer_use_cuda_graph = use_cuda_graph
            self.unified_wrapper = self._create_attention_wrapper(
                self.unified_buffer,
                use_cuda_graph=use_cuda_graph,
                qo_indptr_buf=self.unified_qo_indptr if use_cuda_graph else None,
                kv_indptr_buf=self.unified_paged_kv_indptr if use_cuda_graph else None,
                kv_indices_buf=self.unified_paged_kv_indices if use_cuda_graph else None,
                kv_last_page_len_buf=self.unified_paged_kv_last_page_len if use_cuda_graph else None,
            )

            # Initial plan() with the dummy padding row.
            attn_bsz = max_batch_size + 1
            self.unified_qo_indptr[:attn_bsz] = torch.arange(attn_bsz, dtype=torch.int32, device=self.device) * self.max_dec_len
            self.unified_qo_indptr[attn_bsz] = self.max_unified_tokens
            self.unified_paged_kv_indptr[:attn_bsz] = torch.arange(attn_bsz, dtype=torch.int32, device=self.device) * max_pages_per_sample
            self.unified_paged_kv_indptr[attn_bsz] = self.unified_paged_kv_indptr[attn_bsz - 1]
            self.unified_paged_kv_indices[:max_batch_size * max_pages_per_sample] = torch.arange(
                max_batch_size * max_pages_per_sample, dtype=torch.int32, device=self.device
            )
            self.unified_paged_kv_last_page_len[:max_batch_size] = page_size
            self.unified_paged_kv_last_page_len[max_batch_size] = 0
            self.unified_wrapper.plan(
                qo_indptr=self.unified_qo_indptr,
                paged_kv_indptr=self.unified_paged_kv_indptr,
                paged_kv_indices=self.unified_paged_kv_indices,
                paged_kv_last_page_len=self.unified_paged_kv_last_page_len,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=page_size,
                q_data_type=self.dtype,
                causal=True,
                disable_split_kv=self.deterministic_attention,
            )

            self._define_unified_attention_ops()

            self._unified_forward_impl = lambda model, x, input_pos, kv_append_indptr, target_page_indices, target_page_indptr, target_page_lastlen: model.unified_forward(
                x, input_pos, kv_append_indptr,
                target_page_indices, target_page_indptr, target_page_lastlen
            )
            self.unified_forward = self._unified_forward_impl

        self.model.top_k = self._top_k

        # TP gather buffers for the local top-k merge.
        if not self._greedy_mode and self.model.process_group is not None:
            ws = self.model.world_size
            max_tokens = max_batch_size * self.max_dec_len
            self.model._gather_vals = [
                torch.zeros(
                    max_tokens, self._top_k, dtype=self.dtype, device=self.device
                )
                for _ in range(ws)
            ]
            self.model._gather_ids = [
                torch.zeros(
                    max_tokens, self._top_k, dtype=torch.int64, device=self.device
                )
                for _ in range(ws)
            ]
            self.model._gather_merged_vals = torch.zeros(
                max_tokens, self._top_k * ws, dtype=self.dtype, device=self.device
            )
            self.model._gather_merged_ids = torch.zeros(
                max_tokens, self._top_k * ws, dtype=torch.int64, device=self.device
            )

        if not self._greedy_mode:
            self.set_logit_processing(temperature, top_p)

        if self.is_spec:
            with torch.device(self.device):
                self.model.setup_caches(
                    num_pages=max_num_pages, page_size=page_size, spec=self.is_spec,
                    draft_num_pages=self.draft_num_pages, draft_budget=draft_budget, window_size=window_size,
                    max_unified_tokens=self.max_unified_tokens,
                )
        else:
            with torch.device(self.device):
                self.model.setup_caches(num_pages=max_num_pages, page_size=page_size)

        self._register_static_buffers()

    def _register_static(self, name: str, tensor):
        """Register a persistent tensor that participates in graph preparation or replay."""
        return self.static_buffers.register(name, tensor)

    def _register_static_buffers(self):
        base = {
            "state.cachelens": self.cachelens,
            "state.num_pages": self.num_pages_per_request,
            "target.block_table": self.target_block_table,
            "target.workspace.decode": self.decode_buffer,
            "target.metadata.qo_indptr": self.qo_indptr,
            "target.metadata.kv_indptr": self.paged_kv_indptr,
            "target.metadata.kv_indices": self.paged_kv_indices,
            "target.metadata.kv_last_page_len": self.paged_kv_last_page_len,
        }
        gather_vals = getattr(self.model, "_gather_vals", None)
        gather_ids = getattr(self.model, "_gather_ids", None)
        if isinstance(gather_vals, list):
            for rank, (values, indices) in enumerate(zip(gather_vals, gather_ids)):
                base[f"tp.gather_values.rank{rank}"] = values
                base[f"tp.gather_indices.rank{rank}"] = indices
            base["tp.gather_values.merged"] = self.model._gather_merged_vals
            base["tp.gather_indices.merged"] = self.model._gather_merged_ids
        for name, tensor in base.items():
            self._register_static(name, tensor)
        if not self.is_spec:
            return

        speculative = {
            "state.draft_cachelens": self.draft_cachelens,
            "state.draft_num_pages": self.draft_num_pages_per_request,
            "state.draft_actual_tokens": self.draft_actual_tokens,
            "draft.block_table": self.draft_block_table,
            "draft.block_table_unified": self.draft_block_table_unified,
            "draft.workspace": self.draft_buffer,
            "draft.metadata.kv_indptr": self.draft_paged_kv_indptr,
            "draft.metadata.kv_indices": self.draft_paged_kv_indices,
            "draft.metadata.kv_last_page_len": self.draft_paged_kv_last_page_len,
            "unified.workspace": self.unified_buffer,
            "unified.tokens": self._padded_tokens,
            "unified.input_pos": self._input_pos,
            "unified.metadata.qo_indptr": self.unified_qo_indptr,
            "unified.metadata.kv_indptr": self.unified_paged_kv_indptr,
            "unified.metadata.kv_indices": self.unified_paged_kv_indices,
            "unified.metadata.kv_last_page_len": self.unified_paged_kv_last_page_len,
            "unified.mode_flags": self.mode_flags,
            "unified.draft_indices": self._draft_indices,
        }
        if self._greedy_mode:
            speculative["unified.output"] = self._output_buffer
        else:
            speculative["unified.output.topk_values"] = self._topk_values_buffer
            speculative["unified.output.topk_indices"] = self._topk_indices_buffer
        for name, tensor in speculative.items():
            self._register_static(name, tensor)

        for layer_index, layer in enumerate(self.model.layers):
            attention = layer.attention
            self._register_static(
                f"model.layer.{layer_index}.kv_cache",
                attention.kv_cache.unified_cache,
            )
            self._register_static(
                f"model.layer.{layer_index}.attention_output",
                attention.unified_out,
            )

    def _register_graph_output(self, phase: str, bucket: int, kind: str, tensor):
        self._register_static(f"graph.{phase}.{bucket}.{kind}", tensor)

    def seal_graph_buffers(self):
        self.static_buffers.seal()

    def assert_graph_buffers_stable(self):
        self.static_buffers.assert_stable()

    def graph_buffer_pointers(self):
        return self.static_buffers.pointers()

    def graph_buffer_schema(self):
        return self.static_buffers.schema()

    def _define_logit_op(self, pipe, kwargs):
        """Define mylib::logit_process custom op wrapping LogitsPipe."""
        torch.library.define("mylib::logit_process", "(Tensor vals) -> Tensor")

        @torch.library.impl("mylib::logit_process", "cuda")
        def logit_process(vals):
            return pipe(vals, **kwargs)

        @torch.library.register_fake("mylib::logit_process")
        def logit_process_abstract(vals):
            return torch.empty_like(vals, dtype=torch.float32)

    def set_logit_processing(self, temperature: float = 0, top_p: float = 0):
        """Register the logits pipeline (temperature, softmax, top-p; 0 disables an op).
        Called from setup_caches(), before compile()."""
        from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP
        ops = []
        kwargs = {}
        if temperature > 0:
            ops.append(Temperature())
            kwargs['temperature'] = temperature
        ops.append(Softmax())
        if top_p > 0:
            ops.append(TopP())
            kwargs['top_p'] = top_p
        pipe = LogitsPipe(ops)
        self._define_logit_op(pipe, kwargs)
        self.model.logit_process = torch.ops.mylib.logit_process

    def _define_target_attention_ops(self):
        """Define target attention torch.library ops (in-place output)."""
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

    def _define_draft_attention_ops(self):
        """Define draft attention torch.library ops (in-place output)."""
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

    def _define_unified_attention_ops(self):
        """Define unified attention torch.library ops."""
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

    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, benchmark=False):
        """Chunked prefill; on the last chunk the draft cache is built and
        the draft attention indices are precomputed."""
        seq_len = input_ids.shape[1]
        assert (seq_len - self.window_size) % self.page_size == 0, (
            f"(prefix_len - window_size) = {seq_len - self.window_size} must be divisible by "
            f"page_size ({self.page_size}); pad the prompt"
        )
        self.window_start_page = (seq_len - self.window_size) // self.page_size

        self.clear_kv()
        logits = None
        chunk_size = 128
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        is_last = False

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            dec_len = end_idx - start_idx
            if i == num_chunks - 1:
                is_last = True
            self.pre_encode(dec_len=dec_len)

            if self.is_spec and is_last:
                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=self.cachelens,
                    kv_append_indptr=self.qo_indptr * dec_len,
                    kv_page_indices=self.paged_kv_indices,
                    kv_page_indptr=self.paged_kv_indptr,
                    kv_page_lastlen=self.paged_kv_last_page_len,
                    is_last=is_last,
                    draft_paged_kv_indptr=self.draft_paged_kv_indptr,
                    draft_paged_kv_indices=self.draft_paged_kv_indices,
                    draft_paged_kv_last_page_len=self.draft_paged_kv_last_page_len
                )
            else:
                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=self.cachelens,
                    kv_append_indptr=self.qo_indptr * dec_len,
                    kv_page_indices=self.paged_kv_indices,
                    kv_page_indptr=self.paged_kv_indptr,
                    kv_page_lastlen=self.paged_kv_last_page_len,
                )
            self.cachelens += dec_len

        if self.is_spec:
            self.draft_cachelens.copy_(self.cachelens)
            self.draft_actual_tokens.fill_(self.topk)

            # Draft attention indices: [draft topk pages, target pages from window start].
            max_window_pages = self.max_num_pages_per_request - self.window_start_page
            self._draft_indices[:, :self.topk_pages_per_sample] = \
                self.draft_block_table_unified[:, :self.topk_pages_per_sample]
            self._draft_indices[:, self.topk_pages_per_sample:self.topk_pages_per_sample + max_window_pages] = \
                self.target_block_table[:, self.window_start_page:self.window_start_page + max_window_pages]

        return logits

    @torch.inference_mode()
    def clear_kv(self):
        """Zero the KV caches and reset all page-table state."""
        for b in self.model.layers:
            b.attention.kv_cache.kv_cache.zero_()
            if self.is_spec:
                b.attention.kv_cache.draft_cache.zero_()

        self.cachelens.zero_()
        self.qo_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices.zero_()
        self.paged_kv_last_page_len.zero_()
        # In place: the replan CUDA graph reads this buffer by address.
        self.num_pages_per_request.zero_()

        if self.is_spec:
            self.draft_cachelens.zero_()
            self.draft_actual_tokens.zero_()
            self.draft_num_pages_per_request.zero_()
            self.draft_paged_kv_indptr.zero_()
            self.draft_paged_kv_indices.zero_()
            self.draft_paged_kv_last_page_len.zero_()

            # Draft page table for gen_draft_kv: topk tokens per request.
            topk_tensor = torch.full((self.batch_size,), self.topk, dtype=torch.int32, device=self.device)
            self._update_draft_page_table(topk_tensor)

            self.unified_paged_kv_indices.zero_()
            self.unified_paged_kv_indptr.zero_()
            self.unified_paged_kv_last_page_len.zero_()
            self.mode_flags.zero_()

    def _get_bucket_size(self, total_tokens: int) -> int:
        """Return smallest bucket size >= total_tokens."""
        for bucket in self.bucket_sizes:
            if total_tokens <= bucket:
                return bucket
        return self.bucket_sizes[-1]

    @torch.inference_mode()
    def prepare_forward_inputs(self, input_ids: torch.LongTensor, mode_flags: torch.Tensor,
                               verify_dec_lens: torch.Tensor):
        """Advance cache lengths, build the attention plan, and flatten/pad the
        input tokens for one mixed draft/verify step."""
        bsz = mode_flags.shape[0]
        is_draft = (mode_flags == DRAFT_MODE)

        with _profile_scope("PREP_CACHELEN"):
            self.cachelens += verify_dec_lens
            # Verify rows manage draft_cachelens externally (accept/rollback).
            self.draft_cachelens += is_draft.to(self.draft_cachelens.dtype)

        with _profile_scope("PREP_PAGEMETA"):
            verify_num_pages = self._get_num_pages(self.cachelens).to(torch.int32)
            last_page_len = self._get_last_page_len(self.cachelens)
            draft_num_pages = self.topk_pages_per_sample + (verify_num_pages - self.window_start_page)

        with _profile_scope("PREP_ATTN_INDICES"):
            self.unified_paged_kv_last_page_len[:bsz] = last_page_len
            build_unified_indices(
                mode_flags=mode_flags,
                draft_indices=self._draft_indices,
                verify_indices=self.target_block_table,
                draft_num_pages=draft_num_pages,
                verify_num_pages=verify_num_pages,
                paged_kv_indices=self.unified_paged_kv_indices,
                paged_kv_indptr=self.unified_paged_kv_indptr,
            )
            # Dummy padding row: 0 pages.
            self.unified_paged_kv_indptr[bsz + 1] = self.unified_paged_kv_indptr[bsz]
            self.unified_paged_kv_last_page_len[bsz] = 0

        with _profile_scope("PREP_QO_INDPTR"):
            self.unified_qo_indptr[0] = 0
            self.unified_qo_indptr[1:bsz + 1] = verify_dec_lens.cumsum(0)
            total_tokens = int(self.unified_qo_indptr[bsz].item())
            if total_tokens > 0:
                bucket_tokens = self._get_bucket_size(total_tokens)
            else:
                bucket_tokens = self.bucket_sizes[0]
            # The dummy row receives the padding tokens.
            self.unified_qo_indptr[bsz + 1] = bucket_tokens

        with _profile_scope("PREP_PLAN"):
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
                disable_split_kv=getattr(self, "deterministic_attention", False),
            )

        with _profile_scope("PREP_INPUT_POS"):
            input_pos = self._input_pos
            input_pos[:bsz] = torch.where(
                is_draft,
                self.draft_cachelens - 1,
                self.cachelens - verify_dec_lens
            ).to(torch.int32)
            input_pos[bsz] = 0

        with _profile_scope("PREP_FLATTEN"):
            sample_indices = torch.repeat_interleave(
                torch.arange(bsz, device=self.device), verify_dec_lens
            )
            offsets = torch.repeat_interleave(self.unified_qo_indptr[:bsz], verify_dec_lens)
            position_indices = torch.arange(total_tokens, device=self.device) - offsets
            flat_tokens = input_ids[sample_indices, position_indices]

        with _profile_scope("PREP_PAD"):
            bucket_view = self._bucket_views[bucket_tokens]
            bucket_view.zero_()
            if total_tokens > 0:
                bucket_view[:total_tokens] = flat_tokens

        # Full fixed-shape buffers (including the dummy row) are returned for torch.compile.
        return (
            bucket_view,
            input_pos,
            self.unified_qo_indptr,
            self.unified_paged_kv_indices,
            self.unified_paged_kv_indptr,
            self.unified_paged_kv_last_page_len,
            total_tokens,
            sample_indices,
            position_indices,
        )

    @torch.inference_mode()
    def unified_inference(self, input_ids: torch.LongTensor, mode_flags: torch.Tensor,
                          verify_dec_lens: torch.Tensor, benchmark=False):
        """One mixed draft/verify step over a padded (batch, max_len) token batch.

        Args:
            mode_flags: (batch_size,) DRAFT_MODE or VERIFY_MODE per row
            verify_dec_lens: (batch_size,) tokens to decode per row
            benchmark: roll back cache lengths after the forward
        """
        bsz = mode_flags.shape[0]
        is_draft = (mode_flags == DRAFT_MODE)
        (
            padded_tokens,
            input_pos,
            kv_append_indptr,
            target_page_indices,
            target_page_indptr,
            target_page_lastlen,
            total_tokens,
            sample_indices,
            position_indices,
        ) = self.prepare_forward_inputs(input_ids, mode_flags, verify_dec_lens)

        # The KV append writes only the new tokens, so draft pages in the
        # attention index list are never written.
        with _profile_scope("FORWARD"):
            bucket_size = padded_tokens.shape[0]
            if hasattr(self, '_cuda_graphs') and bucket_size in self._cuda_graphs:
                self._cuda_graphs[bucket_size].replay()
                if self._greedy_mode:
                    output_flat = self._graph_output[bucket_size][:total_tokens]
                else:
                    topk_vals_flat = self._graph_topk_values[bucket_size][:total_tokens]
                    topk_ids_flat = self._graph_topk_indices[bucket_size][:total_tokens]
                if not hasattr(self, '_graph_replay_count'):
                    self._graph_replay_count = 0
                    self._graph_fallback_count = 0
                self._graph_replay_count += 1
            else:
                result = self.unified_forward(
                    model=self.model,
                    x=padded_tokens,
                    input_pos=input_pos,
                    kv_append_indptr=kv_append_indptr,
                    target_page_indices=target_page_indices,
                    target_page_indptr=target_page_indptr,
                    target_page_lastlen=target_page_lastlen,
                )
                if self._greedy_mode:
                    output_flat = result[:total_tokens]
                else:
                    topk_vals_flat, topk_ids_flat = result
                    topk_vals_flat = topk_vals_flat[:total_tokens]
                    topk_ids_flat = topk_ids_flat[:total_tokens]
                if not hasattr(self, '_graph_fallback_count'):
                    self._graph_replay_count = 0
                    self._graph_fallback_count = 0
                self._graph_fallback_count += 1

        with _profile_scope("POSTPROCESS"):
            max_dec_len = verify_dec_lens.max().item()

            if self._greedy_mode:
                with _profile_scope("POST_SCATTER"):
                    self._output_buffer[:bsz, :max_dec_len].zero_()
                    self._output_buffer[sample_indices, position_indices] = output_flat
                if benchmark:
                    with _profile_scope("ROLLBACK"):
                        self.cachelens -= verify_dec_lens
                        self.draft_cachelens -= is_draft.to(self.draft_cachelens.dtype)
                with _profile_scope("POST_CLONE"):
                    return self._output_buffer[:bsz, :max_dec_len].clone()
            else:
                with _profile_scope("POST_SCATTER"):
                    self._topk_values_buffer[:bsz, :max_dec_len].zero_()
                    self._topk_indices_buffer[:bsz, :max_dec_len].zero_()
                    self._topk_values_buffer[sample_indices, position_indices] = topk_vals_flat
                    self._topk_indices_buffer[sample_indices, position_indices] = topk_ids_flat
                if benchmark:
                    with _profile_scope("ROLLBACK"):
                        self.cachelens -= verify_dec_lens
                        self.draft_cachelens -= is_draft.to(self.draft_cachelens.dtype)
                with _profile_scope("POST_CLONE"):
                    topk_vals_out = self._topk_values_buffer[:bsz, :max_dec_len].clone()
                    topk_ids_out = self._topk_indices_buffer[:bsz, :max_dec_len].clone()
                return topk_vals_out, topk_ids_out

    def set_top_k(self, k: int):
        """Set the output top-k; call before compile()."""
        self._top_k = k
        self._greedy_mode = (k == 0)
        self.model.top_k = k
        if self._greedy_mode:
            self._output_buffer = torch.zeros(
                self.batch_size, self.max_dec_len, dtype=torch.int64, device=self.device)
        else:
            self._topk_values_buffer = torch.zeros(
                self.batch_size, self.max_dec_len, k,
                dtype=torch.float32, device=self.device)
            self._topk_indices_buffer = torch.zeros(
                self.batch_size, self.max_dec_len, k,
                dtype=torch.int64, device=self.device)

    def compile(self):
        """torch.compile the forward functions."""
        import torch._dynamo.config
        import torch._inductor.config

        # torch.compile needs a wrapper with stable workspace pointers (use_cuda_graph=True).
        if self.is_spec and hasattr(self, '_flashinfer_use_cuda_graph') and not self._flashinfer_use_cuda_graph:
            print("Recreating FlashInfer wrapper with use_cuda_graph=True for torch.compile compatibility...")
            self.unified_wrapper = self._create_attention_wrapper(
                self.unified_buffer,
                use_cuda_graph=True,
                qo_indptr_buf=self.unified_qo_indptr,
                kv_indptr_buf=self.unified_paged_kv_indptr,
                kv_indices_buf=self.unified_paged_kv_indices,
                kv_last_page_len_buf=self.unified_paged_kv_last_page_len,
            )
            self._flashinfer_use_cuda_graph = True

            with torch.inference_mode():
                attn_bsz = self.batch_size + 1
                max_pages_per_sample = self.unified_paged_kv_indptr[1].item()
                self.unified_qo_indptr[:attn_bsz] = torch.arange(attn_bsz, dtype=torch.int32, device=self.device) * self.max_dec_len
                self.unified_qo_indptr[attn_bsz] = self.max_unified_tokens
                self.unified_paged_kv_indptr[:attn_bsz] = torch.arange(attn_bsz, dtype=torch.int32, device=self.device) * max_pages_per_sample
                self.unified_paged_kv_indptr[attn_bsz] = self.unified_paged_kv_indptr[attn_bsz - 1]
                self.unified_paged_kv_indices[:self.batch_size * max_pages_per_sample] = torch.arange(
                    self.batch_size * max_pages_per_sample, dtype=torch.int32, device=self.device
                )
                self.unified_paged_kv_last_page_len[:self.batch_size] = self.page_size
                self.unified_paged_kv_last_page_len[self.batch_size] = 0
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
                    disable_split_kv=getattr(self, 'deterministic_attention', False),
                )

        # One compiled graph per bucket.
        num_buckets = len(self.bucket_sizes) if hasattr(self, 'bucket_sizes') else 32
        torch._dynamo.config.cache_size_limit = max(64, num_buckets * 2)

        # CUDA graphs are captured manually in warmup_buckets().
        torch._inductor.config.coordinate_descent_tuning = False
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.triton.cudagraph_trees = False
        torch._functorch.config.enable_autograd_cache = True
        mode = "default"
        self.model_forward = torch.compile(self.model_forward, mode=mode, fullgraph=True)
        if self.is_spec:
            self.draft_forward = torch.compile(self.draft_forward, mode=mode, fullgraph=True)
            self.unified_forward = torch.compile(self.unified_forward, mode=mode, fullgraph=True)

        self._cuda_graphs = {}  # bucket_size -> graph
        self._graph_pool = None
        if self._greedy_mode:
            self._graph_output = {}
        else:
            self._graph_topk_values = {}
            self._graph_topk_indices = {}
        self._graph_replay_count = 0
        self._graph_fallback_count = 0

    def warmup_buckets(self, verbose: bool = True, bucket_sizes: list = None, capture_cuda_graphs: bool = True):
        """Compile each bucket size and capture a CUDA graph for it.
        Call after compile() and encode()."""
        import time

        buckets_to_warmup = bucket_sizes if bucket_sizes is not None else self.bucket_sizes

        if verbose:
            print(f"Warming up {len(buckets_to_warmup)} bucket sizes...")

        bsz = self.batch_size

        saved_cachelens = self.cachelens.clone()
        saved_draft_cachelens = self.draft_cachelens.clone()

        total_warmup_time = 0.0

        mode_flags = torch.full((bsz,), DRAFT_MODE, dtype=torch.int32, device=self.device)

        for bucket in buckets_to_warmup:
            if verbose:
                print(f"  Bucket {bucket:4d}...", end=" ", flush=True)

            torch.cuda.synchronize()
            start = time.perf_counter()

            dec_lens = torch.zeros(bsz, dtype=torch.int32, device=self.device)
            tokens_per_sample = bucket // bsz
            remainder = bucket % bsz
            dec_lens[:] = tokens_per_sample
            dec_lens[:remainder] += 1
            max_dec_len = dec_lens.max().item() if bucket > 0 else 1
            dummy_input = torch.zeros((bsz, max(max_dec_len, 1)), dtype=torch.long, device=self.device)

            with torch.inference_mode():
                (
                    bucket_view, input_pos, qo_indptr, kv_indices, kv_indptr, kv_lastlen,
                    total_tokens, sample_indices, position_indices
                ) = self.prepare_forward_inputs(dummy_input, mode_flags, dec_lens)

                _ = self.unified_forward(
                    model=self.model,
                    x=bucket_view,
                    input_pos=input_pos,
                    kv_append_indptr=qo_indptr,
                    target_page_indices=kv_indices,
                    target_page_indptr=kv_indptr,
                    target_page_lastlen=kv_lastlen,
                )

                if capture_cuda_graphs:
                    self.cachelens.copy_(saved_cachelens)
                    self.draft_cachelens.copy_(saved_draft_cachelens)
                    (
                        bucket_view, input_pos, qo_indptr, kv_indices, kv_indptr, kv_lastlen,
                        total_tokens, sample_indices, position_indices
                    ) = self.prepare_forward_inputs(dummy_input, mode_flags, dec_lens)

                    if self._greedy_mode:
                        static_out = torch.zeros(bucket, dtype=torch.int64, device=self.device)
                        self._graph_output[bucket] = static_out
                        self._register_graph_output("draft", bucket, "tokens", static_out)
                    else:
                        K = self._top_k
                        static_vals = torch.zeros(bucket, K, dtype=torch.float32, device=self.device)
                        static_ids = torch.zeros(bucket, K, dtype=torch.int64, device=self.device)
                        self._graph_topk_values[bucket] = static_vals
                        self._graph_topk_indices[bucket] = static_ids
                        self._register_graph_output("draft", bucket, "topk_values", static_vals)
                        self._register_graph_output("draft", bucket, "topk_indices", static_ids)

                    # One more run outside capture to stabilize allocations.
                    _ = self.unified_forward(
                        model=self.model,
                        x=bucket_view,
                        input_pos=input_pos,
                        kv_append_indptr=qo_indptr,
                        target_page_indices=kv_indices,
                        target_page_indptr=kv_indptr,
                        target_page_lastlen=kv_lastlen,
                    )

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self._graph_pool):
                        result = self.unified_forward(
                            model=self.model,
                            x=bucket_view,
                            input_pos=input_pos,
                            kv_append_indptr=qo_indptr,
                            target_page_indices=kv_indices,
                            target_page_indptr=kv_indptr,
                            target_page_lastlen=kv_lastlen,
                        )
                        if self._greedy_mode:
                            static_out.copy_(result)
                        else:
                            vals, ids = result
                            static_vals.copy_(vals)
                            static_ids.copy_(ids)

                    if self._graph_pool is None:
                        self._graph_pool = graph.pool()

                    self._cuda_graphs[bucket] = graph

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_warmup_time += elapsed

            if verbose:
                if capture_cuda_graphs:
                    print(f"done + captured ({elapsed:.2f}s)")
                else:
                    print(f"done ({elapsed:.2f}s)")

            with torch.inference_mode():
                self.cachelens.copy_(saved_cachelens)
                self.draft_cachelens.copy_(saved_draft_cachelens)

        if verbose:
            print(f"Total warmup time: {total_warmup_time:.2f}s")
            if capture_cuda_graphs:
                print(f"Captured {len(self._cuda_graphs)} CUDA graphs")
                print(f"Graph buckets: {sorted(self._cuda_graphs.keys())}")

    def run_forward(self, model, x, input_pos, kv_append_indptr, target_page_indices, target_page_indptr, target_page_lastlen):
        """Forward with CUDA-graph replay when a graph exists for this bucket."""
        bucket_size = x.shape[0]

        if bucket_size in self._cuda_graphs:
            self._cuda_graphs[bucket_size].replay()
            self._graph_replay_count += 1
            if self._greedy_mode:
                return self._graph_output[bucket_size]
            else:
                return self._graph_topk_values[bucket_size], self._graph_topk_indices[bucket_size]
        else:
            self._graph_fallback_count += 1
            return self.unified_forward(
                model=model,
                x=x,
                input_pos=input_pos,
                kv_append_indptr=kv_append_indptr,
                target_page_indices=target_page_indices,
                target_page_indptr=target_page_indptr,
                target_page_lastlen=target_page_lastlen,
            )
