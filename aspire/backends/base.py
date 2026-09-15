import math

import flashinfer
import torch

from aspire.models.mixed_runner import build_kv_indices_from_block_table
from aspire.models.sparse_draft_runner import (
    DRAFT_MODE,
    LMBackendSparseDraft,
    VERIFY_MODE,
    _profile_scope,
    build_unified_indices,
)
from aspire.attention.holistic_wrapper import build_holistic_dump_jit_args
from aspire.backends.jit_kernels import build_dump_jit_args
from aspire.backends.page_selector import PageSelector
from aspire.attention.flashinfer import (
    compute_max_cta_tile_kv,
    resolve_flashinfer_attention_backend,
)


_UNIFIED_DUMP_OP_INSTANCE_COUNTER = 0


def _next_unified_dump_op_instance_id() -> int:
    """Unique suffix so repeated backend construction in one process does
    not collide on a fixed Torch op name."""
    global _UNIFIED_DUMP_OP_INSTANCE_COUNTER
    out = _UNIFIED_DUMP_OP_INSTANCE_COUNTER
    _UNIFIED_DUMP_OP_INSTANCE_COUNTER += 1
    return out


class LMBackendBase(LMBackendSparseDraft):
    """Shared speculative backend: mixed draft/verify forward, verify-time
    attention dumps, and per-request draft page tables."""

    @torch.inference_mode()
    def setup_caches(
        self,
        max_batch_size: int = 1,
        max_seq_length: int = 2048,
        draft_budget: int = 0,
        window_size: int = 0,
        page_size: int = 1,
        top_k_pages: int = 16,
        L: int = 4,
        use_cuda_graph: bool = True,
        deterministic_attention: bool = False,
        weight_dump_layer: int | None = None,
        top_k: int = 50,
        temperature: float = 0,
        top_p: float = 0,
        holistic_attention: bool = False,
        draft_percent: float | None = None,
    ):
        """Allocate all shared runtime state: page-selection state, dump
        buffers, the full-attention wrapper used for verify-time signal
        collection, and the dump op binding.

        ``draft_percent`` sets the draft cache to X% of each request's
        current cachelen at runtime.  The max allocation (buffer size)
        is derived from ``max_seq_length``; the ``top_k_pages`` arg is
        ignored when ``draft_percent`` is set.
        """
        # Set before super() so LMBackendSparseDraft._create_attention_wrapper()
        # can read it when creating unified_wrapper.
        self._holistic_attention = holistic_attention

        super().setup_caches(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            draft_budget=0,
            window_size=0,
            page_size=page_size,
            use_cuda_graph=use_cuda_graph,
            deterministic_attention=deterministic_attention,
            top_k=top_k,
            temperature=temperature,
            top_p=top_p,
        )

        self._draft_percent = draft_percent
        assert top_k_pages >= L, (
            f"top_k_pages ({top_k_pages}) must be >= L ({L})"
        )
        self._top_k_pages = top_k_pages  # runtime floor for draft pages
        if draft_percent is not None:
            max_draft_pages = max(
                top_k_pages,
                math.ceil(self.max_num_pages_per_request * draft_percent / 100),
            )
        else:
            max_draft_pages = top_k_pages
        self.max_draft_pages = max_draft_pages
        self.L = L

        if not self.is_spec:
            return

        self._dump_op_instance_id = _next_unified_dump_op_instance_id()
        self._draft_num_pages_per_request = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device=self.device,
        )

        self._allocate_dump_state(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
        )
        self.page_selector = self._make_page_selector(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            page_size=page_size,
        )
        self._setup_unified_full_wrapper(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            page_size=page_size,
            use_cuda_graph=use_cuda_graph,
            deterministic_attention=deterministic_attention,
        )
        self._bind_dump_attention_ops(weight_dump_layer)

        import torch._dynamo as dynamo

        dynamo.mark_static_address(self.unified_full_qo_indptr)
        dynamo.mark_static_address(self.unified_full_paged_kv_indices)
        dynamo.mark_static_address(self.unified_full_paged_kv_indptr)
        dynamo.mark_static_address(self.unified_full_paged_kv_last_page_len)

        static = {
            "selection.draft_num_pages": "_draft_num_pages_per_request",
            "dump.attention_weights": "attention_weights_buffer",
            "dump.attention_neg_inf": "attention_weights_buffer_neg_inf",
            "dump.lse": "_lse_buffer",
            "dump.mode_flags": "_dump_mode_flags",
            "dump.workspace": "unified_full_buffer",
            "dump.metadata.qo_indptr": "unified_full_qo_indptr",
            "dump.metadata.kv_indices": "unified_full_paged_kv_indices",
            "dump.metadata.kv_indptr": "unified_full_paged_kv_indptr",
            "dump.metadata.kv_last_page_len": "unified_full_paged_kv_last_page_len",
            "dump.all_verify_flags": "_all_verify_flags",
        }
        for name, attribute in static.items():
            tensor = getattr(self, attribute, None)
            if tensor is not None:
                self._register_static(name, tensor)
        self._register_page_selector_static_buffers()

    def _register_page_selector_static_buffers(self):
        selector = self.page_selector
        names = {
            "j": "_j",
            "scores": "_scores",
            "position_bias": "_pos_bias",
            "selected": "_selected_buf",
            "column_ids": "_col_ids",
            "valid_counts": "_valid_counts_buf",
            "head_max": "_head_max_buf",
            "gather_indices": "_gather_idx",
            "recent_offsets": "_recent_offsets",
        }
        for public_name, attribute in names.items():
            tensor = getattr(selector, attribute, None)
            if tensor is not None:
                self._register_static(f"selection.{public_name}", tensor)

    def _make_page_selector(
        self,
        *,
        max_batch_size: int,
        max_seq_length: int,
        page_size: int,
    ):
        """Construct the page selector used after verify-time attention dumps.

        Subclasses override this to score pages from a different dumped signal.
        """
        return PageSelector(
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

    def _dump_query_slots(self) -> int:
        """Return how many query positions each dump writes per head.

        The base backend dumps only the last verify token, so one slot is
        enough. The aspire backend overrides this to reserve two
        slots for first+last compact dumps.
        """
        return 1

    def _dump_kind(self) -> str:
        """Name the JIT dump layout consumed by build_dump_jit_args().

        This keeps dump-kernel selection declarative at the backend level
        instead of hard-coding kernel variants inside setup logic.
        """
        return "last_token"

    def _dump_jit_name_suffix(self) -> str:
        """JIT symbol suffix so backends with different CUDA structs are not
        compiled under the same FlashInfer symbol name."""
        return ""

    def _dump_buffer_dtype(self) -> torch.dtype:
        """Return the dtype used for the dumped attention-logit buffer."""
        return torch.float16

    def _build_dump_jit_args(self):
        """Build FlashInfer JIT metadata for the active dump kernel shape.

        The base wrapper setup does not know whether a subclass wants last-only
        dumps, first+last dumps, or mode-flag gating, so this helper packages
        those hook values into the single jit_args list FlashInfer expects.
        """
        suffix = self._dump_jit_name_suffix()
        if self._use_holistic_attention():
            class_name = "HolisticDumpAttentionWeights" + suffix
            op_name = "holistic_batch_dump_attention_weights"
            if suffix:
                op_name = f"{op_name}_{suffix.lower()}"
            return build_holistic_dump_jit_args(
                dtype=self.dtype,
                head_dim=self.model.config.head_dim,
                dump_kind=self._dump_kind(),
                class_name=class_name,
                op_name=op_name,
                use_mode_flags=self._dump_uses_mode_flags(),
                dump_buffer_dtype=self._dump_buffer_dtype(),
            )
        class_name = "BatchDumpAttentionWeights" + suffix
        op_name = "batch_prefill_dump_attention_weights"
        if suffix:
            op_name = f"{op_name}_{suffix.lower()}"
        return build_dump_jit_args(
            backend=self._jit_backend,
            dtype=self.dtype,
            head_dim=self.model.config.head_dim,
            dump_kind=self._dump_kind(),
            class_name=class_name,
            op_name=op_name,
            use_mode_flags=self._dump_uses_mode_flags(),
            dump_buffer_dtype=self._dump_buffer_dtype(),
        )

    def _dump_torch_op_name(self) -> str:
        """Return the base Torch op name bound into the model dump layer.

        The actual registered op name is made unique per instance later, but
        subclasses still need a semantic base name so different behaviors are
        easy to distinguish.
        """
        return "mylib::unified_full_decode"

    def _dump_uses_mode_flags(self) -> bool:
        """Whether the dump kernel takes a per-sample enable flag."""
        return False

    def _dump_run_extra_scalars(self) -> list:
        """Return the extra scalar args appended to wrapper.run() for JIT dump kernels.

        FA2 kernels need sm_scale as an additional scalar; the holistic wrapper
        has sm_scale built into PersistentParams, so it is omitted.
        """
        if self._use_holistic_attention():
            return [self._jit_num_heads, self._jit_kv_stride]
        return [self._jit_sm_scale, self._jit_num_heads, self._jit_kv_stride]

    def _allocate_dump_state(self, *, max_batch_size: int, max_seq_length: int):
        """Allocate buffers that hold dumped logits, LSE, and per-sample dump modes.

        Preallocated so the model can collect verify-time attention signals
        without allocating tensors inside the decode loop.

        The KV stride is padded to ``CTA_TILE_KV`` alignment so that the JIT
        logits-transform callback can omit the ``kv_idx < kv_len`` bounds
        check without risking out-of-bounds writes in the last KV tile.

        Guard-removal safety contract (see ``jit_kernels.py`` for details):
            The kernel's last CTA tile writes up to CTA_TILE_KV − 1 positions
            past kv_len.  Correctness relies on four co-operating invariants:

            1. Stride padding (this method) — prevents out-of-bounds writes.
            2. −inf fill / reset — ``reset_attention_weights_buffer()`` sets the
               entire buffer to −inf.  This serves two purposes: (a) untouched
               positions beyond the last CTA tile produce zero after exp2, and
               (b) for the ``first_last`` kernel, slot 1 is never written when
               ``qo_len == 1``, so stale logits at valid kv positions would
               corrupt page scores if not cleared.  Positions inside the CTA
               overshoot ARE overwritten with garbage and are NOT protected by
               −inf.
            3. Narrowing on read — ``PageSelector._view_buffer`` slices away the
               padding tail ``[max_seq_length, kv_stride_alloc)``.
            4. Per-sample ``num_pages`` — ``_finalize_scores`` excludes pages
               beyond each sample's actual kv_len, making any residual garbage
               in the last partial page harmless.
        """
        head_dim = self.model.config.head_dim
        pad = compute_max_cta_tile_kv(
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            dtype_bytes=self.dtype.itemsize,
        )
        self._kv_stride_alloc = ((max_seq_length + pad - 1) // pad) * pad

        max_heads = self.model.config.n_head
        dump_slots = self._dump_query_slots()
        decode_attention_elements = max_batch_size * max_heads * dump_slots * self._kv_stride_alloc
        self.attention_weights_buffer = torch.full(
            (int(decode_attention_elements),),
            float("-inf"),
            dtype=self._dump_buffer_dtype(),
            device=self.device,
        )
        self.attention_weights_buffer_neg_inf = torch.tensor(
            float("-inf"),
            dtype=self.attention_weights_buffer.dtype,
            device=self.attention_weights_buffer.device,
        )
        max_total_q = max_batch_size * self.dec_len
        self._lse_buffer = torch.zeros(
            max_total_q,
            max_heads,
            dtype=torch.float32,
            device=self.device,
        )
        self._dump_mode_flags = torch.full(
            (max_batch_size,),
            VERIFY_MODE,
            dtype=torch.int32,
            device=self.device,
        )

    def _setup_unified_full_wrapper(
        self,
        *,
        max_batch_size: int,
        max_seq_length: int,
        page_size: int,
        use_cuda_graph: bool,
        deterministic_attention: bool,
    ):
        """Create the full-attention FlashInfer wrapper used only for signal dumps.

        The main unified wrapper follows draft/verify mixed indices, but page
        selection needs a second wrapper that can replay a dump-capable attention
        program on verify context. This method allocates and warm-plans that
        secondary wrapper once during setup.
        """
        self.unified_full_buffer = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=self.device,
        )
        self.unified_full_qo_indptr = torch.zeros(
            max_batch_size + 2,
            dtype=torch.int32,
            device=self.device,
        )
        # Sized like the main wrapper's index buffer.
        self.unified_full_paged_kv_indices = torch.empty(
            self.unified_paged_kv_indices.shape[0],
            dtype=torch.int32,
            device=self.device,
        )
        self.unified_full_paged_kv_indptr = torch.zeros(
            max_batch_size + 2,
            dtype=torch.int32,
            device=self.device,
        )
        self.unified_full_paged_kv_last_page_len = torch.zeros(
            max_batch_size + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self._all_verify_flags = torch.full(
            (max_batch_size,),
            VERIFY_MODE,
            dtype=torch.int32,
            device=self.device,
        )

        self._jit_backend = resolve_flashinfer_attention_backend(self.dtype)
        self._unified_full_jit_args = self._build_dump_jit_args()

        self.unified_full_wrapper = self._create_attention_wrapper(
            self.unified_full_buffer,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=self.unified_full_qo_indptr if use_cuda_graph else None,
            kv_indptr_buf=self.unified_full_paged_kv_indptr if use_cuda_graph else None,
            kv_indices_buf=self.unified_full_paged_kv_indices if use_cuda_graph else None,
            kv_last_page_len_buf=self.unified_full_paged_kv_last_page_len if use_cuda_graph else None,
            jit_args=self._unified_full_jit_args,
        )

        self._jit_sm_scale = 1.0 / math.sqrt(self.model.config.head_dim)
        self._jit_num_heads = self.model.config.n_head
        self._jit_kv_stride = self._kv_stride_alloc

        attn_bsz = max_batch_size + 1
        max_pages_per_sample = self.max_num_pages_per_request
        self.unified_full_qo_indptr[:attn_bsz] = torch.arange(
            attn_bsz,
            dtype=torch.int32,
            device=self.device,
        ) * self.dec_len
        self.unified_full_qo_indptr[attn_bsz] = max_batch_size * self.dec_len
        self.unified_full_paged_kv_indptr[:attn_bsz] = torch.arange(
            attn_bsz,
            dtype=torch.int32,
            device=self.device,
        ) * max_pages_per_sample
        self.unified_full_paged_kv_indptr[attn_bsz] = self.unified_full_paged_kv_indptr[attn_bsz - 1]
        self.unified_full_paged_kv_indices[: max_batch_size * max_pages_per_sample] = torch.arange(
            max_batch_size * max_pages_per_sample,
            dtype=torch.int32,
            device=self.device,
        )
        self.unified_full_paged_kv_last_page_len[:max_batch_size] = page_size
        self.unified_full_paged_kv_last_page_len[max_batch_size] = 0

        self.unified_full_wrapper.plan(
            qo_indptr=self.unified_full_qo_indptr,
            paged_kv_indptr=self.unified_full_paged_kv_indptr,
            paged_kv_indices=self.unified_full_paged_kv_indices,
            paged_kv_last_page_len=self.unified_full_paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=page_size,
            q_data_type=self.dtype,
            causal=True,
            disable_split_kv=deterministic_attention,
        )

    def _register_dump_kernel_op(self):
        """Register the low-level FlashInfer dump op for this backend instance.

        The registered op is intentionally narrow: it forwards directly into the
        FlashInfer wrapper and returns the LSE tensor. Higher-level bookkeeping
        such as mode gating, per-layer state updates, and post-run caching stays
        in Python wrappers attached to `attn_unified`, which makes that logic
        visible to `torch.compile` instead of hiding it inside the custom op.
        """
        op_name = f"{self._dump_torch_op_name()}_{self._dump_op_instance_id}"
        torch.library.define(
            op_name,
            "(Tensor q, Tensor kv_cache, Tensor(a!) dump_buffer, Tensor(b!) out) -> Tensor",
        )

        @torch.library.impl(op_name, "cuda")
        def unified_dump_kernel(q, kv_cache, dump_buffer, out):
            run_args = [q, kv_cache, dump_buffer]
            if self._dump_uses_mode_flags():
                run_args.append(self._dump_mode_flags)
            run_args.extend(self._dump_run_extra_scalars())
            _, lse = self.unified_full_wrapper.run(
                *run_args,
                return_lse=True,
                out=out,
            )
            return lse

        @torch.library.register_fake(op_name)
        def unified_dump_kernel_abstract(q, kv_cache, dump_buffer, out):
            return torch.empty(
                (q.shape[0], self._jit_num_heads),
                dtype=torch.float32,
                device=q.device,
            )

        opname = op_name.split("::", 1)[1]
        return getattr(torch.ops.mylib, opname)

    def _dump_buffer_for_step(self) -> torch.Tensor:
        """Destination tensor for the dump kernel this step."""
        return self.attention_weights_buffer

    def _make_dump_attention_wrapper(self, dump_kernel_op):
        """Build the callable bound into `attention.attn_unified` for dump layers.

        The wrapper owns backend-local state updates and invokes the registered
        low-level op only for the actual FlashInfer execution. Keeping this as a
        Python callable lets `torch.compile` see the surrounding tensor logic.
        """
        def attn_unified_wrapper(q, kv_cache, out):
            lse = dump_kernel_op(q, kv_cache, self._dump_buffer_for_step(), out)
            self._consume_dump_outputs(lse)

        return attn_unified_wrapper

    def _consume_dump_outputs(self, lse: torch.Tensor):
        """Store dump outputs needed by the default page-selection path.

        Only the latest LSE tensor is kept; the dumped logits already live in
        attention_weights_buffer.
        """
        self._lse_buffer[: lse.shape[0]] = lse

    def _resolve_weight_dump_layer(self, weight_dump_layer: int | None) -> int:
        """Choose which transformer layer is responsible for producing dump signal.

        Dumping from one layer keeps the runtime cheap. This helper centralizes
        the defaulting/clamping logic so setup and subclasses do not duplicate
        layer-index validation.
        """
        num_layers = len(self.model.layers)
        if weight_dump_layer is None:
            return num_layers - 1
        if weight_dump_layer < 0:
            weight_dump_layer = num_layers + weight_dump_layer
        return min(max(weight_dump_layer, 0), num_layers - 1)

    def _bind_dump_attention_ops(self, weight_dump_layer: int | None):
        """Attach the dump-layer wrapper used by unified backends.

        The selected transformer layer receives a Python wrapper closure that
        performs backend bookkeeping and then calls a low-level registered op
        which only runs the FlashInfer dump kernel.
        """
        dump_kernel_op = self._register_dump_kernel_op()
        self._weight_dump_layer = self._resolve_weight_dump_layer(weight_dump_layer)
        self.model.layers[self._weight_dump_layer].attention.attn_unified = self._make_dump_attention_wrapper(
            dump_kernel_op
        )

    @torch.inference_mode()
    def reset_attention_weights_buffer(self):
        """Reset the dump buffer to `-inf` between benchmark/debug uses.

        Dump kernels overwrite only the entries they touch, so callers need a
        cheap way to restore the sentinel value before a fresh capture.
        """
        self.attention_weights_buffer.fill_(self.attention_weights_buffer_neg_inf)

    @torch.inference_mode()
    def encode(self, input_ids, benchmark=False, max_prefill_tokens: int = 8192):
        """Encode a batch of prompts by assigning them to the first `n` slots.

        The chunked-prefill logic lives in `encode_slots`.
        """
        n = len(input_ids) if isinstance(input_ids, (list, tuple)) else input_ids.shape[0]
        all_slots = torch.arange(n, device=self.device)
        return self.encode_slots(all_slots, input_ids, max_prefill_tokens=max_prefill_tokens)

    def _init_draft_indices(self):
        """Seed every active slot with a simple recent-pages draft cache.

        Before any verify-driven replanning has happened, the backend still
        needs a legal draft page set. The base policy starts with the most recent
        pages so speculative decoding can begin immediately after prefill.
        """
        self._draft_indices.zero_()
        self._draft_num_pages_per_request.zero_()
        slot_ids = torch.arange(self.batch_size, device=self.device, dtype=torch.long)
        rows, valid_counts = self._build_recent_draft_rows(slot_ids)
        self._store_draft_rows(
            slot_ids,
            rows,
            valid_counts,
            count_buffer=self._draft_num_pages_per_request,
        )

    def _replan_draft_indices(self):
        """Refresh draft page tables for the currently active batch.

        Runs the selector on the latest dumped signal and updates every slot.
        The whole-batch form is shape-static over preallocated buffers, so
        under CUDA-graph execution it is captured once and replayed.
        """
        bsz = int(getattr(self, "_current_replan_bsz", self.batch_size))
        if bsz == self.page_selector._bsz:
            graph = self._replan_graph_for_whole_batch()
            with _profile_scope("REPLAN_SELECT"):
                if graph is not None:
                    graph.replay()
                    if getattr(self, "_replan_graph_check", None) is None:
                        import os

                        self._replan_graph_check = (
                            os.environ.get("ASPIRE_REPLAN_GRAPH_CHECK", "0") == "1"
                        )
                    if self._replan_graph_check:
                        self._check_replan_replay()
                else:
                    self._whole_batch_replan_impl()
            return
        # Partial-batch fallback (warm-up and tests only).
        slot_ids = torch.arange(bsz, device=self.device)
        with _profile_scope("REPLAN_SELECT"):
            selected, valid_counts = self.page_selector.select_subset(
                slot_ids,
                self.attention_weights_buffer,
                self._lse_buffer,
                self.num_pages_per_request,
                self.target_block_table,
                self.unified_full_qo_indptr,
            )
        self._store_draft_rows(
            slot_ids,
            selected,
            valid_counts,
            count_buffer=self._draft_num_pages_per_request,
        )

    def _check_replan_replay(self):
        """Debug gate (ASPIRE_REPLAN_GRAPH_CHECK=1): after a graph replay,
        recompute the replan eagerly and assert the outputs match. Synchronizes
        the device — validation runs only."""
        replay_table = self._draft_indices.clone()
        replay_counts = self._draft_num_pages_per_request.clone()
        self._whole_batch_replan_impl()
        torch.cuda.synchronize(self.device)
        if not torch.equal(replay_table, self._draft_indices) or not torch.equal(
            replay_counts, self._draft_num_pages_per_request
        ):
            bad = (replay_table != self._draft_indices).any(dim=1)
            rows = bad.nonzero().flatten().tolist()
            detail = ""
            if rows:
                r = rows[0]
                detail = (
                    f"\n  row {r} replay: {replay_table[r].tolist()}"
                    f"\n  row {r} eager:  {self._draft_indices[r].tolist()}"
                )
            raise RuntimeError(
                "replan graph replay diverged from eager recompute; rows "
                f"{rows}, counts replay {replay_counts.tolist()} vs eager "
                f"{self._draft_num_pages_per_request.tolist()}{detail}"
            )

    def _whole_batch_replan_impl(self):
        """The graphable whole-batch replan body: select over buffer views and
        write the full table prefix. select() reads the dump buffer as a view,
        avoiding select_subset's full-buffer gather; values are identical
        because the subset path with slot_ids == arange(bsz) gathers the same
        rows in the same order."""
        selected, valid_counts = self.page_selector.select(
            self.attention_weights_buffer,
            self._lse_buffer,
            self.num_pages_per_request,
            self.target_block_table,
            self.unified_full_qo_indptr,
        )
        self._store_draft_rows(
            None,
            selected,
            valid_counts,
            count_buffer=self._draft_num_pages_per_request,
            full_prefix=self.page_selector._bsz,
        )

    def _replan_graph_for_whole_batch(self):
        """Capture (once, lazily) and return the whole-batch replan graph.

        Only active when the backend itself runs CUDA graphs; the first hot
        call performs two stabilizing eager runs and then captures into the
        shared graph memory pool. Every input and output of the body is a
        preallocated static buffer, so replay is bit-identical to the eager
        form by construction. ``ASPIRE_REPLAN_GRAPH=0`` disables capture.
        """
        graph = getattr(self, "_replan_graph", None)
        if graph is not None:
            return graph
        if getattr(self, "_replan_graph_disabled", False):
            return None
        import os

        if os.environ.get("ASPIRE_REPLAN_GRAPH", "1") != "1":
            self._replan_graph_disabled = True
            return None
        # Not latched: the main graphs fill in during warm-up, after which
        # capture becomes worthwhile (and safe to share their memory pool).
        if not getattr(self, "_cuda_graphs", None):
            return None
        # Stabilize transient allocations, then capture. Capture records
        # without executing, so one replay immediately after keeps this
        # step's outputs in place. The graph gets its OWN memory pool:
        # sharing the model graphs' pool would require replays to follow the
        # capture order, and replan replays interleave with model replays
        # every step — sharing corrupts pool-allocated intermediates.
        self._whole_batch_replan_impl()
        self._whole_batch_replan_impl()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._whole_batch_replan_impl()
        graph.replay()
        self._replan_graph = graph
        return graph

    def _prepare_replan_state(self, bsz: int, verify_num_pages: torch.Tensor):
        """Stage per-step metadata needed by `_replan_draft_indices()`.

        Replanning depends on the current number of valid verify pages and the
        active batch size; the whole batch is refreshed.
        """
        self.num_pages_per_request[:bsz] = verify_num_pages
        self._current_replan_bsz = int(bsz)
        self._replan_draft_indices()

    def _handle_draft_page_spill(
        self,
        spill: torch.Tensor,
        draft_pages_needed: torch.Tensor,
        draft_num_pages: torch.Tensor,
    ):
        """Append a spill page to draft indices for samples that crossed a page boundary.

        The new page is at logical index ``draft_pages_needed - 1`` in the
        block table (the page containing the latest draft token).
        """
        spill_ids = spill.nonzero(as_tuple=True)[0]
        new_page_logical = (draft_pages_needed[spill_ids] - 1).long()
        new_page_ids = self.target_block_table[
            spill_ids, new_page_logical
        ].to(self._draft_indices.dtype)
        insert_pos = draft_num_pages[spill_ids].long()
        max_width = self._draft_indices.shape[1]
        can_insert = insert_pos < max_width
        if can_insert.any():
            ok = spill_ids[can_insert]
            pos = insert_pos[can_insert]
            self._draft_indices[ok, pos] = new_page_ids[can_insert]
            draft_num_pages[ok] += 1

    def _handle_draft_page_spill_ids(self, spill_ids: torch.Tensor, bsz: int):
        """Index-driven spill insert for host-identified rows.

        The caller (host plan mirror) guarantees every row both needs the
        spill and has table capacity, so the tail comparison, `any()` and
        `nonzero()` of the mask-driven path are unnecessary. Returns the
        insert positions and page ids so subclasses can extend per-layer
        tables without recomputing them.
        """
        needed = self._get_num_pages(self.draft_cachelens[:bsz][spill_ids]).to(torch.long)
        new_page_ids = self.target_block_table[
            spill_ids, (needed - 1).clamp(min=0)
        ].to(self._draft_indices.dtype)
        insert_pos = self._draft_num_pages_per_request[spill_ids].long()
        self._draft_indices[spill_ids, insert_pos] = new_page_ids
        self._draft_num_pages_per_request[spill_ids] += 1
        return insert_pos, new_page_ids

    def _upload_row_ids(self, rows, key: str) -> torch.Tensor:
        """Asynchronously upload a small host row-index list to the device.

        Uses per-key pinned staging + a persistent device buffer so the copy
        is a true async H2D (a pageable-memory copy would drain the stream).
        """
        import numpy as np

        cache = getattr(self, "_row_id_staging", None)
        if cache is None:
            cache = {}
            self._row_id_staging = cache
        buffers = cache.get(key)
        if buffers is None:
            pinned = torch.empty(self.batch_size, dtype=torch.long, pin_memory=True)
            device = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
            buffers = (pinned, device)
            cache[key] = buffers
        pinned, device = buffers
        count = len(rows)
        pinned[:count] = torch.from_numpy(np.ascontiguousarray(rows))
        device[:count].copy_(pinned[:count], non_blocking=True)
        return device[:count]

    def _step_upload(self, host_plan: dict, total_tokens: int) -> dict:
        """Upload host-rendered step metadata through an event-fenced pinned ring.

        Copies land directly in the buffers the rest of the step reads
        (`unified_qo_indptr`, `_input_pos`, and dedicated int buffers for page
        counts and flatten indices).
        """
        ring = getattr(self, "_upload_ring", None)
        if ring is None:
            mb = self.batch_size
            max_tokens = self.max_bucket_size
            ring = []
            for _ in range(4):
                ring.append(
                    {
                        "qo": torch.zeros(mb + 2, dtype=torch.int32, pin_memory=True),
                        "i32": torch.zeros((4, mb), dtype=torch.int32, pin_memory=True),
                        "idx": torch.zeros((2, max_tokens), dtype=torch.int64, pin_memory=True),
                        "event": torch.cuda.Event(),
                        "used": False,
                    }
                )
            self._upload_ring = ring
            self._upload_slot = 0
            self._up_i32 = torch.zeros((4, mb), dtype=torch.int32, device=self.device)
            self._up_idx = torch.zeros(
                (2, max_tokens), dtype=torch.int64, device=self.device
            )
        slot = ring[self._upload_slot]
        self._upload_slot = (self._upload_slot + 1) % len(ring)
        if slot["used"]:
            slot["event"].synchronize()

        mb = self.batch_size
        slot["qo"].numpy()[:] = host_plan["qo_indptr"].numpy()
        i32 = slot["i32"].numpy()
        # Per-row plan arrays are sized to the actual generate batch, which
        # may be smaller than the engine's max batch; zero the tail rows.
        n = host_plan["verify_num_pages"].shape[0]
        i32[:, n:] = 0
        i32[0, :n] = host_plan["verify_num_pages"]
        i32[1, :n] = host_plan["effective_num_pages"]
        i32[2, :n] = host_plan["last_page_len"]
        i32[3, :n] = host_plan["input_pos"]
        idx = slot["idx"].numpy()
        idx[0, :total_tokens] = host_plan["sample_indices"]
        idx[1, :total_tokens] = host_plan["position_indices"]

        self.unified_qo_indptr.copy_(slot["qo"], non_blocking=True)
        self._up_i32.copy_(slot["i32"], non_blocking=True)
        self._input_pos[:mb].copy_(slot["i32"][3], non_blocking=True)
        if total_tokens > 0:
            self._up_idx[:, :total_tokens].copy_(
                slot["idx"][:, :total_tokens], non_blocking=True
            )
        slot["event"].record()
        slot["used"] = True
        return {
            "num_pages": self._up_i32[0],
            "effective_num_pages": self._up_i32[1],
            "last_page_len": self._up_i32[2],
            "sample_indices": self._up_idx[0],
            "position_indices": self._up_idx[1],
        }

    def make_host_plan_state(self):
        """Build the CPU plan mirror matching this backend's semantics.

        Returns None when the backend has no speculative page state to mirror.
        The `_host_plan_draft_kind` / `_host_plan_full_kind` class attributes
        encode the backend's refresh rule (see host_plan.HostPlanConfig).
        """
        if not self.is_spec or not hasattr(self, "page_selector"):
            return None
        from aspire.backends.host_plan import HostPlanConfig, HostPlanState

        selector = self.page_selector
        return HostPlanState(
            HostPlanConfig(
                page_size=self.page_size,
                max_batch=self.batch_size,
                draft_kind=self._host_plan_draft_kind,
                full_kind=self._host_plan_full_kind,
                subset_cache_size=selector.subset_cache_size,
                table_width=self._draft_indices.shape[1],
                min_draft_pages=selector._min_draft_pages,
                draft_percent=selector._draft_percent,
                valid_floor_l=selector.L,
                pages_per_slot=self.max_num_pages_per_request,
            )
        )

    _host_plan_draft_kind = "replan"
    _host_plan_full_kind = "effective_full"

    def _plan_parity_enabled(self) -> bool:
        cached = getattr(self, "_plan_parity_flag", None)
        if cached is None:
            import os

            cached = os.environ.get("ASPIRE_PLAN_PARITY", "0") == "1"
            self._plan_parity_flag = cached
        return cached

    def _verify_host_plan_parity(
        self,
        host_plan: dict,
        bsz: int,
        mode_flags=None,
        verify_dec_lens=None,
        uploads=None,
        total_tokens=None,
    ):
        """Debug gate: assert the host plan arrays equal the device-derived ones.

        Synchronizes the device — enable only for validation runs
        (``ASPIRE_PLAN_PARITY=1``).
        """
        from flashinfer.page import get_seq_lens

        checks = [
            ("qo_indptr", self.unified_qo_indptr, host_plan["qo_indptr"]),
            ("kv_indptr", self.unified_paged_kv_indptr, host_plan["kv_indptr"]),
            (
                "kv_len",
                get_seq_lens(
                    self.unified_paged_kv_indptr,
                    self.unified_paged_kv_last_page_len,
                    self.page_size,
                ),
                host_plan["kv_len"],
            ),
        ]
        if host_plan["full_qo_indptr"] is not None:
            checks += [
                ("full_qo_indptr", self.unified_full_qo_indptr, host_plan["full_qo_indptr"]),
                ("full_kv_indptr", self.unified_full_paged_kv_indptr, host_plan["full_kv_indptr"]),
            ]
            if uploads is not None and host_plan["full_kv_indptr"] is host_plan["kv_indptr"]:
                pass  # 'mirror' kind: covered by the mixed kv_len check above.
            elif uploads is not None:
                # Static full-index layout: the device indptr carries the
                # fixed stride, so kv_len is validated against the uploaded
                # (device-recomputed above) effective pages and lengths.
                ref_full_len = (
                    (uploads["effective_num_pages"][:bsz].to(torch.int64) - 1).clamp(min=0)
                    * self.page_size
                    + self.unified_full_paged_kv_last_page_len[:bsz].to(torch.int64)
                )
                checks.append(
                    ("full_kv_len", ref_full_len, host_plan["full_kv_len"][:bsz])
                )
            else:
                checks.append(
                    (
                        "full_kv_len",
                        get_seq_lens(
                            self.unified_full_paged_kv_indptr,
                            self.unified_full_paged_kv_last_page_len,
                            self.page_size,
                        ),
                        host_plan["full_kv_len"],
                    )
                )
        if "draft_num_pages" in host_plan:
            checks.append(
                (
                    "draft_num_pages",
                    self._draft_num_pages_per_request[:bsz],
                    torch.from_numpy(host_plan["draft_num_pages"][:bsz].copy()),
                )
            )
        if uploads is not None and mode_flags is not None and verify_dec_lens is not None:
            # Recompute the device formulas and compare against the
            # uploaded metadata.
            is_draft = mode_flags == DRAFT_MODE
            cachelens = self.cachelens[:bsz]
            draft_cachelens = self.draft_cachelens[:bsz]
            effective = torch.where(is_draft, draft_cachelens, cachelens)
            ref_pos = torch.where(
                is_draft, draft_cachelens - 1, cachelens - verify_dec_lens
            ).to(torch.int32)
            checks += [
                ("up_num_pages", self._get_num_pages(cachelens).to(torch.int32), uploads["num_pages"][:bsz]),
                ("up_eff_num_pages", self._get_num_pages(effective).to(torch.int32), uploads["effective_num_pages"][:bsz]),
                ("up_last_page_len", self._get_last_page_len(effective).to(torch.int32), uploads["last_page_len"][:bsz]),
                ("up_input_pos", ref_pos, self._input_pos[:bsz]),
            ]
            if total_tokens:
                ref_sample = torch.repeat_interleave(
                    torch.arange(bsz, device=self.device),
                    verify_dec_lens,
                    output_size=total_tokens,
                )
                ref_offsets = torch.repeat_interleave(
                    self.unified_qo_indptr[:bsz],
                    verify_dec_lens,
                    output_size=total_tokens,
                )
                ref_position = (
                    torch.arange(total_tokens, device=self.device) - ref_offsets
                )
                checks += [
                    ("up_sample_indices", ref_sample, uploads["sample_indices"][:total_tokens]),
                    ("up_position_indices", ref_position, uploads["position_indices"][:total_tokens]),
                ]
        for name, device_tensor, host_tensor in checks:
            device_cpu = device_tensor.detach().to("cpu", dtype=torch.int64)
            host_cpu = host_tensor.detach().to("cpu", dtype=torch.int64)
            if not torch.equal(device_cpu[: host_cpu.numel()], host_cpu[: device_cpu.numel()]):
                raise RuntimeError(
                    f"host plan parity failure on '{name}':\n"
                    f"  device: {device_cpu.tolist()}\n"
                    f"  host:   {host_cpu.tolist()}"
                )

    def _prepare_patch_metadata(self, bsz: int, is_draft: torch.Tensor):
        """Hook for per-step patch metadata; unused because all layers share
        one draft table."""
        return None

    def _plan_step_dump_wrapper(
        self,
        *,
        bsz: int,
        verify_num_pages: torch.Tensor,
        last_page_len: torch.Tensor,
        mode_flags: torch.Tensor,
    ):
        """Prepare the dump wrapper for the next model step.

        The dump wrapper needs its own planned KV indices because it may run a
        different attention program than the main mixed wrapper. The base version
        plans full verify-page attention so page selection sees the full context.
        """
        self.unified_full_qo_indptr[: bsz + 1].copy_(
            self.unified_qo_indptr[: bsz + 1]
        )
        self.unified_full_qo_indptr[bsz + 1 :] = self.unified_qo_indptr[-1]
        host_plan = getattr(self, "_prep_host_plan", None)
        if (
            host_plan is not None
            and host_plan.get("full_qo_indptr") is not None
            and hasattr(self.unified_full_wrapper, "plan_host")
        ):
            # Static full-index layout: the block table is a contiguous
            # arange, so the setup-time index buffer (row i's pages at offset
            # i*pages_per_slot) already addresses every page the full-context
            # dump attends to; the host plan carries the strided indptr and
            # the exact kv lengths. The kernel reads the identical pages in
            # the identical order as the packed per-step build, so attention
            # is bitwise unchanged while the Triton build and indptr fills
            # disappear from the hot path.
            self._ensure_static_full_indices()
            self.unified_full_paged_kv_last_page_len[:bsz].copy_(last_page_len)
            self.unified_full_paged_kv_last_page_len[bsz:] = 0
            self._plan_full_wrapper_from_buffers()
            return
        self._full_static_indices_dirty = True
        build_unified_indices(
            mode_flags=self._all_verify_flags[:bsz],
            draft_indices=self.target_block_table,
            verify_indices=self.target_block_table,
            draft_num_pages=verify_num_pages,
            verify_num_pages=verify_num_pages,
            paged_kv_indices=self.unified_full_paged_kv_indices,
            paged_kv_indptr=self.unified_full_paged_kv_indptr,
        )
        self.unified_full_paged_kv_indptr[bsz + 1 :] = (
            self.unified_full_paged_kv_indptr[bsz]
        )
        self.unified_full_paged_kv_last_page_len[:bsz].copy_(last_page_len)
        self.unified_full_paged_kv_last_page_len[bsz:] = 0
        self._plan_full_wrapper_from_buffers()

    def _ensure_static_full_indices(self):
        """Restore the setup-time static full-index layout after a packed
        build overwrote it (warm-up, tests). No-op on the hot path."""
        if not getattr(self, "_full_static_indices_dirty", False):
            return
        mb = self.batch_size
        pages = self.max_num_pages_per_request
        self.unified_full_paged_kv_indices[: mb * pages] = torch.arange(
            mb * pages, dtype=torch.int32, device=self.device
        )
        self.unified_full_paged_kv_indptr[: mb + 1] = (
            torch.arange(mb + 1, dtype=torch.int32, device=self.device) * pages
        )
        self.unified_full_paged_kv_indptr[mb + 1 :] = mb * pages
        self._full_static_indices_dirty = False

    def _plan_full_wrapper_from_buffers(self):
        """Call FlashInfer `plan()` on the already-populated full-wrapper buffers.

        With a host plan available the sync-free `plan_host` path is used; the
        device index buffers are still populated because the kernel reads them
        at run time.
        """
        host_plan = getattr(self, "_prep_host_plan", None)
        if (
            host_plan is not None
            and host_plan["full_qo_indptr"] is not None
            and hasattr(self.unified_full_wrapper, "plan_host")
        ):
            self.unified_full_wrapper.plan_host(
                host_plan["full_qo_indptr"],
                host_plan["full_kv_indptr"],
                host_plan["full_kv_len"],
                self.unified_full_paged_kv_indices,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=self.page_size,
                causal=True,
            )
            return
        self.unified_full_wrapper.plan(
            qo_indptr=self.unified_full_qo_indptr,
            paged_kv_indptr=self.unified_full_paged_kv_indptr,
            paged_kv_indices=self.unified_full_paged_kv_indices,
            paged_kv_last_page_len=self.unified_full_paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head,
            num_kv_heads=self.model.config.n_local_heads,
            head_dim_qk=self.model.config.head_dim,
            page_size=self.page_size,
            q_data_type=self.dtype,
            causal=True,
            disable_split_kv=getattr(self, "deterministic_attention", False),
        )

    def _after_prepare_step(self, mode_flags: torch.Tensor, bsz: int):
        """Record per-sample dump modes after the step inputs are prepared.

        The dump kernel may need to know which samples are verify vs draft so it
        can suppress unnecessary writes. Capturing that state here keeps the
        model-side callback simple.
        """
        self._dump_mode_flags[:bsz].copy_(mode_flags.to(torch.int32))
        if bsz < self._dump_mode_flags.shape[0]:
            self._dump_mode_flags[bsz:] = VERIFY_MODE
        return None

    @torch.inference_mode()
    def prepare_forward_inputs(
        self,
        input_ids: torch.LongTensor,
        mode_flags: torch.Tensor,
        verify_dec_lens: torch.Tensor,
        *,
        total_tokens: int | None = None,
        num_draft: int | None = None,
        num_verify: int | None = None,
        spill_candidate: bool = True,
        refresh_candidate: bool = True,
        host_plan: dict | None = None,
    ):
        """Build all per-step tensors needed by the unified decode forward pass.

        This is the central step-preparation routine for the unified backends.
        It updates cache lengths, refreshes draft page sets, builds mixed KV
        indices, plans both wrappers, flattens the token inputs, and returns the
        exact buffers consumed by the compiled model.

        The keyword hints carry CPU-known step shape from the scheduler so the
        hot path avoids device synchronizations: ``total_tokens`` skips the
        qo-indptr readback, ``num_draft`` skips the draft-mask reduction, and
        ``spill_candidate`` gates the page-boundary spill check to the steps
        where a draft token can actually open a new page. Callers that do not
        know these values (warm-up, tests) omit them and get the
        synchronizing behavior.
        """
        bsz = mode_flags.shape[0]
        is_draft = mode_flags == DRAFT_MODE
        has_draft = bool(is_draft.any()) if num_draft is None else num_draft > 0
        # Hints for subclass hooks so their per-step mask reductions do not
        # synchronize the device when the caller already knows the answer.
        self._prep_num_verify = num_verify
        self._prep_refresh_candidate = refresh_candidate
        # Host-plan bundle (see aspire.backends.host_plan): CPU-built
        # plan arrays plus spill/refresh row lists. When present, both
        # FlashInfer plans run through plan_host with zero device readbacks
        # and the spill/refresh mask reductions are skipped.
        self._prep_host_plan = host_plan
        cachelens = self.cachelens[:bsz]
        draft_cachelens = self.draft_cachelens[:bsz]
        # When the host plan carries the rendered step metadata, upload it in
        # a handful of async pinned copies instead of recomputing it with
        # ~20 small kernels (page counts, last-page lens, qo indptr, input
        # positions, flatten indices). Values are bit-identical mirrors of
        # the device formulas, gated by ASPIRE_PLAN_PARITY.
        uploads = None
        if (
            host_plan is not None
            and total_tokens is not None
            and host_plan.get("sample_indices") is not None
            and torch.device(self.device).type == "cuda"
        ):
            uploads = self._step_upload(host_plan, total_tokens)

        with _profile_scope("PREP_CACHELEN"):
            cachelens += verify_dec_lens
            draft_cachelens += is_draft.to(draft_cachelens.dtype)

        with _profile_scope("PREP_PAGEMETA"):
            if uploads is not None:
                verify_num_pages = uploads["num_pages"][:bsz]
                last_page_len = uploads["last_page_len"][:bsz]
            else:
                verify_num_pages = self._get_num_pages(cachelens).to(torch.int32)
                # Draft samples must see tokens generated in previous draft
                # steps, so last_page_len is derived from draft_cachelens
                # (not cachelens).
                effective_cachelens = torch.where(is_draft, draft_cachelens, cachelens)
                last_page_len = self._get_last_page_len(effective_cachelens)

        with _profile_scope("PREP_REPLAN"):
            self._prepare_replan_state(int(bsz), verify_num_pages)

        with _profile_scope("PREP_DRAFT_SPILL"):
            # If a draft token is the first entry in a new page (boundary
            # crossing), that page must be added to the draft index set.
            # Each step generates 1 token per sample → at most 1 new page.
            #
            # The KV append for a draft step writes the new token at the tail
            # of the mixed page list, so the draft table's last page must be
            # the target page that owns position ``draft_cachelens - 1``.
            # The spill check compares the actual table tail against that
            # page; backends that replan every step always end the refreshed
            # table there, so it only fires for a stale table.
            draft_num_pages = self._draft_num_pages_per_request[:bsz]
            if host_plan is not None:
                # The host mirror already identified the spill rows (empty for
                # backends that replan every step), so the synchronizing tail
                # comparison is skipped.
                spill_np = host_plan["spill_rows"]
                if spill_np.size:
                    spill_ids = self._upload_row_ids(spill_np, "spill")
                    self._handle_draft_page_spill_ids(spill_ids, bsz)
            elif has_draft and spill_candidate:
                draft_pages_needed = self._get_num_pages(draft_cachelens).to(torch.int32)
                just_crossed = (
                    ((draft_cachelens - 1) % self.page_size == 0)
                    & is_draft
                    & (verify_dec_lens > 0)
                )
                tail_index = (draft_num_pages.to(torch.long) - 1).clamp(min=0)
                tail_page = self._draft_indices[:bsz].gather(
                    1, tail_index.unsqueeze(1)
                ).squeeze(1)
                new_page_index = (draft_pages_needed.to(torch.long) - 1).clamp(min=0)
                new_page = self.target_block_table[:bsz].gather(
                    1, new_page_index.unsqueeze(1)
                ).squeeze(1).to(tail_page.dtype)
                need_spill = just_crossed & (tail_page != new_page)
                if need_spill.any():
                    self._handle_draft_page_spill(need_spill, draft_pages_needed, draft_num_pages)

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
            self.unified_paged_kv_indptr[bsz + 1 :] = (
                self.unified_paged_kv_indptr[bsz]
            )
            self.unified_paged_kv_last_page_len[bsz:] = 0

        with _profile_scope("PREP_PATCH_META"):
            self._prepare_patch_metadata(int(bsz), is_draft)

        with _profile_scope("PREP_QO_INDPTR"):
            if uploads is None:
                self.unified_qo_indptr[0] = 0
                self.unified_qo_indptr[1 : bsz + 1] = verify_dec_lens.cumsum(0)
                if total_tokens is None:
                    total_tokens = int(self.unified_qo_indptr[bsz].item())
            # With uploads, unified_qo_indptr (including the bucket padding
            # rows) was copied from the host plan's pinned array.
            bucket_tokens = self._get_bucket_size(total_tokens) if total_tokens > 0 else self.bucket_sizes[0]
            if uploads is None:
                self.unified_qo_indptr[bsz + 1 :] = bucket_tokens

        with _profile_scope("PREP_PLAN"):
            if host_plan is not None and hasattr(self.unified_wrapper, "plan_host"):
                self.unified_wrapper.plan_host(
                    host_plan["qo_indptr"],
                    host_plan["kv_indptr"],
                    host_plan["kv_len"],
                    self.unified_paged_kv_indices,
                    num_qo_heads=self.model.config.n_head,
                    num_kv_heads=self.model.config.n_local_heads,
                    head_dim_qk=self.model.config.head_dim,
                    page_size=self.page_size,
                    causal=True,
                )
            else:
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

        with _profile_scope("PREP_PLAN_FULL"):
            # Dump wrapper should see full context including draft tokens,
            # so use effective_num_pages (accounts for spill) and last_page_len
            # (from effective_cachelens).
            if uploads is not None:
                effective_num_pages = uploads["effective_num_pages"][:bsz]
            else:
                effective_num_pages = self._get_num_pages(effective_cachelens).to(torch.int32)
            self._plan_step_dump_wrapper(
                bsz=int(bsz),
                verify_num_pages=effective_num_pages,
                last_page_len=last_page_len,
                mode_flags=mode_flags,
            )

        with _profile_scope("PREP_INPUT_POS"):
            input_pos = self._input_pos
            if uploads is None:
                input_pos[:bsz] = torch.where(
                    is_draft,
                    draft_cachelens - 1,
                    cachelens - verify_dec_lens,
                ).to(torch.int32)
                input_pos[bsz:] = 0
            # With uploads, _input_pos was copied from the host plan.

        with _profile_scope("PREP_FLATTEN"):
            if uploads is not None:
                sample_indices = uploads["sample_indices"][:total_tokens]
                position_indices = uploads["position_indices"][:total_tokens]
            else:
                # output_size lets repeat_interleave skip its device-side
                # length readback (an implicit synchronization) — the token
                # total is already known on the host here.
                sample_indices = torch.repeat_interleave(
                    torch.arange(bsz, device=self.device),
                    verify_dec_lens,
                    output_size=total_tokens,
                )
                offsets = torch.repeat_interleave(
                    self.unified_qo_indptr[:bsz],
                    verify_dec_lens,
                    output_size=total_tokens,
                )
                position_indices = torch.arange(total_tokens, device=self.device) - offsets
            flat_tokens = input_ids[sample_indices, position_indices]

        with _profile_scope("PREP_PAD"):
            bucket_view = self._bucket_views[bucket_tokens]
            bucket_view.zero_()
            if total_tokens > 0:
                bucket_view[:total_tokens] = flat_tokens

        if host_plan is not None and self._plan_parity_enabled():
            self._verify_host_plan_parity(
                host_plan,
                int(bsz),
                mode_flags=mode_flags,
                verify_dec_lens=verify_dec_lens,
                uploads=uploads,
                total_tokens=total_tokens,
            )

        self._after_prepare_step(mode_flags, int(bsz))

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
    def clear_slots(self, slot_ids: torch.Tensor):
        """Zero KV/cache bookkeeping for a subset of slots before reusing them.

        Slot reuse is common in continuous batching. This method clears both the
        model KV cache and the backend's speculative page-selection state so a
        new request cannot inherit stale pages from an old slot occupant.
        """
        slot_ids = slot_ids.to(device=self.device, dtype=torch.long)
        pages_per_slot = self.max_num_pages_per_request
        for layer in self.model.layers:
            cache = layer.attention.kv_cache.kv_cache
            cache_view = cache.view(cache.shape[0] // pages_per_slot, pages_per_slot, *cache.shape[1:])
            cache_view[slot_ids] = 0

        self.cachelens[slot_ids] = 0
        self.num_pages_per_request[slot_ids] = 0
        if self.is_spec:
            self.draft_cachelens[slot_ids] = 0
            self.draft_num_pages_per_request[slot_ids] = 0
            self.draft_actual_tokens[slot_ids] = 0
            self._draft_indices[slot_ids] = 0
            self._draft_num_pages_per_request[slot_ids] = 0
            self.mode_flags[slot_ids] = 0
        self._clear_variant_slot_state(slot_ids)

    def _clear_variant_slot_state(self, slot_ids: torch.Tensor):
        """Clear slot-local dump state."""
        if not hasattr(self, "attention_weights_buffer"):
            return
        dump = self.attention_weights_buffer.view(
            self.batch_size,
            self.model.config.n_head,
            self._dump_query_slots(),
            self._kv_stride_alloc,
        )
        dump[slot_ids] = self.attention_weights_buffer_neg_inf
        old_lse_rows = (
            self.unified_full_qo_indptr[slot_ids + 1].long() - 1
        ).clamp(min=0, max=self._lse_buffer.shape[0] - 1)
        self._lse_buffer[old_lse_rows] = 0
        self._dump_mode_flags[slot_ids] = VERIFY_MODE

    def _store_draft_rows(
        self,
        slot_ids: torch.Tensor | None,
        selected: torch.Tensor,
        valid_counts: torch.Tensor,
        *,
        table: torch.Tensor | None = None,
        count_buffer: torch.Tensor | None = None,
        full_prefix: int | None = None,
    ):
        """Write selected page rows into a draft table while zeroing stale tail data.

        Pages arrive already masked and sorted by page ID from the page
        selector.  This method simply writes the valid prefix and zeros
        the tail so stale page IDs never leak.  When ``full_prefix`` is given
        the rows are the leading prefix of the table, so slice writes replace
        the index writes and no ``slot_ids`` tensor is needed.
        """
        if full_prefix is None and slot_ids.numel() == 0:
            return

        if table is None:
            table = self._draft_indices

        valid_counts = valid_counts.to(device=table.device, dtype=torch.int32)
        if full_prefix is None:
            slot_ids = slot_ids.to(device=table.device, dtype=torch.long)
            row_count = slot_ids.shape[0]
        else:
            row_count = int(full_prefix)
        if count_buffer is not None:
            if full_prefix is None:
                count_buffer[slot_ids] = valid_counts
            else:
                count_buffer[:row_count] = valid_counts
        if row_count == 0:
            return

        width = table.shape[1]
        selected = selected.to(device=table.device, dtype=table.dtype)
        n_cols = min(width, selected.shape[1])
        rows = self._staging_rows(table, row_count, width)
        rows.zero_()
        if n_cols > 0:
            col_ids = torch.arange(n_cols, device=table.device, dtype=torch.int32)
            valid_mask = col_ids.unsqueeze(0) < valid_counts.unsqueeze(1)
            rows[:, :n_cols] = torch.where(valid_mask, selected[:, :n_cols], rows[:, :n_cols])
        if full_prefix is None:
            table[slot_ids] = rows
        else:
            table[:row_count] = rows

    def _staging_rows(self, table: torch.Tensor, row_count: int, width: int) -> torch.Tensor:
        """Reusable zero-staging buffer for `_store_draft_rows` (avoids a
        per-step allocation on the decode hot path)."""
        cache = getattr(self, "_staging_rows_cache", None)
        if cache is None:
            cache = {}
            self._staging_rows_cache = cache
        key = (table.dtype, width)
        buf = cache.get(key)
        if buf is None or buf.shape[0] < row_count:
            buf = torch.zeros(
                (max(row_count, self.batch_size), width),
                dtype=table.dtype,
                device=table.device,
            )
            cache[key] = buf
        return buf[:row_count]

    def _store_layer_draft_rows(
        self,
        slot_ids: torch.Tensor,
        selected: torch.Tensor,
        valid_counts: torch.Tensor,
        *,
        table: torch.Tensor,
    ):
        """Vectorized counterpart of `_store_draft_rows()` for per-layer draft tables.

        Pages arrive already masked and sorted from the page selector.
        """
        if slot_ids.numel() == 0:
            return

        slot_ids = slot_ids.to(device=table.device, dtype=torch.long)
        valid_counts = valid_counts.to(device=table.device, dtype=torch.int32)
        selected = selected.to(device=table.device, dtype=table.dtype)

        layer_count, row_count, width = table.shape
        if row_count == 0 or layer_count == 0:
            return

        n_cols = min(width, selected.shape[-1])
        rows = table.new_zeros((layer_count, slot_ids.shape[0], width))
        if n_cols > 0:
            col_ids = torch.arange(n_cols, device=table.device, dtype=torch.int32)
            valid_mask = col_ids.view(1, 1, n_cols) < valid_counts.view(1, -1, 1)
            rows[:, :, :n_cols] = torch.where(valid_mask, selected[:, :, :n_cols], rows[:, :, :n_cols])
        table[:, slot_ids, :] = rows

    def _build_recent_draft_rows(self, slot_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a recent-pages fallback draft table for a set of slots.

        Used by the initial seeding/refill paths, where no verify signal
        exists yet.
        """
        slot_ids = slot_ids.to(device=self.device, dtype=torch.long)
        num_pages = self._get_num_pages(self.cachelens)
        valid_counts = self.page_selector.compute_valid_counts(num_pages[slot_ids])
        topk_offsets = torch.arange(self.max_draft_pages, device=self.device, dtype=torch.int64)
        start_pages = (num_pages[slot_ids].to(torch.int64) - valid_counts.to(torch.int64)).clamp(min=0)
        gather_positions = start_pages[:, None] + topk_offsets[None, :]
        rows = self.target_block_table[slot_ids].gather(1, gather_positions.to(torch.long))
        valid_mask = topk_offsets[None, :] < valid_counts.to(torch.int64)[:, None]
        rows = torch.where(valid_mask, rows, torch.zeros_like(rows))
        return rows, valid_counts

    @torch.inference_mode()
    def _init_draft_indices_for_slots(self, slot_ids: torch.Tensor):
        """Re-seed draft tables for only the slots that were just refilled.

        Continuous batching frequently recycles a subset of slots. This method is
        the partial-slot counterpart to `_init_draft_indices()`, avoiding a full
        batch reset when only some requests changed.
        """
        slot_ids = slot_ids.to(device=self.device, dtype=torch.long)
        self._draft_indices[slot_ids] = 0
        self._draft_num_pages_per_request[slot_ids] = 0
        rows, valid_counts = self._build_recent_draft_rows(slot_ids)
        self._store_draft_rows(
            slot_ids,
            rows,
            valid_counts,
            count_buffer=self._draft_num_pages_per_request,
        )

    @torch.inference_mode()
    def encode_slots(
        self,
        slot_ids: torch.Tensor,
        input_ids: torch.Tensor,
        max_prefill_tokens: int = 8192,
    ) -> torch.Tensor:
        """Chunk-prefill prompts into specific slots and return each prompt's last token.

        Prompt prefill can exceed the runtime token budget, so this method breaks
        each prompt into manageable chunks, appends them into paged KV cache, and
        seeds speculative draft tables once prefill is complete.
        """
        self.clear_slots(slot_ids)

        n_slots = slot_ids.shape[0]
        if isinstance(input_ids, (list, tuple)):
            id_list = [ids.to(self.device) for ids in input_ids]
        else:
            id_list = [input_ids[i] for i in range(n_slots)]
        seq_lens = torch.tensor([ids.shape[0] for ids in id_list], device=self.device, dtype=torch.int64)
        processed = torch.zeros(n_slots, device=self.device, dtype=torch.int64)
        result = None

        while bool((processed < seq_lens).any()):
            step_indices = []
            step_chunks = []
            budget = max_prefill_tokens
            for i in range(n_slots):
                remaining = int((seq_lens[i] - processed[i]).item())
                if remaining <= 0:
                    continue
                chunk = min(remaining, budget)
                if chunk <= 0:
                    break
                start = int(processed[i].item())
                step_indices.append(i)
                step_chunks.append((start, start + chunk))
                budget -= chunk

            if not step_indices:
                break

            n_step = len(step_indices)
            step_indices_t = torch.tensor(step_indices, device=self.device, dtype=torch.long)
            step_slot_ids = slot_ids[step_indices_t]
            chunk_sizes_t = torch.tensor(
                [end - start for start, end in step_chunks],
                dtype=torch.int32,
                device=self.device,
            )
            flat_tokens = torch.cat(
                [id_list[i][start:end] for i, (start, end) in zip(step_indices, step_chunks)]
            ).unsqueeze(0)

            qo_indptr = torch.zeros(n_step + 1, dtype=torch.int32, device=self.device)
            qo_indptr[1:] = chunk_sizes_t.cumsum(0)

            new_cachelens = self.cachelens[step_slot_ids] + chunk_sizes_t
            mini_num_pages = self._get_num_pages(new_cachelens).to(torch.int32)
            mini_last_page_len = self._get_last_page_len(new_cachelens)

            mini_block_table = self.target_block_table[step_slot_ids]
            mini_indptr = torch.zeros(n_step + 1, dtype=torch.int32, device=self.device)
            total_pages = mini_num_pages.sum().item()
            mini_indices = torch.zeros(max(total_pages, 1), dtype=torch.int32, device=self.device)
            build_kv_indices_from_block_table(
                mini_block_table,
                mini_num_pages,
                mini_indices,
                mini_indptr,
            )

            self.prefill_wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=mini_indptr,
                paged_kv_indices=mini_indices,
                paged_kv_last_page_len=mini_last_page_len,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=self.page_size,
                q_data_type=self.dtype,
                causal=True,
            )

            step_result = self.prefill(
                model=self.model,
                x=flat_tokens,
                input_pos=self.cachelens[step_slot_ids],
                kv_append_indptr=qo_indptr,
                kv_page_indices=mini_indices,
                kv_page_indptr=mini_indptr,
                kv_page_lastlen=mini_last_page_len,
            )

            self.cachelens[step_slot_ids] += chunk_sizes_t
            processed[step_indices_t] = torch.tensor(
                [end for _, end in step_chunks],
                dtype=torch.int64,
                device=self.device,
            )

            flat_result = step_result.squeeze(0)
            last_token_results = flat_result[qo_indptr[1:].to(torch.long) - 1]
            if result is None:
                result = torch.zeros(n_slots, dtype=last_token_results.dtype, device=self.device)
            finished_mask = processed[step_indices_t] >= seq_lens[step_indices_t]
            if bool(finished_mask.any()):
                finished_slots = step_indices_t[finished_mask]
                result[finished_slots] = last_token_results[finished_mask]

        if self.is_spec:
            self.draft_cachelens[slot_ids] = self.cachelens[slot_ids].clone()
            self._init_draft_indices_for_slots(slot_ids)

        return result

    @torch.inference_mode()
    def clear_kv(self):
        """Reset all KV cache and speculative state owned by the unified backend.

        Full-cache counterpart to `clear_slots()`, used for benchmark restarts.
        """
        super().clear_kv()
        if self.is_spec and hasattr(self, "_draft_num_pages_per_request"):
            self._draft_indices.zero_()
            self._draft_num_pages_per_request.zero_()
        self._clear_variant_kv_state()

    def _clear_variant_kv_state(self):
        """Hook for subclass state cleared on a full-cache reset."""
        return None

    def compile(self):
        """Finalize compiled execution and ensure the dump wrapper is graph-backed.

        Upgrades the dump wrapper to CUDA-graph mode when setup skipped graph
        buffers, so the model and dump path share the same replay regime.
        """
        import torch._dynamo.config
        import torch._inductor.config

        if (
            self.is_spec
            and hasattr(self, "unified_full_buffer")
            and hasattr(self, "_flashinfer_use_cuda_graph")
            and not self._flashinfer_use_cuda_graph
        ):
            self.unified_full_wrapper = self._create_attention_wrapper(
                self.unified_full_buffer,
                use_cuda_graph=True,
                qo_indptr_buf=self.unified_full_qo_indptr,
                kv_indptr_buf=self.unified_full_paged_kv_indptr,
                kv_indices_buf=self.unified_full_paged_kv_indices,
                kv_last_page_len_buf=self.unified_full_paged_kv_last_page_len,
                jit_args=self._unified_full_jit_args,
            )
            with torch.inference_mode():
                self._plan_full_wrapper_from_buffers()

        super().compile()
