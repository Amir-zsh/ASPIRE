"""Drop-in replacement for BatchPrefillWithPagedKVCacheWrapper using the
holistic BatchAttention module.

The holistic module provides per-request tile routing ({64, 16} after the
tile64 patch), which is optimal for mixed draft/verify batches.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple, Union

import jinja2
import torch

from flashinfer.attention import get_holistic_attention_module
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.utils import (
    dtype_map,
    mask_mode_literal,
    pos_encoding_mode_literal,
    write_if_different,
)
from flashinfer.jit.attention.utils import generate_additional_params
from flashinfer.page import get_seq_lens
from flashinfer.utils import MaskMode, PosEncodingMode, TensorLayout, _unpack_paged_kv_cache


# ---------------------------------------------------------------------------
# Module generator with the ADDITIONAL_PARAMS_SETTER indexed for the
# persistent kernel's params[i] loop.
# ---------------------------------------------------------------------------

def _gen_persistent_batch_attention_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_logits_soft_cap: bool = False,
    use_profiler: bool = False,
):
    """Like ``gen_customize_batch_attention_module`` but fixes the
    ``ADDITIONAL_PARAMS_SETTER`` for BatchAttention's persistent kernel.

    Upstream generates ``params.field = ...`` but the setter expands inside
    ``for (int i = 0; i < 2; i++) { ... }`` over a ``PersistentParams params[2]``
    array, so it must be ``params[i].field = ...``.
    """
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map[dtype_q],
        "dtype_kv": dtype_map[dtype_kv],
        "dtype_o": dtype_map[dtype_o],
        "idtype": dtype_map[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    (additional_params_decl, additional_func_params, additional_params_setter) = (
        generate_additional_params(
            additional_tensor_names,
            additional_tensor_dtypes,
            additional_scalar_names,
            additional_scalar_dtypes,
        )
    )
    # The persistent kernel's setter runs inside a loop over params[i].
    additional_params_setter = additional_params_setter.replace("params.", "params[i].")

    kwargs |= {
        "additional_params_decl": additional_params_decl,
        "additional_func_params": additional_func_params,
        "additional_params_setter": additional_params_setter,
    }

    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_attention_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())
    with open(jit_env.FLASHINFER_CSRC_DIR / "batch_attention_paged_kernel_inst.jinja") as f:
        paged_kernel_inst_templ = jinja2.Template(f.read())

    generated_inc_str = config_templ.render(**kwargs)
    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []
    for mask_mode in [0, 1, 2, 3]:
        dest_path = gen_directory / f"batch_attention_paged_kernel_mask_{mask_mode}.cu"
        source_paths.append(dest_path)
        source = paged_kernel_inst_templ.render(
            mask_mode=mask_mode_literal[mask_mode], **kwargs,
        )
        write_if_different(dest_path, source)

    for filename in ["batch_attention.cu", "batch_attention_jit_binding.cu"]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_attention_config.inc"
    write_if_different(generated_config_path, generated_inc_str)

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=["-DFLASHINFER_ENABLE_PROFILER"] if use_profiler else [],
    )


def _build_and_load_custom_module(*args, **kwargs):
    return _gen_persistent_batch_attention_module(*args, **kwargs).build_and_load()


# ---------------------------------------------------------------------------
# HolisticAttentionWrapper
# ---------------------------------------------------------------------------

class HolisticAttentionWrapper:
    """Drop-in replacement for ``BatchPrefillWithPagedKVCacheWrapper``.

    Internally uses FlashInfer's holistic ``BatchAttention`` module which
    provides per-request tile routing.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        jit_args: Optional[list] = None,
        jit_kwargs: Optional[dict] = None,
        **_ignored,
    ):
        self._kv_layout = kv_layout
        device = float_workspace_buffer.device

        self._float_workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._int_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._pinned_int_workspace = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=torch.device("cpu"), pin_memory=True,
        )

        self._jit_module: Optional[object] = None
        self._has_jit = jit_args is not None
        if jit_args is not None:
            self._jit_module = _build_and_load_custom_module(
                *jit_args, **(jit_kwargs or {}),
            )
        self._cached_module: Optional[object] = None

        self._plan_info = None
        self._kv_indices: Optional[torch.Tensor] = None
        self._page_size: int = 0
        self._sm_scale: Optional[float] = None
        self._mask_mode: int = MaskMode.NON_CAUSAL.value
        self._num_qo_heads: int = 0
        self._num_kv_heads: int = 0

        # plan_host runs without any stream synchronization, so the pinned
        # int workspace the C++ planner fills (and asynchronously copies to
        # the device) must not be reused while a previous copy may still be
        # in flight. A small event-guarded ring makes reuse safe under host
        # run-ahead; slot 0 aliases the buffer plan() uses so both entry
        # points share state.
        self._pinned_ring = [self._pinned_int_workspace]
        self._pinned_ring_events: List[Optional[torch.cuda.Event]] = [None]
        for _ in range(3):
            self._pinned_ring.append(
                torch.empty(
                    self._pinned_int_workspace.shape[0],
                    dtype=torch.uint8,
                    device=torch.device("cpu"),
                    pin_memory=True,
                )
            )
            self._pinned_ring_events.append(None)
        self._pinned_slot = 0

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        causal: bool = False,
        sm_scale: Optional[float] = None,
        **kwargs,
    ) -> None:
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk
        if isinstance(q_data_type, str):
            q_data_type = getattr(torch, q_data_type)

        # Resolve module (lazy for non-JIT path)
        if self._jit_module is not None:
            module = self._jit_module
        elif self._cached_module is not None:
            module = self._cached_module
        else:
            module = get_holistic_attention_module(
                q_data_type, q_data_type, q_data_type,
                paged_kv_indptr.dtype,
                head_dim_qk, head_dim_vo,
                PosEncodingMode["NONE"].value,
                False,  # use_logits_soft_cap
                False,  # use_profiler
            )
        self._cached_module = module

        kv_len_arr = get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, page_size)
        self._kv_indices = paged_kv_indices

        qo_indptr_host = qo_indptr.to("cpu", non_blocking=True)
        kv_indptr_host = paged_kv_indptr.to("cpu", non_blocking=True)
        kv_len_arr_host = kv_len_arr.to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()

        batch_size = kv_len_arr.shape[0]

        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads

        self._plan_info = module.plan(
            self._float_workspace,
            self._int_workspace,
            self._pinned_int_workspace,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )
        # Fence slot 0 of the pinned ring (aliased to this workspace) so a
        # following plan_host cannot overwrite it while the planner's async
        # upload is still in flight.
        if self._pinned_ring_events[0] is None:
            self._pinned_ring_events[0] = torch.cuda.Event()
        self._pinned_ring_events[0].record()

    def plan_host(
        self,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        kv_len_host: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        causal: bool = False,
        sm_scale: Optional[float] = None,
    ) -> None:
        """Plan from host-built arrays without any device readback or sync.

        The C++ planner only ever consumes host copies of ``qo_indptr``,
        ``kv_indptr`` and ``kv_len``; ``plan()`` obtains them by reading the
        device buffers back behind a full stream synchronization every step.
        When the caller already maintains those integers on the CPU (see
        ``aspire.backends.host_plan``), this entry point passes them
        straight through: the only device work left is the planner's own
        asynchronous workspace upload. Page indices stay on the device and are
        consumed at ``run()`` exactly as before.

        Requires a prior ``plan()`` call (setup warm-up) so the kernel module
        is resolved; host arrays must match the shapes/dtype the device path
        would produce, which the parity gate verifies.
        """
        module = self._jit_module if self._jit_module is not None else self._cached_module
        if module is None:
            raise RuntimeError(
                "plan_host requires a prior plan() call to resolve the kernel module"
            )
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

        self._kv_indices = paged_kv_indices
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads

        slot = self._pinned_slot
        event = self._pinned_ring_events[slot]
        if event is not None:
            # Wait for the workspace upload issued the last time this slot was
            # used; with the ring depth this is almost always already done.
            event.synchronize()
        self._plan_info = module.plan(
            self._float_workspace,
            self._int_workspace,
            self._pinned_ring[slot],
            qo_indptr_host,
            kv_indptr_host,
            kv_len_host,
            kv_len_host.shape[0],
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )
        if event is None:
            event = torch.cuda.Event()
            self._pinned_ring_events[slot] = event
        # The planner enqueues its pinned->device copy on the current stream;
        # recording here fences the slot for its next reuse.
        event.record()
        self._pinned_slot = (slot + 1) % len(self._pinned_ring)

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        out: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)

        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.empty(q.shape[0], q.shape[1], device=q.device, dtype=torch.float32)

        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[2])

        run_args = [
            self._float_workspace,
            self._int_workspace,
            self._plan_info,
            q, k_cache, v_cache,
            self._kv_indices,
            out, lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            1.0,       # v_scale
            sm_scale,
            0.0,       # logits_soft_cap
        ]
        if self._has_jit:
            run_args.extend(args)

        self._cached_module.run(*run_args)

        if return_lse:
            return out, lse
        return out


# ---------------------------------------------------------------------------
# Holistic JIT kernel generation — delegates to the unified builder in
# jit_kernels.py with backend="holistic".
# ---------------------------------------------------------------------------

from aspire.backends.jit_kernels import (
    _BASE_TENSOR_NAMES,
    _scalar_spec,
    build_dump_struct,
)


def build_holistic_dump_jit_args(
    dtype: torch.dtype,
    head_dim: int,
    dump_kind: str = "last_token",
    class_name: str = "HolisticDumpAttentionWeights",
    op_name: str = "holistic_batch_dump_attention_weights",
    use_mode_flags: bool = False,
    dump_buffer_dtype: torch.dtype = torch.float32,
) -> list:
    """Build JIT args for the holistic attention weight dump kernel.

    The ``wrapper.run()`` call signature becomes::

        wrapper.run(q, kv_cache, attention_weights_buf,
                    num_heads, jit_kv_stride, return_lse=..., out=...)
    """
    scalar_names, scalar_types = _scalar_spec("holistic")

    attn_type = dtype_map[dump_buffer_dtype]
    code = build_dump_struct("holistic", dump_kind, class_name, use_mode_flags, attn_type=attn_type)

    tensor_names = list(_BASE_TENSOR_NAMES)
    tensor_types = [attn_type]
    if use_mode_flags and dump_kind != "noop":
        tensor_names.append("mode_flags")
        tensor_types.append("int32_t")

    return [
        op_name, dtype, dtype, dtype, torch.int32,
        head_dim, head_dim,
        tensor_names, tensor_types,
        scalar_names, scalar_types,
        class_name, code,
    ]
