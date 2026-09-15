from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from aspire.config import BenchmarkConfig, EngineConfig, GenerationConfig, Method, SchedulerConfig
from aspire.engine.api import GenerationResult
from aspire.engine.buffers import graph_buckets
from aspire.engine.cyclic import CyclicController
from aspire.engine.scheduler import SchedulerBridge
from aspire.policies import AspirePolicy, CalibrationKey, FeedbackPolicy, FixedDraftPolicy, LatencyModel


def generation_limits(
    prompt_lengths: Sequence[int], max_new_tokens: int, max_seq_len: int
) -> np.ndarray:
    lengths = np.asarray(prompt_lengths, dtype=np.int64)
    if lengths.ndim != 1 or lengths.size < 1 or np.any(lengths < 1):
        raise ValueError("prompt lengths must be a nonempty positive sequence")
    if np.any(lengths >= max_seq_len):
        raise ValueError("each prompt must leave room for one generated token")
    return np.minimum(max_new_tokens, max_seq_len - lengths).astype(np.int32)


@dataclass(slots=True)
class StepMetrics:
    forward_ms: float
    model_tokens: int
    graph_bucket: int
    draft_requests: int
    verify_requests: int
    committed_tokens: int
    accepted_tokens: int
    verify_kv_tokens: int
    policy: dict[str, float | int] = field(default_factory=dict)


@dataclass(slots=True)
class RunMetrics:
    steps: list[StepMetrics] = field(default_factory=list)
    prefill_seconds: float = 0.0
    wall_seconds: float = 0.0
    scheduler_decisions: int = 0
    scheduler_syncs: int = 0
    scheduler_uploads: int = 0
    scheduler_broadcasts: int = 0
    scheduler_transferred_bytes: int = 0
    scheduler_seconds: float = 0.0
    graph_replays: int = 0
    graph_fallbacks: int = 0
    refill_seconds: float = 0.0
    refills: int = 0
    completed_requests: int = 0
    warmup_requests: int = 0
    stop_reason: str = "batch_complete"

    def to_dict(self) -> dict:
        return {
            "prefill_seconds": self.prefill_seconds,
            "wall_seconds": self.wall_seconds,
            "scheduler_decisions": self.scheduler_decisions,
            "scheduler_syncs": self.scheduler_syncs,
            "scheduler_uploads": self.scheduler_uploads,
            "scheduler_broadcasts": self.scheduler_broadcasts,
            "scheduler_transferred_bytes": self.scheduler_transferred_bytes,
            "scheduler_seconds": self.scheduler_seconds,
            "graph_replays": self.graph_replays,
            "graph_fallbacks": self.graph_fallbacks,
            "refill_seconds": self.refill_seconds,
            "refills": self.refills,
            "completed_requests": self.completed_requests,
            "warmup_requests": self.warmup_requests,
            "stop_reason": self.stop_reason,
            "steps": [asdict(step) for step in self.steps],
        }


class CudaBackend:
    """Static-batch CUDA implementation used by the public engine facade."""

    def __init__(self, config: EngineConfig, method: Method, scheduler: SchedulerConfig):
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("ASPIRE requires a CUDA device")
        from aspire.attention.flashinfer import validate_flashinfer_patch

        self.flashinfer_build = validate_flashinfer_patch()
        self.config = config
        self.method = Method(method)
        self.scheduler_config = scheduler
        import os

        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        if config.tp_size > 1:
            # Intra-node one-shot all-reduce for the small per-layer TP collectives.
            os.environ.setdefault("ENABLE_INTRA_NODE_COMM", "1")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", device_id=self.device)
            if dist.get_world_size() != config.tp_size:
                raise RuntimeError(
                    f"torchrun world size {dist.get_world_size()} does not match tp_size={config.tp_size}"
                )
        elif dist.is_initialized() and dist.get_world_size() != 1:
            raise RuntimeError("tp_size=1 cannot run inside a larger distributed world")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
        self._low_level = None
        self._generation_signature = None
        self._graphs_ready = False
        self.last_metrics: RunMetrics | None = None

    def _max_draft(self) -> int:
        if self.method is Method.AUTOREGRESSIVE:
            return 0
        if self.method is Method.ASPIRE_FIXED:
            return self.scheduler_config.fixed_draft
        return self.scheduler_config.max_draft

    def _backend_class(self):
        if self.method is Method.AUTOREGRESSIVE:
            from aspire.backends.autoregressive import LMBackendAutoregressive

            return LMBackendAutoregressive
        from aspire.backends.aspire import LMBackendAspire

        return LMBackendAspire

    def _initialize(self, generation: GenerationConfig) -> None:
        import torch
        import torch.distributed as dist

        signature = (generation.sampling, generation.temperature, generation.top_k, generation.top_p)
        if self._low_level is not None:
            if signature != self._generation_signature:
                raise RuntimeError("sampling configuration is fixed when the CUDA backend is initialized")
            return
        checkpoint = Path(self.config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        max_draft = self._max_draft()
        backend = self._backend_class()(
            dtype=self.dtype,
            device=str(self.device),
            dec_len=max_draft + 1,
            draft_dec_len=1,
        )
        group = dist.group.WORLD if dist.is_initialized() else None
        rank_group = list(range(self.world_size))
        if checkpoint.is_dir():
            from aspire.models import load_hf_transformer

            backend.model = load_hf_transformer(
                checkpoint,
                self.config.model,
                device=self.device,
                dtype=self.dtype,
                use_tp=self.world_size > 1,
                rank_group=rank_group,
                group=group,
            )
        elif checkpoint.suffix == ".pth":
            backend.load_model(
                checkpoint,
                self.config.model,
                use_tp=self.world_size > 1,
                rank_group=rank_group,
                group=group,
            )
        else:
            raise ValueError("checkpoint must be a Hugging Face directory or converted .pth file")
        backend.setup_caches(
            max_batch_size=self.config.max_batch_size,
            max_seq_length=self.config.max_seq_len,
            draft_budget=0,
            window_size=0,
            page_size=self.config.page_size,
            top_k_pages=self.config.top_k_pages,
            L=self.config.recent_pages,
            use_cuda_graph=self.config.cuda_graphs,
            deterministic_attention=self.config.deterministic_attention,
            weight_dump_layer=self.config.refresh_layer,
            top_k=generation.top_k if generation.sampling else 0,
            temperature=generation.temperature,
            top_p=generation.top_p,
            holistic_attention=self.config.holistic_attention,
            draft_percent=self.config.draft_percent,
        )
        if self.config.compile:
            backend.compile()
        self._low_level = backend
        self._generation_signature = signature

    def _warm_graphs(self, prompts: Sequence[Sequence[int]]) -> None:
        import torch

        if self._graphs_ready or not self.config.compile:
            return
        tensors = [torch.as_tensor(prompt, dtype=torch.long, device=self.device) for prompt in prompts]
        self._low_level.encode(
            tensors, max_prefill_tokens=self.config.max_prefill_tokens
        )
        requested = list(self.config.graph_buckets) if self.config.graph_buckets else None
        self._low_level.warmup_buckets(
            verbose=self.rank == 0,
            bucket_sizes=requested,
            capture_cuda_graphs=self.config.cuda_graphs,
        )
        if self.config.cuda_graphs:
            self._low_level.seal_graph_buffers()
            schema = self._low_level.graph_buffer_schema()
            if self.world_size > 1:
                import torch.distributed as dist

                schemas = [None] * self.world_size
                dist.all_gather_object(schemas, schema)
                if any(candidate != schema for candidate in schemas):
                    raise RuntimeError("CUDA graph buffer schema differs across TP ranks")
        self._low_level.clear_kv()
        self._graphs_ready = True

    def graph_buffer_pointers(self) -> dict[str, int]:
        if self._low_level is None or not self._graphs_ready or not self.config.cuda_graphs:
            raise RuntimeError("CUDA graph buffers are not captured")
        return self._low_level.graph_buffer_pointers()

    def calibration_key(self, max_draft: int) -> CalibrationKey:
        import torch

        buckets = self.config.graph_buckets or graph_buckets(self.config.max_batch_size * (max_draft + 1))
        return CalibrationKey(
            model=self.config.model,
            gpu_name=torch.cuda.get_device_name(self.device),
            tp_size=self.world_size,
            dtype=self.config.dtype,
            max_batch_size=self.config.max_batch_size,
            page_size=self.config.page_size,
            top_k_pages=self.config.top_k_pages,
            graph_buckets=tuple(buckets),
        )

    def _policy(self, batch_size: int):
        cfg = self.scheduler_config
        if self.method is Method.AUTOREGRESSIVE:
            return FixedDraftPolicy(batch_size, 0)
        if self.method is Method.ASPIRE_FIXED:
            return FixedDraftPolicy(batch_size, cfg.fixed_draft)
        if self.method is Method.ASPIRE_FSM:
            return FeedbackPolicy(batch_size, cfg.min_draft, cfg.max_draft, cfg.initial_draft)
        if cfg.calibration is None:
            raise ValueError("ASPIRE requires scheduler.calibration")
        latency = LatencyModel.load(cfg.calibration)
        latency.validate_key(self.calibration_key(cfg.max_draft))
        return AspirePolicy(
            batch_size,
            cfg.min_draft,
            cfg.max_draft,
            cfg.initial_draft,
            latency,
            omega=cfg.omega,
            alpha_init=cfg.alpha_init,
        )

    @staticmethod
    def _is_eos(tokens, eos_token_ids: tuple[int, ...]):
        import torch

        result = torch.zeros_like(tokens, dtype=torch.bool)
        for token_id in eos_token_ids:
            result |= tokens == token_id
        return result

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        generation: GenerationConfig,
    ) -> list[GenerationResult]:
        import torch

        with torch.inference_mode():
            return self._generate(prompts, generation)

    def generate_cyclic(
        self,
        prompts: Sequence[Sequence[int]],
        generation: GenerationConfig,
        benchmark: BenchmarkConfig,
    ) -> list[GenerationResult]:
        import torch

        if benchmark.mode != "cyclic":
            raise ValueError("generate_cyclic requires benchmark mode=cyclic")
        with torch.inference_mode():
            return self._generate(prompts, generation, benchmark=benchmark)

    def _generate(
        self,
        prompts: Sequence[Sequence[int]],
        generation: GenerationConfig,
        benchmark: BenchmarkConfig | None = None,
    ) -> list[GenerationResult]:
        import torch
        import torch.distributed as dist
        from flashinfer.sampling import sampling_from_probs

        from aspire.engine.sampling import first_eos, verify_topk_speculation

        self._initialize(generation)
        if len(prompts) > self.config.max_batch_size:
            raise ValueError("prompt count exceeds configured max_batch_size")
        if any(len(prompt) == 0 for prompt in prompts):
            raise ValueError("prompts must contain at least one token")
        cyclic = benchmark is not None
        controller = None
        current_prompt_indices = list(range(len(prompts)))
        if benchmark is not None:
            max_evaluation = benchmark.max_evaluation_samples or (
                self.config.max_batch_size * 100
            )
            controller = CyclicController(
                pool_size=len(prompts),
                warmup_samples=benchmark.warmup_samples,
                max_evaluation_samples=max_evaluation,
                converge=benchmark.converge,
                converge_threshold=benchmark.converge_threshold,
                converge_window=benchmark.converge_window,
                converge_min_samples=benchmark.converge_min_samples,
            )
            current_prompt_indices = controller.next_indices(len(prompts))
        active_prompts = [prompts[index] for index in current_prompt_indices]
        prompt_lengths = [len(prompt) for prompt in active_prompts]
        generation_limits_cpu = generation_limits(
            prompt_lengths, generation.max_new_tokens, self.config.max_seq_len
        )

        tensors = [
            torch.as_tensor(prompt, dtype=torch.long, device=self.device)
            for prompt in active_prompts
        ]
        self._warm_graphs(active_prompts)
        if self.config.cuda_graphs:
            self._low_level.assert_graph_buffers_stable()
        graph_replays_start = int(getattr(self._low_level, "_graph_replay_count", 0))
        graph_fallbacks_start = int(getattr(self._low_level, "_graph_fallback_count", 0))
        torch.manual_seed(generation.seed)
        torch.cuda.manual_seed_all(generation.seed)
        self._low_level.clear_kv()
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        prefill_start = time.perf_counter()
        first_tokens = self._low_level.encode(
            tensors, max_prefill_tokens=self.config.max_prefill_tokens
        )
        torch.cuda.synchronize(self.device)
        prefill_seconds = time.perf_counter() - prefill_start

        batch_size = len(active_prompts)
        generation_limits_device = torch.as_tensor(
            generation_limits_cpu, dtype=torch.int32, device=self.device
        )
        max_draft = self._max_draft()
        width = max_draft + 1
        tokens = torch.zeros((batch_size, width + 1), dtype=torch.long, device=self.device)
        tokens[:, 0] = first_tokens
        draft_counts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        generated = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        seq_lens = torch.as_tensor([len(prompt) for prompt in prompts], dtype=torch.int64, device=self.device)
        finished = (generated >= generation_limits_device) | self._is_eos(
            first_tokens, generation.eos_token_ids
        )
        eos_finished = self._is_eos(first_tokens, generation.eos_token_ids)
        output_history = torch.full(
            (batch_size, generation.max_new_tokens), -1, dtype=torch.long, device=self.device
        )
        output_history[:, 0] = first_tokens
        step_accepted = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        step_committed = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        step_hit_eos = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if generation.sampling:
            draft_probs = torch.zeros(
                (batch_size, max(max_draft, 1), generation.top_k),
                dtype=torch.float32,
                device=self.device,
            )
            draft_ids = torch.zeros_like(draft_probs, dtype=torch.int64)
        else:
            draft_probs = draft_ids = None

        # The planner is replicated: every rank runs the identical policy on
        # identical observations, so no per-step decision traffic is needed.
        policy = self._policy(batch_size)
        known_finished = None
        if not generation.eos_token_ids:
            known_finished = generation_limits_cpu <= 1
        scheduler = SchedulerBridge(
            policy,
            prompt_lengths,
            generation_limits_cpu,
            self.device,
            initial_finished=known_finished,
            page_size=self.config.page_size,
        )
        if known_finished is None:
            scheduler.initialize_finished(finished)
        metrics = RunMetrics(prefill_seconds=prefill_seconds)
        completed_results: list[GenerationResult] = []
        if controller is not None and controller.warmup_done:
            wall_start = time.perf_counter()
        step = 0
        request_budget = batch_size
        if controller is not None:
            request_budget += controller.warmup_samples + controller.max_evaluation_samples
        max_steps = request_budget * generation.max_new_tokens * (max(max_draft, 1) + 2)
        import os

        # Steps of CPU run-ahead before forward-timing events are drained.
        drain_depth = max(1, int(os.environ.get("ASPIRE_TIMING_DRAIN", "64")))
        pending_timing: list[tuple] = []
        previous_num_verify = 0
        event_pool: list = []
        # Each step's [accepted, committed, eos] readback is issued on the copy
        # stream and consumed at the top of the next iteration, overlapped with
        # the in-flight forward; its StepMetrics totals are backpatched then.
        observation_pending = False
        pending_step_index = -1

        # Build the FlashInfer plan arrays on the CPU from the scheduler's row
        # state so planning needs no device readback (see backends/host_plan.py).
        # ASPIRE_HOST_PLAN=0 plans from device state instead.
        host_plan_state = None
        if os.environ.get("ASPIRE_HOST_PLAN", "1") == "1":
            make_state = getattr(self._low_level, "make_host_plan_state", None)
            wrapper = getattr(self._low_level, "unified_wrapper", None)
            if make_state is not None and hasattr(wrapper, "plan_host"):
                host_plan_state = make_state()
                if host_plan_state is not None:
                    host_plan_state.reset_all(prompt_lengths)

        def drain_timing() -> None:
            """Backpatch deferred forward timings; one sync for all pending steps."""
            if not pending_timing:
                return
            pending_timing[-1][1].synchronize()
            for start_event, end_event, index in pending_timing:
                metrics.steps[index].forward_ms = float(
                    start_event.elapsed_time(end_event)
                )
                event_pool.append(start_event)
                event_pool.append(end_event)
            pending_timing.clear()

        def apply_step_result(result_totals: dict, index: int) -> None:
            """Backpatch a deferred observation's totals into its step entry."""
            entry = metrics.steps[index]
            entry.committed_tokens = result_totals["committed_tokens"]
            entry.accepted_tokens = result_totals["accepted_tokens"]

        def handle_cyclic_completion() -> bool:
            """Retire finished requests and refill their slots (cyclic mode).

            Runs once per completion event with a single bulk device->host
            transfer for the retired rows. Returns True when the controller
            asks the loop to stop.
            """
            nonlocal metrics, wall_start, graph_replays_start, graph_fallbacks_start
            drain_timing()
            finished_slots = torch.where(finished)[0]
            slot_list = [int(slot) for slot in finished_slots.cpu().tolist()]
            was_warmup = not controller.warmup_done
            if not was_warmup and slot_list:
                generated_cpu = generated[finished_slots].cpu().tolist()
                eos_cpu = eos_finished[finished_slots].cpu().tolist()
                history_cpu = output_history[finished_slots].cpu()
                for position, slot in enumerate(slot_list):
                    count = int(generated_cpu[position])
                    prompt = prompts[current_prompt_indices[slot]]
                    completed_results.append(
                        GenerationResult(
                            token_ids=history_cpu[position, :count].tolist(),
                            finish_reason=(
                                "eos" if bool(eos_cpu[position]) else "length"
                            ),
                            prompt_tokens=len(prompt),
                            generated_tokens=count,
                        )
                    )

            forward_seconds = sum(item.forward_ms for item in metrics.steps) / 1000.0
            committed_tokens = sum(item.committed_tokens for item in metrics.steps)
            throughput = None
            if not was_warmup and forward_seconds > 0:
                throughput = committed_tokens / forward_seconds
            update = controller.complete(
                len(slot_list),
                cumulative_forward_throughput=throughput,
            )
            if not was_warmup:
                metrics.completed_requests = update.evaluated
            if update.warmup_crossed:
                metrics = RunMetrics(warmup_requests=controller.total_completed)
                scheduler.reset_stats()
                graph_replays_start = int(
                    getattr(self._low_level, "_graph_replay_count", 0)
                )
                graph_fallbacks_start = int(
                    getattr(self._low_level, "_graph_fallback_count", 0)
                )
                wall_start = time.perf_counter()
            if update.stop:
                metrics.stop_reason = update.stop_reason or "cyclic_complete"
                return True

            refill_indices = controller.next_indices(len(slot_list))
            refill_prompts = [prompts[index] for index in refill_indices]
            refill_tensors = [
                torch.as_tensor(prompt, dtype=torch.long, device=self.device)
                for prompt in refill_prompts
            ]
            refill_lengths = [len(prompt) for prompt in refill_prompts]
            refill_limits = generation_limits(
                refill_lengths,
                generation.max_new_tokens,
                self.config.max_seq_len,
            )
            refill_start = time.perf_counter()
            refill_first = self._low_level.encode_slots(
                finished_slots,
                refill_tensors,
                max_prefill_tokens=self.config.max_prefill_tokens,
            )
            torch.cuda.synchronize(self.device)
            metrics.refill_seconds += time.perf_counter() - refill_start
            metrics.refills += 1

            tokens[finished_slots].zero_()
            tokens[finished_slots, 0] = refill_first
            draft_counts[finished_slots] = 0
            generated[finished_slots] = 1
            seq_lens[finished_slots] = torch.as_tensor(
                refill_lengths, dtype=torch.int64, device=self.device
            )
            generation_limits_device[finished_slots] = torch.as_tensor(
                refill_limits, dtype=torch.int32, device=self.device
            )
            refill_finished = (
                generation_limits_device[finished_slots] <= 1
            ) | self._is_eos(refill_first, generation.eos_token_ids)
            refill_eos = self._is_eos(refill_first, generation.eos_token_ids)
            finished[finished_slots] = refill_finished
            eos_finished[finished_slots] = refill_eos
            output_history[finished_slots] = -1
            output_history[finished_slots, 0] = refill_first
            step_accepted[finished_slots] = 0
            step_committed[finished_slots] = 0
            step_hit_eos[finished_slots] = False
            if generation.sampling:
                draft_probs[finished_slots].zero_()
                draft_ids[finished_slots].zero_()
            scheduler.reset_slots(
                slot_list,
                refill_lengths,
                refill_limits.tolist(),
                refill_finished.cpu().tolist(),
            )
            if host_plan_state is not None:
                host_plan_state.reset_slots(slot_list, refill_lengths)
            for slot, prompt_index in zip(slot_list, refill_indices, strict=True):
                current_prompt_indices[slot] = prompt_index
            return False

        while step < max_steps:
            if observation_pending:
                # Verify outcome of the previous iteration.
                result_totals = scheduler.consume_observation()
                observation_pending = False
                apply_step_result(result_totals, pending_step_index)
                if controller is not None and result_totals["any_finished"]:
                    if handle_cyclic_completion():
                        break
            verify, policy_metrics = scheduler.decide(finished)
            # Step shape is known on the host, so no device synchronization here.
            shape = scheduler.step_shape
            num_draft = shape["num_draft"]
            num_verify = shape["num_verify"]
            if num_draft + num_verify == 0:
                break
            step_width = shape["step_width"]
            scheduled_verify_kv_tokens = shape["verify_kv_tokens"] if num_verify else 0

            host_plan = None
            if host_plan_state is not None:
                rows = scheduler.step_rows
                bucket_tokens = (
                    self._low_level._get_bucket_size(shape["total_tokens"])
                    if shape["total_tokens"] > 0
                    else self._low_level.bucket_sizes[0]
                )
                host_plan = host_plan_state.build_step(
                    verify=rows["verify"],
                    finished=rows["finished"],
                    draft_counts=rows["draft_counts"],
                    seq_lens=rows["seq_lens"],
                    total_tokens=shape["total_tokens"],
                    bucket_tokens=bucket_tokens,
                )

            step_accepted.zero_()
            step_committed.zero_()
            step_hit_eos.zero_()
            active = ~finished
            draft = active & ~verify

            pre_verify_drafts = draft_counts.clone()
            self._low_level.cachelens[:batch_size] -= torch.where(
                verify, draft_counts, 0
            )
            # Per-row decode lengths and modes, uploaded by the planner.
            dec_lens = scheduler.device_step_ints[0]
            modes = scheduler.device_step_ints[1]
            positions = torch.arange(step_width, device=self.device).unsqueeze(0)
            valid = positions < dec_lens.unsqueeze(1)
            input_tokens = torch.where(
                (verify.unsqueeze(1) & valid),
                tokens[:, :step_width],
                torch.zeros((), dtype=torch.long, device=self.device),
            )
            draft_first = tokens.gather(1, draft_counts.long().unsqueeze(1))
            input_tokens[:, :1] = torch.where(
                draft.unsqueeze(1), draft_first, input_tokens[:, :1]
            )

            prepared = self._low_level.prepare_forward_inputs(
                input_tokens,
                modes,
                dec_lens,
                total_tokens=shape["total_tokens"],
                num_draft=num_draft,
                num_verify=num_verify,
                spill_candidate=shape["spill_candidate"],
                refresh_candidate=previous_num_verify > 0,
                host_plan=host_plan,
            )
            previous_num_verify = num_verify
            bucket_view, input_pos, qo_indptr, kv_indices, kv_indptr, kv_lastlen, total_tokens, sample_indices, position_indices = prepared
            if event_pool:
                event_start = event_pool.pop()
                event_end = event_pool.pop()
            else:
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            if getattr(self._low_level, "_cuda_graphs", None) or getattr(self._low_level, "_draft_cuda_graphs", None):
                result = self._low_level.run_forward(
                    self._low_level.model,
                    bucket_view,
                    input_pos,
                    qo_indptr,
                    kv_indices,
                    kv_indptr,
                    kv_lastlen,
                )
            else:
                result = self._low_level.unified_forward(
                    self._low_level.model,
                    bucket_view,
                    input_pos,
                    qo_indptr,
                    kv_indices,
                    kv_indptr,
                    kv_lastlen,
                )
            event_end.record()
            pending_timing.append((event_start, event_end, len(metrics.steps)))

            if generation.sampling:
                values_flat, ids_flat = result
                values_flat = values_flat[:total_tokens]
                ids_flat = ids_flat[:total_tokens]
                values = torch.zeros(
                    (batch_size, step_width, generation.top_k), dtype=torch.float32, device=self.device
                )
                ids = torch.zeros_like(values, dtype=torch.int64)
                values[sample_indices, position_indices] = values_flat
                ids[sample_indices, position_indices] = ids_flat
            else:
                flat = result[:total_tokens]
                output = torch.zeros((batch_size, step_width), dtype=torch.int64, device=self.device)
                output[sample_indices, position_indices] = flat

            self._low_level.draft_cachelens[:batch_size] -= (
                finished & (modes == 0)
            ).to(torch.int32)
            if num_draft:
                if generation.sampling:
                    # Draft row indices come from the planner (ascending, equal
                    # to torch.where(draft)[0]) so no synchronization is needed.
                    draft_slot_ids = scheduler.device_draft_rows(num_draft)
                    draft_positions = draft_counts[draft_slot_ids]
                    probabilities = values[draft_slot_ids, 0]
                    candidates = ids[draft_slot_ids, 0]
                    local = sampling_from_probs(probabilities, deterministic=True)
                    proposal = candidates[
                        torch.arange(draft_slot_ids.numel(), device=self.device), local
                    ]
                    tokens[draft_slot_ids, draft_positions + 1] = proposal
                    draft_probs[draft_slot_ids, draft_positions] = probabilities
                    draft_ids[draft_slot_ids, draft_positions] = candidates
                    draft_counts[draft_slot_ids] += 1
                else:
                    write_pos = (draft_counts + 1).long().unsqueeze(1)
                    current = tokens.gather(1, write_pos)
                    tokens.scatter_(
                        1,
                        write_pos,
                        torch.where(
                            draft.unsqueeze(1), output[:, :1], current
                        ),
                    )
                    draft_counts += draft.to(draft_counts.dtype)

            if num_verify:
                max_verify_draft = shape["max_verify_draft"]
                if generation.sampling:
                    # Verify row indices come from the planner (ascending, equal
                    # to torch.where(verify)[0]).
                    verify_ids_idx = scheduler.device_verify_rows(num_verify)
                    verify_drafts = pre_verify_drafts[verify_ids_idx]
                    if max_verify_draft:
                        verify_output_rows, accepted_rows = verify_topk_speculation(
                            draft_probs[verify_ids_idx, :max_verify_draft],
                            draft_ids[verify_ids_idx, :max_verify_draft],
                            tokens[verify_ids_idx, 1 : max_verify_draft + 1].to(torch.int32),
                            verify_drafts,
                            values[verify_ids_idx, : max_verify_draft + 1],
                            ids[verify_ids_idx, : max_verify_draft + 1],
                        )
                    else:
                        probabilities = values[verify_ids_idx, 0]
                        candidates = ids[verify_ids_idx, 0]
                        local = sampling_from_probs(probabilities, deterministic=True)
                        bonus_rows = candidates[
                            torch.arange(verify_ids_idx.numel(), device=self.device), local
                        ]
                        verify_output_rows = bonus_rows.unsqueeze(1)
                        accepted_rows = torch.zeros_like(verify_drafts)
                    accepted = torch.zeros_like(draft_counts)
                    accepted[verify_ids_idx] = accepted_rows
                    verify_output = torch.zeros(
                        (batch_size, max_verify_draft + 1),
                        dtype=torch.long,
                        device=self.device,
                    )
                    verify_output[verify_ids_idx] = verify_output_rows
                    draft_probs[verify_ids_idx].zero_()
                    draft_ids[verify_ids_idx].zero_()
                else:
                    if max_verify_draft:
                        positions = torch.arange(max_verify_draft, device=self.device).unsqueeze(0)
                        valid = positions < pre_verify_drafts.unsqueeze(1)
                        proposals = tokens[:, 1 : max_verify_draft + 1]
                        target = output[:, :max_verify_draft]
                        matches = valid & (proposals == target) & ~self._is_eos(proposals, generation.eos_token_ids)
                        accepted = matches.cumprod(dim=1).sum(dim=1).to(torch.int32)
                    else:
                        accepted = torch.zeros_like(draft_counts)
                    accepted = torch.where(verify, accepted, torch.zeros_like(accepted))
                    bonus = output.gather(
                        1, accepted.long().clamp(max=output.shape[1] - 1).unsqueeze(1)
                    ).squeeze(1)
                    positions = torch.arange(max_verify_draft + 1, device=self.device).unsqueeze(0)
                    verify_output = torch.where(
                        positions < accepted.unsqueeze(1),
                        tokens[:, 1 : max_verify_draft + 2],
                        bonus.unsqueeze(1),
                    )

                committed = accepted + 1
                committed, hit_eos = first_eos(verify_output, committed, generation.eos_token_ids)
                remaining = generation_limits_device - generated
                committed = torch.minimum(committed, remaining)
                committed = torch.where(verify, committed, torch.zeros_like(committed))
                hit_eos = hit_eos & verify
                # Masked history write without data-dependent indexing: invalid
                # entries are redirected to the row's first-commit column with
                # the value that column will hold, so the scatter is deterministic.
                positions = torch.arange(verify_output.shape[1], device=self.device).unsqueeze(0)
                valid_commit = positions < committed.unsqueeze(1)
                history_width = output_history.shape[1]
                redirect_col = generated.long().clamp(max=history_width - 1).unsqueeze(1)
                redirect_existing = output_history.gather(1, redirect_col)
                redirect_value = torch.where(
                    committed.unsqueeze(1) > 0, verify_output[:, :1], redirect_existing
                )
                history_pos = torch.where(
                    valid_commit,
                    generated.long().unsqueeze(1) + positions,
                    redirect_col,
                )
                flat_history = output_history.view(-1)
                flat_index = (
                    torch.arange(batch_size, device=self.device).unsqueeze(1) * history_width
                    + history_pos
                )
                flat_history.scatter_(
                    0,
                    flat_index.reshape(-1),
                    torch.where(
                        valid_commit,
                        verify_output,
                        redirect_value,
                    ).reshape(-1),
                )
                generated += committed
                seq_lens += committed
                self._low_level.cachelens[:batch_size] += torch.where(
                    verify, committed - pre_verify_drafts - 1, torch.zeros_like(committed)
                )
                self._low_level.draft_cachelens[:batch_size] += torch.where(
                    verify, committed - pre_verify_drafts, torch.zeros_like(committed)
                )
                last_position = (committed - 1).clamp(min=0).long()
                last_token = verify_output.gather(1, last_position.unsqueeze(1)).squeeze(1)
                done = verify & (
                    hit_eos | (generated >= generation_limits_device)
                )
                finished |= done
                eos_finished |= hit_eos
                update_first = verify & ~finished
                tokens[:, 0] = torch.where(update_first, last_token, tokens[:, 0])
                draft_counts = torch.where(
                    verify, torch.zeros_like(draft_counts), draft_counts
                )
                step_accepted.copy_(accepted)
                step_committed.copy_(committed)
                step_hit_eos.copy_(hit_eos)
                self._low_level._spec_offset = 0

            # Issue this step's observation; it is consumed next iteration.
            scheduler.issue_observation(step_accepted, step_committed, step_hit_eos)
            metrics.steps.append(
                StepMetrics(
                    forward_ms=0.0,
                    model_tokens=int(total_tokens),
                    graph_bucket=int(bucket_view.shape[0]),
                    draft_requests=num_draft,
                    verify_requests=num_verify,
                    committed_tokens=0,
                    accepted_tokens=0,
                    verify_kv_tokens=scheduled_verify_kv_tokens if num_verify else 0,
                    policy=policy_metrics,
                )
            )
            observation_pending = True
            pending_step_index = len(metrics.steps) - 1
            step += 1
            if len(pending_timing) >= drain_depth:
                drain_timing()

        if observation_pending:
            result_totals = scheduler.consume_observation()
            observation_pending = False
            apply_step_result(result_totals, pending_step_index)
        if step >= max_steps and not bool(finished.all().item()):
            raise RuntimeError("generation exceeded the scheduler progress bound")
        torch.cuda.synchronize(self.device)
        metrics.wall_seconds = time.perf_counter() - wall_start
        drain_timing()
        metrics.scheduler_decisions = scheduler.stats.decisions
        metrics.scheduler_syncs = scheduler.stats.device_to_host_syncs
        metrics.scheduler_uploads = scheduler.stats.decision_uploads
        metrics.scheduler_broadcasts = scheduler.stats.broadcasts
        metrics.scheduler_transferred_bytes = scheduler.stats.transferred_bytes
        metrics.scheduler_seconds = scheduler.stats.decision_seconds
        metrics.graph_replays = (
            int(getattr(self._low_level, "_graph_replay_count", 0))
            - graph_replays_start
        )
        metrics.graph_fallbacks = (
            int(getattr(self._low_level, "_graph_fallback_count", 0))
            - graph_fallbacks_start
        )
        if self.config.cuda_graphs:
            self._low_level.assert_graph_buffers_stable()
        self.last_metrics = metrics
        if cyclic:
            return completed_results
        history_cpu = output_history.cpu()
        generated_cpu = generated.cpu().tolist()
        eos_cpu = eos_finished.cpu().tolist()
        results = []
        for index, prompt in enumerate(prompts):
            count = int(generated_cpu[index])
            results.append(
                GenerationResult(
                    token_ids=history_cpu[index, :count].tolist(),
                    finish_reason="eos" if eos_cpu[index] else "length",
                    prompt_tokens=len(prompt),
                    generated_tokens=count,
                )
            )
        return results

    def close(self) -> None:
        import torch

        if self._low_level is not None:
            self._low_level.clear_kv()
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
