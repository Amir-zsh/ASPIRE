"""Run one ASPIRE benchmark and write a plain-JSON result.

Single GPU:
    python benchmark.py --model qwen3-8b --dataset longbench --method aspire-fixed \
        --checkpoint checkpoints/Qwen/Qwen3-8B/model.pth

Tensor parallel:
    torchrun --standalone --nproc_per_node=2 benchmark.py --tp-size 2 ...

Run with --help for the full list of flags. The result file contains the
resolved config, per-step metrics, generated token counts, and a sha256 over
the output tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aspire import AspireEngine
from aspire.config import (
    BenchmarkConfig,
    DatasetConfig,
    EngineConfig,
    GenerationConfig,
    Method,
    RunConfig,
    SchedulerConfig,
)
from aspire.data import load_tokenized_dataset
from aspire.io_utils import sha256_json, write_json_atomic


# Per-dataset settings.
DATASETS = {
    "aime25": dict(name="aime25", split="test", revision="563bb8404243c5f09de6ec262f2db674fe5bce9b",
                   max_seq_len=16384, max_new_tokens=16384, chat=True, thinking=True),
    "codeelo": dict(name="codeelo", split="test", revision="45a2cbba03bd9d20fbca040c2f7da91b2594c444",
                    max_seq_len=16384, max_new_tokens=16384, chat=True, thinking=True),
    "longbench": dict(name="longbench", subset="all", split="test",
                      revision="5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
                      max_seq_len=20048, max_new_tokens=2048, chat=False, thinking=False,
                      min_prompt_tokens=16000, max_prompt_tokens=18000),
    "longbench_v2_30k": dict(name="longbench-v2", split="train",
                             revision="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
                             max_seq_len=42048, max_new_tokens=2048, chat=False, thinking=False,
                             min_prompt_tokens=30000, max_prompt_tokens=40000),
    "longbench_v2_80k": dict(name="longbench-v2", split="train",
                             revision="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
                             max_seq_len=102048, max_new_tokens=2048, chat=False, thinking=False,
                             min_prompt_tokens=80000, max_prompt_tokens=100000),
    "synthetic": dict(name="synthetic", split="test", max_seq_len=20048, max_new_tokens=2048,
                      chat=False, thinking=False),
}

# Models by short name: Hugging Face id and the batch size per dataset
# (one NVIDIA H100 NVL 96GB, TP=1). --model also accepts any Hugging Face id.
MODELS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B",
                   {"aime25": 42, "codeelo": 42, "longbench": 35, "longbench_v2_30k": 18, "longbench_v2_80k": 7}),
    "qwen3-8b": ("Qwen/Qwen3-8B",
                 {"aime25": 30, "codeelo": 29, "longbench": 24, "longbench_v2_30k": 12, "longbench_v2_80k": 5}),
    "ds-llama-8b": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    {"aime25": 34, "codeelo": 33, "longbench": 28, "longbench_v2_30k": 14, "longbench_v2_80k": 5}),
}


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset flags from the dataset and model tables; explicitly passed
    flags always win. A short model name also sets the batch size for the
    dataset and runs cyclic replay until the throughput converges."""
    d = DATASETS[args.dataset]
    model = args.model or "qwen3-8b"
    fill = {"mode": "cyclic", "converge": False, "warmup_samples": 10, "temperature": 0.7,
            "top_k": 50, "top_p": 0.9, "seed": 123, "model": model,
            "max_batch_size": 24, "converge_window": 10}
    if model in MODELS:
        model_id, batch_sizes = MODELS[model]
        fill["model"] = model_id
        if args.dataset in batch_sizes:
            batch = batch_sizes[args.dataset]
            fill.update(max_batch_size=batch, converge=True, converge_window=1 if batch >= 10 else 2)
    for flag, key in (("max_seq_len", "max_seq_len"), ("max_new_tokens", "max_new_tokens"),
                      ("apply_chat_template", "chat"), ("enable_thinking", "thinking"),
                      ("min_prompt_tokens", "min_prompt_tokens"), ("max_prompt_tokens", "max_prompt_tokens"),
                      ("subset", "subset"), ("split", "split"), ("dataset_revision", "revision")):
        fill[flag] = d.get(key)
    explicit_converge = args.converge is not None
    args.model = None  # resolved below from the short name or the given id
    for flag, value in fill.items():
        if getattr(args, flag) is None:
            setattr(args, flag, value)
    if args.mode == "static" and not explicit_converge:
        args.converge = False  # convergence only applies to cyclic replay
    return args


def eos_token_ids_for(tokenizer) -> tuple[int, ...]:
    """End-of-text ids: the tokenizer's eos plus its unk, or the last token of
    "<|eot_id|>" when no unk is defined."""
    second = tokenizer.unk_token_id
    if second is None:
        second = tokenizer.encode("<|eot_id|>")[-1]
    ids = [tokenizer.eos_token_id, second]
    return tuple(dict.fromkeys(int(i) for i in ids if i is not None))


def tokenizer_source(checkpoint: str | Path) -> str:
    """Return the tokenizer directory for an HF or gpt-fast checkpoint."""
    path = Path(checkpoint)
    return str(path.parent if path.suffix == ".pth" else path)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    boolean = argparse.BooleanOptionalAction

    group = parser.add_argument_group("model / engine")
    group.add_argument("--model", default=None, help="short name (" + ", ".join(MODELS) + ") or Hugging Face model id; default qwen3-8b")
    group.add_argument(
        "--checkpoint",
        required=True,
        help="converted gpt-fast model.pth (a local Hugging Face directory is also accepted)",
    )
    group.add_argument("--tp-size", type=int, default=1)
    group.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    group.add_argument("--max-batch-size", type=int, default=None,
                       help="default: per model and dataset for short model names, else 24")
    group.add_argument("--max-seq-len", type=int, default=None, help="default: per dataset")
    group.add_argument("--max-prefill-tokens", type=int, default=8192)
    group.add_argument("--page-size", type=int, default=16, help="KV page size in tokens")
    group.add_argument("--top-k-pages", type=int, default=32, help="minimum sparse-context pages")
    group.add_argument("--recent-pages", type=int, default=8, help="recent pages always kept")
    group.add_argument("--draft-percent", type=float, default=7.0,
                       help="sparse context as %% of full context; negative = fixed top-k pages")
    group.add_argument("--refresh-layer", type=int, default=-2,
                       help="layer doing full attention during drafting")
    group.add_argument("--deterministic-attention", action="store_true")
    group.add_argument("--holistic-attention", action=boolean, default=True)
    group.add_argument("--compile", action=boolean, default=True)
    group.add_argument("--cuda-graphs", action=boolean, default=True)

    group = parser.add_argument_group("scheduler")
    group.add_argument("--min-draft", type=int, default=1)
    group.add_argument("--max-draft", type=int, default=16)
    group.add_argument("--initial-draft", type=int, default=3)
    group.add_argument("--fixed-draft", type=int, default=5,
                       help="draft length for the fixed-gamma methods")
    group.add_argument("--omega", type=float, default=0.8, help="acceptance-rate EMA decay")
    group.add_argument("--alpha-init", type=float, default=0.9, help="acceptance-rate prior")
    group.add_argument("--calibration", default=None,
                       help="timing-model JSON from a --fit-calibration run (required for --method aspire)")

    group = parser.add_argument_group("generation")
    group.add_argument("--max-new-tokens", type=int, default=None, help="default: per dataset")
    group.add_argument("--seed", type=int, default=None, help="default 123")
    group.add_argument("--temperature", type=float, default=None, help="0 = greedy; default 0.7")
    group.add_argument("--top-k", type=int, default=None, help="default 50")
    group.add_argument("--top-p", type=float, default=None, help="default 0.9")
    group.add_argument("--eos-token-ids", type=int, nargs="*", default=[],
                       help="default: derived from the tokenizer (eos plus unk / <|eot_id|>)")

    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset", default="synthetic", choices=sorted(DATASETS),
                       help="task; longbench = 16k-18k prompts, longbench_v2_{30k,80k} = 30-40k / 80-100k")
    group.add_argument("--subset", default=None, help="default: per dataset (longbench: all)")
    group.add_argument("--split", default=None, help="default: per dataset")
    group.add_argument("--dataset-revision", default=None, help="default: the pinned revision per dataset")
    group.add_argument("--limit", type=int, default=None, help="max prompts")
    group.add_argument("--min-prompt-tokens", type=int, default=None, help="default: per dataset")
    group.add_argument("--max-prompt-tokens", type=int, default=None, help="default: per dataset")
    group.add_argument("--prompt-lengths", type=int, nargs="+", default=[],
                       help="synthetic dataset prompt lengths")
    group.add_argument("--apply-chat-template", action=boolean, default=None,
                       help="default: on for aime25/codeelo, off otherwise")
    group.add_argument("--enable-thinking", action=boolean, default=None,
                       help="default: on for aime25/codeelo, off otherwise")
    group.add_argument("--domain", default=None, help="LongBench-v2 domain filter")
    group.add_argument("--difficulty", default=None, choices=["easy", "hard"])
    group.add_argument("--length-category", default=None, choices=["short", "medium", "long"])

    group = parser.add_argument_group("benchmark")
    group.add_argument("--warmup-samples", type=int, default=None, help="default 10")
    group.add_argument("--repetitions", type=int, default=1)
    group.add_argument("--output-dir", default="results/run")
    group.add_argument("--mode", default=None, choices=["static", "cyclic"], help="default cyclic")
    group.add_argument("--converge", action=boolean, default=None,
                       help="run until forward throughput stabilizes (cyclic mode); default off, on for short model names")
    group.add_argument("--converge-threshold", type=float, default=0.01)
    group.add_argument("--converge-window", type=int, default=None,
                       help="stable completions required; default 10 (1 for short model names, 2 when B<10)")
    group.add_argument("--converge-min-samples", type=int, default=20)
    group.add_argument("--max-evaluation-samples", type=int, default=None)

    parser.add_argument("--method", default="aspire",
                        choices=[str(method) for method in Method])
    parser.add_argument("--fit-calibration", default=None, metavar="PATH",
                        help="after the run, fit the scheduler's forward-time model from "
                             "this run's step timings and save it to PATH (run with "
                             "--method aspire-fsm on the task you will benchmark)")
    parser.add_argument("--fit-skip-steps", type=int, default=0,
                        help="drop the first N post-warmup steps from the fit")


def config_from_args(args: argparse.Namespace) -> RunConfig:
    draft_percent = args.draft_percent if args.draft_percent >= 0 else None
    return RunConfig(
        engine=EngineConfig(
            model=args.model,
            checkpoint=Path(args.checkpoint),
            tp_size=args.tp_size,
            dtype=args.dtype,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            max_prefill_tokens=args.max_prefill_tokens,
            page_size=args.page_size,
            top_k_pages=args.top_k_pages,
            recent_pages=args.recent_pages,
            draft_percent=draft_percent,
            refresh_layer=args.refresh_layer,
            deterministic_attention=args.deterministic_attention,
            holistic_attention=args.holistic_attention,
            compile=args.compile,
            cuda_graphs=args.cuda_graphs,
        ),
        scheduler=SchedulerConfig(
            min_draft=args.min_draft,
            max_draft=args.max_draft,
            initial_draft=args.initial_draft,
            fixed_draft=args.fixed_draft,
            omega=args.omega,
            alpha_init=args.alpha_init,
            calibration=Path(args.calibration) if args.calibration else None,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_ids=tuple(args.eos_token_ids),
        ),
        dataset=DatasetConfig(
            name=DATASETS[args.dataset]["name"],
            subset=args.subset,
            revision=args.dataset_revision,
            split=args.split,
            limit=args.limit,
            min_prompt_tokens=args.min_prompt_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            prompt_lengths=tuple(args.prompt_lengths),
            apply_chat_template=args.apply_chat_template,
            enable_thinking=args.enable_thinking,
            domain=args.domain,
            difficulty=args.difficulty,
            length_category=args.length_category,
        ),
        benchmark=BenchmarkConfig(
            warmup_samples=args.warmup_samples,
            repetitions=args.repetitions,
            output_dir=Path(args.output_dir),
            mode=args.mode,
            converge=args.converge,
            converge_threshold=args.converge_threshold,
            converge_window=args.converge_window,
            converge_min_samples=args.converge_min_samples,
            max_evaluation_samples=args.max_evaluation_samples,
        ),
        method=Method(args.method),
    )


def is_rank_zero() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except ImportError:
        return True


def load_prompts(config: RunConfig):
    dataset = load_tokenized_dataset(
        config.dataset,
        model=tokenizer_source(config.engine.checkpoint),
        batch_size=config.engine.max_batch_size,
        seed=config.generation.seed,
    )
    if len(dataset.prompts) > config.engine.max_batch_size:
        raise ValueError(
            f"dataset returned {len(dataset.prompts)} prompts but max_batch_size is "
            f"{config.engine.max_batch_size}"
        )
    return dataset


def run_warmup(engine, prompts, generation, sample_count: int) -> None:
    if sample_count <= 0:
        return
    if not prompts:
        raise ValueError("warm-up requires at least one prompt")
    cursor = 0
    remaining = sample_count
    while remaining:
        count = min(remaining, len(prompts))
        batch = [prompts[(cursor + index) % len(prompts)] for index in range(count)]
        engine.generate(batch, generation)
        cursor = (cursor + count) % len(prompts)
        remaining -= count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(parser)
    args = resolve_defaults(parser.parse_args())
    if not args.eos_token_ids:
        from transformers import AutoTokenizer

        args.eos_token_ids = list(
            eos_token_ids_for(AutoTokenizer.from_pretrained(tokenizer_source(args.checkpoint)))
        )
    config = config_from_args(args)

    dataset = load_prompts(config)
    output_dir = config.benchmark.output_dir

    from aspire.policies.timing import TimingSample

    samples: list[TimingSample] = []
    with AspireEngine(config.engine, config.method, config.scheduler) as engine:
        if config.benchmark.mode == "static":
            run_warmup(
                engine, dataset.prompts, config.generation, config.benchmark.warmup_samples
            )
        for repetition in range(config.benchmark.repetitions):
            if config.benchmark.mode == "cyclic":
                results = engine.generate_cyclic(
                    dataset.prompts, config.generation, config.benchmark
                )
            else:
                results = engine.generate(dataset.prompts, config.generation)
            if not is_rank_zero():
                continue
            outputs = [result.token_ids for result in results]
            payload = {
                "config": config.to_dict(),
                "dataset": {
                    **(dataset.metadata or {}),
                    "prompt_count": len(dataset.prompts),
                    "prompt_tokens": [len(p) for p in dataset.prompts],
                },
                "metrics": engine.metrics.to_dict(),
                "outputs": {
                    "generated_tokens": [r.generated_tokens for r in results],
                    "finish_reasons": [r.finish_reason for r in results],
                    "token_sha256": sha256_json(outputs),
                },
                "repetition": repetition,
            }
            path = output_dir / f"{config.method}-rep{repetition:02d}.json"
            write_json_atomic(path, payload)
            print(
                json.dumps(
                    {
                        "method": str(config.method),
                        "output": str(path),
                        "repetition": repetition,
                        "token_sha256": payload["outputs"]["token_sha256"],
                    },
                    sort_keys=True,
                )
            )
            samples.extend(
                TimingSample(
                    forward_ms=step.forward_ms,
                    active_requests=step.draft_requests + step.verify_requests,
                    extra_decode_tokens=step.model_tokens
                    - (step.draft_requests + step.verify_requests),
                    draft_requests=step.draft_requests,
                    verify_kv_tokens=step.verify_kv_tokens,
                )
                for step in engine.metrics.steps[args.fit_skip_steps :]
                if step.draft_requests + step.verify_requests
            )
        if args.fit_calibration is not None and is_rank_zero():
            from aspire.policies.timing import fit_latency_model

            model = fit_latency_model(
                samples,
                engine.calibration_key(config.scheduler.max_draft),
                draft_cache_tokens=config.engine.top_k_pages * config.engine.page_size,
            )
            model.save(args.fit_calibration)
            print(
                json.dumps(
                    {
                        "calibration": args.fit_calibration,
                        "samples": model.samples,
                        "r2": model.r2,
                        "rmse_ms": model.rmse_ms,
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
