from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle, islice
from typing import Any, Iterable

import numpy as np

from aspire.io_utils import sha256_json
from aspire.config import DatasetConfig


LONGBENCH_ENGLISH_PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": 'You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:',
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": 'Please complete the code given below. \n{context}Next line of code:\n',
    "repobench-p": 'Please complete the code given below. \n{context}{input}Next line of code:\n',
}

CODEELO_INSTRUCTION = (
    "You are a coding expert. Given a competition-level coding problem, you need "
    "to write a C++ program to solve it. You may start by outlining your thought "
    "process. In the end, please provide the complete code in a code block enclosed "
    "with ``` ```."
)


@dataclass(frozen=True, slots=True)
class TokenizedDataset:
    prompts: list[list[int]]
    metadata: dict[str, Any]


class _CombinedRows(list):
    def __init__(self, rows: Iterable[dict[str, Any]], fingerprint: str):
        super().__init__(rows)
        self._fingerprint = fingerprint


def _replace_leading_bos(tokens: list[int], tokenizer) -> list[int]:
    """Set position 0 to the tokenizer BOS (eos for models without a BOS)."""
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if tokens and bos_id is not None:
        tokens[0] = int(bos_id)
    return tokens


def _chat_tokens(tokenizer, content: str, enable_thinking: bool) -> list[int]:
    """Render the chat template to text, then encode that text."""
    messages = [{"role": "user", "content": content}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    tokens = tokenizer.encode(text)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return _replace_leading_bos([int(token) for token in tokens], tokenizer)


def _plain_tokens(tokenizer, content: str) -> list[int]:
    tokens = tokenizer.encode(content)
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return _replace_leading_bos([int(token) for token in tokens], tokenizer)


def _codeelo_text(row: dict[str, Any]) -> str:
    """Assemble the CodeElo user prompt."""
    parts = []
    title = row.get("title", "")
    if title:
        parts.append(f"# {title}")
    time_limit = row.get("time_limit_ms")
    mem_limit = row.get("memory_limit_mb")
    if time_limit or mem_limit:
        limits = []
        if time_limit:
            limits.append(f"Time limit: {time_limit} ms")
        if mem_limit:
            limits.append(f"Memory limit: {mem_limit} MB")
        parts.append(" | ".join(limits))
    for heading, key in (
        ("Description", "description"),
        ("Input", "input"),
        ("Output", "output"),
        ("Interaction", "interaction"),
    ):
        value = row.get(key, "")
        if value:
            parts.append(f"## {heading}\n{value}")
    examples = row.get("examples") or []
    example_parts = []
    for index, example in enumerate(examples, 1):
        if isinstance(example, (list, tuple)) and len(example) >= 2:
            example_parts.append(
                f"### Example {index}\nInput:\n```\n{example[0]}\n```\n"
                f"Output:\n```\n{example[1]}\n```"
            )
        elif isinstance(example, str):
            example_parts.append(f"### Example {index}\n{example}")
    if example_parts:
        parts.append("## Examples\n" + "\n\n".join(example_parts))
    note = row.get("note", "")
    if note:
        parts.append(f"## Note\n{note}")
    return f"{CODEELO_INSTRUCTION}\n\n" + "\n\n".join(parts)


def _longbench_v2_text(row: dict[str, Any]) -> str:
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{row['context']}\n</text>\n\n"
        f"Question: {row['question']}\n\nAnswer:"
    )


def _dataset_rows(config: DatasetConfig):
    from datasets import load_dataset

    name = config.name.lower()
    if name in {"aime24", "aime25", "aime26"}:
        return load_dataset(f"math-ai/{name}", split=config.split, revision=config.revision)
    if name == "codeelo":
        return load_dataset(
            "Qwen/CodeElo", split=config.split, revision=config.revision
        )
    if name == "longbench":
        subset = config.subset or "all"
        if subset in {"all", "english"}:
            rows = []
            fingerprints = []
            for component in LONGBENCH_ENGLISH_PROMPTS:
                dataset = load_dataset(
                    "zai-org/LongBench",
                    component,
                    split=config.split,
                    revision=config.revision,
                )
                fingerprints.append(
                    {"subset": component, "fingerprint": getattr(dataset, "_fingerprint", None)}
                )
                for row in dataset:
                    tagged = dict(row)
                    tagged["_aspire_subset"] = component
                    rows.append(tagged)
            return _CombinedRows(rows, sha256_json(fingerprints))
        if subset not in LONGBENCH_ENGLISH_PROMPTS:
            raise ValueError(f"unsupported English LongBench subset: {subset}")
        return load_dataset(
            "zai-org/LongBench",
            subset,
            split=config.split,
            revision=config.revision,
        )
    if name == "longbench-v2":
        return load_dataset(
            "THUDM/LongBench-v2", split=config.split, revision=config.revision
        )
    raise ValueError(f"unsupported dataset: {config.name}")


def _row_text(config: DatasetConfig, row: dict[str, Any]) -> str:
    name = config.name.lower()
    if name.startswith("aime"):
        return str(row["problem"]).strip()
    if name == "codeelo":
        return _codeelo_text(row)
    if name == "longbench":
        subset = row["_aspire_subset"] if "_aspire_subset" in row else config.subset
        return LONGBENCH_ENGLISH_PROMPTS[subset].format(
            context=row["context"], input=row.get("input", "")
        )
    if name == "longbench-v2":
        return _longbench_v2_text(row)
    raise ValueError(f"unsupported dataset: {config.name}")


def _in_length_range(config: DatasetConfig, length: int) -> bool:
    return not (
        (config.min_prompt_tokens is not None and length < config.min_prompt_tokens)
        or (config.max_prompt_tokens is not None and length > config.max_prompt_tokens)
    )


def _matches_metadata_filters(config: DatasetConfig, row: dict[str, Any]) -> bool:
    if config.name.lower() != "longbench-v2":
        return True
    return all(
        expected is None or row.get(field) == expected
        for field, expected in (
            ("domain", config.domain),
            ("difficulty", config.difficulty),
            ("length", config.length_category),
        )
    )


def synthetic_dataset(
    config: DatasetConfig,
    *,
    batch_size: int,
    seed: int,
    vocab_size: int = 32000,
) -> TokenizedDataset:
    count = config.limit or batch_size
    lengths = config.prompt_lengths
    if not lengths:
        low = config.min_prompt_tokens or 4096
        high = config.max_prompt_tokens or low
        lengths = tuple(np.linspace(low, high, count, dtype=np.int64).tolist())
    selected_lengths = list(islice(cycle(lengths), count))
    rng = np.random.default_rng(seed)
    prompts = [
        rng.integers(100, vocab_size, size=length, dtype=np.int64).tolist()
        for length in selected_lengths
    ]
    return TokenizedDataset(
        prompts,
        {
            "name": "synthetic",
            "revision": None,
            "seed": seed,
            "token_sha256": sha256_json(prompts),
        },
    )


def load_tokenized_dataset(
    config: DatasetConfig,
    *,
    model: str,
    batch_size: int,
    seed: int,
) -> TokenizedDataset:
    if config.name.lower() == "synthetic":
        return synthetic_dataset(config, batch_size=batch_size, seed=seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    rows = _dataset_rows(config)
    limit = config.limit or batch_size
    prompts: list[list[int]] = []
    for row in rows:
        row = dict(row)
        if not _matches_metadata_filters(config, row):
            continue
        content = _row_text(config, row)
        if config.apply_chat_template:
            tokens = _chat_tokens(
                tokenizer,
                content,
                enable_thinking=config.enable_thinking,
            )
        else:
            tokens = _plain_tokens(tokenizer, content)
        if _in_length_range(config, len(tokens)):
            prompts.append(tokens)
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(
            f"dataset {config.name!r} produced no prompts in the configured token range"
        )
    if len(prompts) < limit:
        prompts = list(islice(cycle(prompts), limit))
    fingerprint = getattr(rows, "_fingerprint", None)
    metadata = {
        "name": config.name,
        "subset": config.subset,
        "revision": config.revision,
        "source_fingerprint": fingerprint,
        "prompt_format": {
            "apply_chat_template": config.apply_chat_template,
            "enable_thinking": config.enable_thinking,
        },
        "filters": {
            "domain": config.domain,
            "difficulty": config.difficulty,
            "length_category": config.length_category,
            "min_prompt_tokens": config.min_prompt_tokens,
            "max_prompt_tokens": config.max_prompt_tokens,
        },
        "token_sha256": sha256_json(prompts),
    }
    return TokenizedDataset(prompts, metadata)
