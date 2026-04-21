"""
Tokenizer-intrinsic benchmark: fertility, compression, OOV, segmentation examples.

Designed to answer "how does the same text look under ByT5 / Llama-2 / GPT-2 / Qwen2.5 / Gemma?"
independently of any diffusion model. Works on any text corpus (OWT, Text8, Sangraha).

Fertility definition: mean number of tokens per word, where "word" is a whitespace-delimited
unit. For scripts without whitespace (CJK) this is misleading — we also report bytes/token
(compression) which is script-agnostic.

Reference for fertility in multilingual settings: Rust et al. 2021 "How Good is Your Tokenizer?"
(ACL), Ahia et al. 2023 "Do All Languages Cost the Same?" (EMNLP).
"""
from __future__ import annotations

import json
import re
import statistics
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tokenizer registry
# ---------------------------------------------------------------------------
# Each entry: (display_name, HF repo or local path, vocab_size_expected, notes).
# `expected_vocab` is used for sanity checks only; the actual size is read from
# the loaded tokenizer at runtime.
#
# Llama-2 is gated on HF. If `meta-llama/Llama-2-7b-hf` fails auth, the mirror
# `NousResearch/Llama-2-7b-hf` is identical in tokenizer and ungated.
#
# LLaDA has its own ~126K-vocab tokenizer — do NOT reuse the Llama-2 32K
# registry entry for LLaDA generations, or ids above 32K (including the
# mask id 126336) will fail to decode.
TOKENIZER_REGISTRY: dict[str, dict] = {
    "byt5": {
        "hf_id": "google/byt5-small",
        "expected_vocab": 384,      # ByT5 reserves some special IDs beyond 256
        "family": "byte",
        "notes": "Byte-level; fertility is bytes/word, not subwords/word.",
    },
    "llama2": {
        "hf_id": "NousResearch/Llama-2-7b-hf",
        "expected_vocab": 32_000,
        "family": "bpe-spm",
        "notes": "SentencePiece BPE. Latin-heavy; Indic coverage is weak.",
    },
    "llada": {
        "hf_id": "GSAI-ML/LLaDA-8B-Base",
        "expected_vocab": 126_464,
        "family": "bpe-spm",
        "notes": (
            "LLaDA's native tokenizer (LLaMA-derived, extended). "
            "mask_token_id=126336. Requires trust_remote_code=True."
        ),
    },
    "gpt2": {
        "hf_id": "gpt2",
        "expected_vocab": 50_257,
        "family": "bpe-byte",
        "notes": "Byte-level BPE; CANDI/MDLM/SEDD reference tokenizer.",
    },
    "qwen25": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "expected_vocab": 151_936,
        "family": "bpe-byte",
        "notes": "Multilingual, strong CJK coverage.",
    },
    "gemma3": {
        "hf_id": "google/gemma-3-4b-pt",
        "expected_vocab": 262_144,
        "family": "bpe-spm",
        "notes": "Gated. Broad multilingual incl. Indic.",
    },
}


@dataclass
class FertilityStats:
    """Per-corpus, per-tokenizer statistics."""
    tokenizer_name: str
    corpus_name: str
    vocab_size: int
    n_documents: int
    n_words: int               # whitespace-delimited tokens in raw text
    n_chars: int
    n_bytes: int
    n_tokens: int              # tokenizer output length
    n_unk: int                 # count of UNK ids in tokenizer output

    # Derived
    tokens_per_word_mean: float = 0.0
    tokens_per_word_median: float = 0.0
    tokens_per_word_p95: float = 0.0
    tokens_per_word_p99: float = 0.0
    bytes_per_token: float = 0.0        # compression ratio, higher = more compressed
    chars_per_token: float = 0.0
    unk_rate: float = 0.0

    # Segmentation examples for qualitative inspection
    examples: list[dict] = field(default_factory=list)

    # True if the corpus had no words (empty / failed fetch). Other fields are
    # zeros in that case and should not be interpreted as measurements.
    is_empty: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Word splitter
# ---------------------------------------------------------------------------
# Whitespace + treat punctuation as separate words, so "hello," -> ["hello", ","].
# This is deliberately simple and language-agnostic. For Devanagari it works
# because Hindi uses ASCII spaces between words. For CJK it would break and we
# would need a language-specific segmenter (e.g. jieba); we report bytes/token
# for those cases instead.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Tokenizer loading with graceful fallback
# ---------------------------------------------------------------------------
def load_tokenizer(name: str, hf_token: str | None = None) -> PreTrainedTokenizerBase:
    """Load a tokenizer by registry name. Raises if the name is unknown."""
    if name not in TOKENIZER_REGISTRY:
        raise KeyError(
            f"Unknown tokenizer '{name}'. "
            f"Known: {sorted(TOKENIZER_REGISTRY)}"
        )
    entry = TOKENIZER_REGISTRY[name]
    hf_id = entry["hf_id"]
    needs_remote_code = name in {"llada"}
    # use_fast=True is important — Rust tokenizers are 10-100x faster here
    tok = AutoTokenizer.from_pretrained(
        hf_id,
        use_fast=True,
        token=hf_token,
        trust_remote_code=needs_remote_code,
    )
    # Sanity check
    expected = entry["expected_vocab"]
    if abs(tok.vocab_size - expected) > 2000:
        print(
            f"[warn] {name}: tokenizer.vocab_size={tok.vocab_size} deviates from "
            f"expected {expected}. Continuing, but verify."
        )
    return tok


# ---------------------------------------------------------------------------
# Fertility computation
# ---------------------------------------------------------------------------
def compute_fertility(
    tokenizer: PreTrainedTokenizerBase,
    documents: Iterable[str],
    tokenizer_name: str,
    corpus_name: str,
    n_examples_to_keep: int = 8,
) -> FertilityStats:
    """
    Compute fertility statistics for one (tokenizer, corpus) pair.

    We iterate once through the corpus, tokenizing document-by-document and
    accumulating statistics. We collect a small number of (word, token_ids)
    examples for qualitative inspection of segmentation quality.

    On an empty corpus (common when an optional data source like Sangraha is
    unreachable on the server), returns a zeroed FertilityStats with is_empty=True
    and emits a warning rather than raising — the reason being that raising
    here would kill the whole tokenizer × corpus grid in run_fertility_suite.
    """
    per_word_counts: list[int] = []
    n_tokens = 0
    n_words = 0
    n_chars = 0
    n_bytes = 0
    n_unk = 0
    n_documents = 0
    unk_id = tokenizer.unk_token_id

    examples: list[dict] = []

    # Words kept for segmentation examples — chosen to stress the tokenizer
    # with multi-morpheme English, CJK-adjacent Latin, and Devanagari.
    interesting_words = {
        "antidisestablishmentarianism",
        "hyperparameter",
        "TensorFlow",
        "सूर्यमंदिर",          # sun temple (Hindi, long compound)
        "नमस्कार",             # greeting
        "tokenization",
        "München",
        "北京",                # Beijing (CJK)
        "🙂",                  # emoji
    }
    interesting_seen: set[str] = set()

    for doc in documents:
        if not doc:
            continue
        n_documents += 1
        n_chars += len(doc)
        n_bytes += len(doc.encode("utf-8", errors="replace"))
        words = split_words(doc)
        n_words += len(words)

        # Tokenize the whole doc once for top-line counts
        ids = tokenizer.encode(doc, add_special_tokens=False)
        n_tokens += len(ids)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)

        # For per-word fertility we need word-level tokenization.
        # Batch the words in chunks of 4K to amortize tokenizer overhead.
        CHUNK = 4096
        for start in range(0, len(words), CHUNK):
            chunk = words[start:start + CHUNK]
            # encode_batch is faster but not uniformly available; use __call__
            enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            per_word_counts.extend(len(x) for x in enc)

        # Grab a few segmentation examples
        if len(examples) < n_examples_to_keep:
            for w in words[:200]:
                if w in interesting_words and w not in interesting_seen:
                    ids_w = tokenizer.encode(w, add_special_tokens=False)
                    pieces = tokenizer.convert_ids_to_tokens(ids_w)
                    examples.append({
                        "word": w,
                        "n_tokens": len(ids_w),
                        "pieces": pieces,
                    })
                    interesting_seen.add(w)
                    if len(examples) >= n_examples_to_keep:
                        break

    if not per_word_counts:
        warnings.warn(
            f"Corpus '{corpus_name}' for tokenizer '{tokenizer_name}' is empty "
            f"(n_documents={n_documents}, n_words={n_words}). "
            f"Returning zeroed stats with is_empty=True.",
            stacklevel=2,
        )
        return FertilityStats(
            tokenizer_name=tokenizer_name,
            corpus_name=corpus_name,
            vocab_size=tokenizer.vocab_size,
            n_documents=n_documents,
            n_words=n_words,
            n_chars=n_chars,
            n_bytes=n_bytes,
            n_tokens=n_tokens,
            n_unk=n_unk,
            examples=examples,
            is_empty=True,
        )

    stats = FertilityStats(
        tokenizer_name=tokenizer_name,
        corpus_name=corpus_name,
        vocab_size=tokenizer.vocab_size,
        n_documents=n_documents,
        n_words=n_words,
        n_chars=n_chars,
        n_bytes=n_bytes,
        n_tokens=n_tokens,
        n_unk=n_unk,
        tokens_per_word_mean=statistics.mean(per_word_counts),
        tokens_per_word_median=statistics.median(per_word_counts),
        tokens_per_word_p95=_percentile(per_word_counts, 0.95),
        tokens_per_word_p99=_percentile(per_word_counts, 0.99),
        bytes_per_token=n_bytes / max(n_tokens, 1),
        chars_per_token=n_chars / max(n_tokens, 1),
        unk_rate=n_unk / max(n_tokens, 1),
        examples=examples,
        is_empty=False,
    )
    return stats


def _percentile(xs: list[int], q: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    k = max(0, min(len(sorted_xs) - 1, int(q * len(sorted_xs))))
    return float(sorted_xs[k])


# ---------------------------------------------------------------------------
# Cross-tokenizer reporting
# ---------------------------------------------------------------------------
def run_fertility_suite(
    corpora: dict[str, Iterable[str]],
    tokenizer_names: list[str],
    out_dir: Path,
    hf_token: str | None = None,
) -> dict[str, dict[str, FertilityStats]]:
    """
    Run the full (tokenizer × corpus) grid and dump JSON + human-readable report.

    `corpora` is a dict mapping corpus_name -> iterable of document strings.
    Iterables are consumed multiple times, so pass lists or re-createable
    generators (see data.py for factories).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, FertilityStats]] = {}
    for tok_name in tokenizer_names:
        print(f"\n=== Loading tokenizer {tok_name} ===")
        try:
            tok = load_tokenizer(tok_name, hf_token=hf_token)
        except Exception as e:
            print(f"[error] could not load {tok_name}: {e}")
            continue
        results[tok_name] = {}
        for corpus_name, docs in corpora.items():
            print(f"  Corpus {corpus_name} ...", flush=True)
            # If docs is a list it can be reused; otherwise callers must re-create.
            stats = compute_fertility(tok, docs, tok_name, corpus_name)
            results[tok_name][corpus_name] = stats
            if stats.is_empty:
                print(f"    [skip] corpus '{corpus_name}' was empty")
            else:
                print(
                    f"    vocab={stats.vocab_size}  "
                    f"tok/word={stats.tokens_per_word_mean:.3f}  "
                    f"bytes/tok={stats.bytes_per_token:.3f}  "
                    f"unk%={100*stats.unk_rate:.4f}"
                )

    # Save
    dump = {
        t: {c: s.to_dict() for c, s in v.items()} for t, v in results.items()
    }
    with open(out_dir / "fertility.json", "w") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)

    _write_markdown_report(results, out_dir / "fertility_report.md")
    return results


def _write_markdown_report(
    results: dict[str, dict[str, FertilityStats]],
    out_path: Path,
) -> None:
    """Write a compact table comparing all tokenizers across all corpora."""
    lines: list[str] = ["# Tokenizer Fertility Report\n"]
    corpora = sorted({c for v in results.values() for c in v})
    for corpus in corpora:
        lines.append(f"\n## Corpus: `{corpus}`\n")
        lines.append(
            "| Tokenizer | \\|V\\| | tok/word (mean) | tok/word (p95) | "
            "bytes/tok | unk% |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for tok_name, by_corpus in results.items():
            if corpus not in by_corpus:
                continue
            s = by_corpus[corpus]
            if s.is_empty:
                lines.append(f"| {tok_name} | {s.vocab_size} | — | — | — | — |  (empty)")
                continue
            lines.append(
                f"| {tok_name} | {s.vocab_size} | {s.tokens_per_word_mean:.3f} | "
                f"{s.tokens_per_word_p95:.1f} | {s.bytes_per_token:.3f} | "
                f"{100*s.unk_rate:.4f} |"
            )
    # Segmentation examples section
    lines.append("\n## Segmentation Examples\n")
    lines.append("These show *how* each tokenizer splits morphologically non-trivial words.\n")
    for tok_name, by_corpus in results.items():
        examples_union: dict[str, list[str]] = {}
        for s in by_corpus.values():
            for ex in s.examples:
                examples_union[ex["word"]] = ex["pieces"]
        if not examples_union:
            continue
        lines.append(f"\n### `{tok_name}`\n")
        for word, pieces in examples_union.items():
            lines.append(f"- `{word}` → `{pieces}`")
    out_path.write_text("\n".join(lines))