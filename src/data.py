"""
Corpus factories for OpenWebText (OWT), Text8, and an optional Indic slice.

Each factory returns a *callable* that produces an iterable of document strings.
This is intentional: a fresh iterator per consumer avoids the "generator already
exhausted" trap that fertility/metric code would otherwise hit.

Datasets are streamed where possible to keep disk footprint low on shared
servers (Aneek's sentinel/megatron setup typically has <500GB per-user quota).

For CANDI-style evaluations we want sequences of approximately 1024 tokens, but
fertility is computed on raw text, not on pre-tokenized sequences. So the data
functions here expose two APIs:
    - `iter_documents(...)` : raw strings, for fertility + judge PPL input
    - `iter_token_sequences(tokenizer, length=1024, ...)` : token id lists
      already chunked, for diffusion sampling conditioning / perplexity
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Callable, Iterable, Iterator

from datasets import load_dataset

from .tokenizers_bench import load_tokenizer


# ---------------------------------------------------------------------------
# OpenWebText
# ---------------------------------------------------------------------------
def owt_factory(
    split: str = "train",
    n_docs: int | None = 10_000,
    seed: int = 0,
    hf_cache: str | None = None,
) -> Callable[[], Iterable[str]]:
    """
    Returns a callable that yields up to `n_docs` OpenWebText documents.

    Each call to the returned function creates a fresh iterator, so the same
    sample can be consumed by multiple metrics.

    We use the `Skylion007/openwebtext` mirror. Streaming is enabled by default;
    if you need a fully deterministic sample set, pass `n_docs` and the same
    `seed` on every call.
    """
    def make_iter() -> Iterator[str]:
        ds = load_dataset(
            "Skylion007/openwebtext",
            split=split,
            streaming=True,
            cache_dir=hf_cache,
            trust_remote_code=True,
        )
        # For a reproducible subset with streaming, we hash-filter.
        rng = random.Random(seed)
        kept = 0
        for ex in ds:
            if n_docs is not None and kept >= n_docs:
                return
            # Light filtering: skip extremely short docs (< 64 chars)
            text = ex.get("text", "")
            if len(text) < 64:
                continue
            # Keep ~25% for faster iteration if the user asked for a cap
            if n_docs is not None and rng.random() > 0.25:
                continue
            kept += 1
            yield text

    return make_iter


# ---------------------------------------------------------------------------
# Text8  (character-level corpus — CANDI uses this as the small-|V| anchor)
# ---------------------------------------------------------------------------
# Text8 is a single ~100MB file of cleaned-up Wikipedia, lowercase a-z and space.
# We slice it into pseudo-documents of ~2048 characters (matches common practice
# in the masked-diffusion literature).
def text8_factory(
    path: str | None = None,
    doc_len_chars: int = 2048,
    n_docs: int | None = 5_000,
) -> Callable[[], Iterable[str]]:
    def make_iter() -> Iterator[str]:
        # Try local file first, else HF
        text: str
        if path and Path(path).exists():
            text = Path(path).read_text(encoding="ascii", errors="ignore")
        else:
            ds = load_dataset("afmck/text8", split="train")
            text = ds[0]["text"]  # text8 is a single long string
        total = len(text)
        step = doc_len_chars
        emitted = 0
        for start in range(0, total - step, step):
            if n_docs is not None and emitted >= n_docs:
                return
            emitted += 1
            yield text[start:start + step]

    return make_iter


# ---------------------------------------------------------------------------
# Indic slice (Sangraha Hindi)
# ---------------------------------------------------------------------------
def sangraha_hindi_factory(
    n_docs: int | None = 5_000,
    hf_cache: str | None = None,
) -> Callable[[], Iterable[str]]:
    """
    Small Hindi slice from the Sangraha corpus (AI4Bharat), used to stress-test
    tokenizer fertility on Devanagari. Matches the Indic thread of our
    larger research agenda.
    """
    def make_iter() -> Iterator[str]:
        try:
            ds = load_dataset(
                "ai4bharat/sangraha",
                "verified.hin",
                split="train",
                streaming=True,
                cache_dir=hf_cache,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[warn] Sangraha Hindi unavailable ({e}); skipping.")
            return
        emitted = 0
        for ex in ds:
            if n_docs is not None and emitted >= n_docs:
                return
            text = ex.get("text", "")
            if len(text) < 64:
                continue
            emitted += 1
            yield text

    return make_iter


# ---------------------------------------------------------------------------
# Token-sequence iterator (for conditioning diffusion models during eval)
# ---------------------------------------------------------------------------
def iter_token_sequences(
    doc_factory: Callable[[], Iterable[str]],
    tokenizer_name: str,
    seq_length: int = 1024,
    max_sequences: int | None = 1000,
    pack: bool = True,
    hf_token: str | None = None,
) -> Iterator[list[int]]:
    """
    Stream fixed-length token id sequences from a raw-document factory.

    pack=True follows the Diffusion-LM / MDLM convention of concatenating
    documents with a separator and slicing into fixed windows. This is necessary
    for comparing methods on equal-length contexts.
    """
    tok = load_tokenizer(tokenizer_name, hf_token=hf_token)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos_id is None:
        eos_id = 0  # last-resort; byte tokenizers may not have EOS

    buffer: list[int] = []
    emitted = 0
    for doc in doc_factory():
        ids = tok.encode(doc, add_special_tokens=False)
        if pack:
            buffer.extend(ids)
            buffer.append(eos_id)
            while len(buffer) >= seq_length:
                chunk = buffer[:seq_length]
                buffer = buffer[seq_length:]
                if max_sequences is not None and emitted >= max_sequences:
                    return
                emitted += 1
                yield chunk
        else:
            # No packing: just take the first seq_length tokens and pad
            if len(ids) < seq_length:
                pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
                ids = ids + [pad_id] * (seq_length - len(ids))
            ids = ids[:seq_length]
            if max_sequences is not None and emitted >= max_sequences:
                return
            emitted += 1
            yield ids
