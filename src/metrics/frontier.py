"""
Primary evaluation metrics (Section 6.1).

All metric functions take a DiffusionSampler and return a dict that we can
dump to JSON and concatenate across the (method, NFE, temperature) grid.

Three metrics implemented:
    - frontier.entropy_perplexity(...)   : OWT-scale, sweeps τ, returns full curve
    - frontier.text8_word_frontier(...)  : Text8-scale, %unique vs %valid
    - frontier.judge_perplexity(...)     : fixed judge LM for cross-tokenizer comparability

Headline vs. diagnostic:
    - `judge_perplexity` is the correct headline for cross-method / cross-tokenizer
      comparisons — it applies a single fixed LM to all generations after
      decode+re-encode.
    - `gen_perplexity` (generative perplexity under the model's own tokenizer)
      is DIAGNOSTIC ONLY. See _internal_perplexity's docstring for the caveat —
      in short, for MDLM-style absorbing diffusion evaluated at t=0 with no
      mask, the training loss was zero at every position, so the logits are
      not well-calibrated for this quantity.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from ..samplers import DiffusionSampler


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FrontierPoint:
    method: str
    nfe: int
    temperature: float
    # OWT-style metrics
    entropy: Optional[float] = None
    gen_perplexity: Optional[float] = None     # under the diffusion model's own tokenizer
    judge_perplexity: Optional[float] = None   # under the fixed judge LM
    # Text8-style metrics
    pct_unique_words: Optional[float] = None
    pct_valid_words: Optional[float] = None
    # Meta
    n_samples: int = 0
    seed: int = 0


@dataclass
class FrontierCurve:
    method: str
    nfe: int
    tokenizer_name: str
    corpus: str
    points: list[FrontierPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        # asdict recursively handles nested FrontierPoint dataclasses, so this
        # stays in sync automatically when fields are added.
        return asdict(self)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _sample_entropy(sequences: Sequence[Sequence[int]]) -> float:
    """
    Sample entropy (in nats) over the concatenated generation:
        H = -Σ p_i log p_i
    where p_i is the empirical frequency of token i across all generated sequences.

    This is the "entropy" on the x-axis of CANDI's frontier plots. Uses the
    empirical distribution over observed tokens only (no smoothing, no |V|
    normalization) — the headline entropy number is the same whether or not
    you include zero-count support, because 0 · log 0 = 0 in the limit.
    """
    counts = Counter()
    total = 0
    for seq in sequences:
        counts.update(seq)
        total += len(seq)
    if total == 0:
        return 0.0
    H = 0.0
    for _, c in counts.items():
        p = c / total
        H -= p * math.log(p)
    return H


def _internal_perplexity(
    sampler: DiffusionSampler,
    sequences: Sequence[Sequence[int]],
    max_seqs: int = 128,
) -> float:
    """
    "Generative perplexity" under the diffusion model's OWN score.

    ⚠️ DIAGNOSTIC ONLY. Not suitable as a headline metric:
      1. Cross-tokenizer: meaningless (different support, different normalizer).
         Use judge_perplexity for cross-tokenizer comparisons.
      2. Cross-method at fixed tokenizer: still fraught. For MDLM-style
         absorbing diffusion, the training loss is only active at masked
         positions; evaluated here at noise_level=0.0 with no mask, every
         position had zero training signal, so the logits are not well-defined
         for this query. Some variants incidentally learn identity-on-clean
         (via the ELBO at small t) but it's not a property we should rely on.
      3. Semantics differ from the CANDI paper's "generative perplexity,"
         which is score-function-based. Matching their protocol exactly
         requires wiring through each method's native density.

    Bottom line: `judge_perplexity` is the headline; this is kept as a sanity
    check only.

    Returns exp of mean NLL of the (clean) sequences under the model at t=0.
    """
    if len(sequences) == 0:
        return float("nan")
    seqs = sequences[:max_seqs]
    x = torch.tensor(seqs, dtype=torch.long)
    # noise_level=0.0 = "no corruption"; MDLM/LLaDA adapters ignore this, but
    # SEDD/CANDI must honor it (see samplers.py contract).
    logits = sampler.logits_at(x, noise_level=0.0, mask=None)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, x.to(logits.device).unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(nll.mean()).item())


# ---------------------------------------------------------------------------
# OWT entropy-perplexity frontier
# ---------------------------------------------------------------------------
def entropy_perplexity_frontier(
    sampler: DiffusionSampler,
    nfe_values: Sequence[int],
    temperatures: Sequence[float],
    seq_length: int = 1024,
    n_sequences: int = 64,
    seed: int = 0,
    judge_sampler: "JudgePerplexity | None" = None,
) -> list[FrontierCurve]:
    """
    Sweep (NFE × temperature), generating n_sequences per cell, computing
    sample entropy and generative perplexity (both native and judge).

    Returns one FrontierCurve per NFE value — each curve has one point per
    temperature. This matches the layout of CANDI Figure 5/7.
    """
    curves: list[FrontierCurve] = []
    for nfe in nfe_values:
        curve = FrontierCurve(
            method=sampler.config.model_name,
            nfe=nfe,
            tokenizer_name=sampler.config.tokenizer_name,
            corpus="owt",
        )
        for tau in temperatures:
            seqs = sampler.sample(
                n_sequences=n_sequences,
                seq_length=seq_length,
                nfe=nfe,
                temperature=tau,
                seed=seed,
            )
            H = _sample_entropy(seqs)
            try:
                gen_ppl = _internal_perplexity(sampler, seqs)
            except Exception as e:
                print(f"  [warn] internal PPL failed at τ={tau} nfe={nfe}: {e}")
                gen_ppl = None
            judge_ppl = None
            if judge_sampler is not None:
                judge_ppl = judge_sampler.perplexity_from_token_ids(
                    seqs,
                    source_tokenizer=sampler.tokenizer,
                )
            curve.points.append(FrontierPoint(
                method=sampler.config.model_name,
                nfe=nfe,
                temperature=tau,
                entropy=H,
                gen_perplexity=gen_ppl,
                judge_perplexity=judge_ppl,
                n_samples=n_sequences,
                seed=seed,
            ))
            print(
                f"  nfe={nfe:>3} τ={tau:.3f}  H={H:.3f}  "
                f"gen_ppl={gen_ppl}  judge_ppl={judge_ppl}"
            )
        curves.append(curve)
    return curves


# ---------------------------------------------------------------------------
# Text8 word-frontier (%unique, %valid)
# ---------------------------------------------------------------------------
class EnglishWordValidator:
    """
    Loads a dictionary once and validates generated words.

    Prefers nltk.corpus.words (large), falls back to /usr/share/dict/words,
    falls back to a tiny built-in list (with a loud warning).
    """

    _FALLBACK: set[str] = set("""
        the of and to in is it you that he was for on are with as i his they be at
        one have this from or had by word but not what all were we when your can
        said there use an each which she do how their if will up other about out
        many then them these so some her would make like him into time has look
        two more write go see number no way could people my than first water been
        call who oil its now find long down day did get come made may part
    """.split())

    def __init__(self) -> None:
        self.words: set[str] = set()
        # Try nltk
        try:
            from nltk.corpus import words as nltk_words
            self.words = {w.lower() for w in nltk_words.words()}
        except Exception:
            pass
        # Try /usr/share/dict/words
        if not self.words:
            for candidate in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
                p = Path(candidate)
                if p.exists():
                    self.words = {
                        w.strip().lower() for w in p.read_text().splitlines()
                        if w.strip()
                    }
                    break
        if not self.words:
            print(
                "[warn] No system word list found. Using a tiny fallback (~80 words). "
                "Install nltk and run nltk.download('words') for robust %valid measurement."
            )
            self.words = set(self._FALLBACK)

    def is_valid(self, word: str) -> bool:
        return word.lower() in self.words


_WORD_RE = re.compile(r"[a-z]+")


def text8_word_frontier(
    sampler: DiffusionSampler,
    nfe_values: Sequence[int],
    temperatures: Sequence[float],
    source_tokenizer: PreTrainedTokenizerBase,
    seq_length: int = 1024,
    n_sequences: int = 64,
    seed: int = 0,
) -> list[FrontierCurve]:
    """
    Text8-scale metrics: %unique words, %valid words.

    The Text8 vocabulary is a-z and space. Each generated sequence is decoded
    to a string, lowercased, and split into words on whitespace. Unique = fraction
    of distinct word types among all tokens; Valid = fraction in an English
    dictionary.

    `source_tokenizer` is required and must be the tokenizer that produced the
    ids the sampler generates (ordinarily `sampler.tokenizer`). We don't default
    it, because decoding with the wrong tokenizer silently produces garbage
    without any error — catching that mistake at the call site is worth the
    extra parameter.
    """
    validator = EnglishWordValidator()

    curves: list[FrontierCurve] = []
    for nfe in nfe_values:
        curve = FrontierCurve(
            method=sampler.config.model_name,
            nfe=nfe,
            tokenizer_name=sampler.config.tokenizer_name,
            corpus="text8",
        )
        for tau in temperatures:
            seqs = sampler.sample(
                n_sequences=n_sequences,
                seq_length=seq_length,
                nfe=nfe,
                temperature=tau,
                seed=seed,
            )
            all_words: list[str] = []
            for ids in seqs:
                text = source_tokenizer.decode(ids, skip_special_tokens=True).lower()
                all_words.extend(_WORD_RE.findall(text))
            if not all_words:
                pct_unique = 0.0
                pct_valid = 0.0
            else:
                pct_unique = len(set(all_words)) / len(all_words)
                pct_valid = sum(validator.is_valid(w) for w in all_words) / len(all_words)
            curve.points.append(FrontierPoint(
                method=sampler.config.model_name,
                nfe=nfe,
                temperature=tau,
                pct_unique_words=pct_unique,
                pct_valid_words=pct_valid,
                n_samples=n_sequences,
                seed=seed,
            ))
            print(
                f"  nfe={nfe:>3} τ={tau:.3f}  "
                f"unique={100*pct_unique:.1f}%  valid={100*pct_valid:.1f}%"
            )
        curves.append(curve)
    return curves


# ---------------------------------------------------------------------------
# Judge perplexity (cross-tokenizer comparability)
# ---------------------------------------------------------------------------
class JudgePerplexity:
    """
    Re-tokenize generations with a fixed judge LM and report its perplexity on them.

    Why: generative perplexity under each method's own tokenizer is not comparable
    across tokenizers because the log-likelihood is over different symbol spaces.
    Using a single fixed judge (GPT-2 Large for English, IndicBERT or a causal
    Hindi LM for Hindi) makes numbers cross-comparable.

    Implementation: decode generated token ids to text with the source tokenizer,
    encode with the judge's tokenizer, compute judge NLL in chunks.
    """

    def __init__(
        self,
        judge_id: str = "gpt2-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        max_length: int = 1024,
    ) -> None:
        print(f"[judge] loading {judge_id}")
        self.tok = AutoTokenizer.from_pretrained(judge_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_id, torch_dtype=dtype
        ).to(device).eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def perplexity_from_token_ids(
        self,
        sequences: Sequence[Sequence[int]],
        source_tokenizer: PreTrainedTokenizerBase,
    ) -> float:
        """
        Decode source token ids → text → encode with judge → compute judge PPL.

        Returns the exponentiated mean NLL in nats, following the convention used
        by the diffusion-LM literature (Sahoo et al. 2024).

        `source_tokenizer` is the tokenizer that produced `sequences` — pass
        `sampler.tokenizer` directly rather than looking up by name, because
        the name→tokenizer registry and the sampler's actual tokenizer can
        diverge (e.g. LLaDA's 126K-vocab tokenizer vs. a "llama2" config).
        """
        total_nll = 0.0
        total_toks = 0
        for ids in sequences:
            text = source_tokenizer.decode(ids, skip_special_tokens=True)
            enc = self.tok(
                text, return_tensors="pt",
                truncation=True, max_length=self.max_length,
            )
            input_ids = enc.input_ids.to(self.device)
            if input_ids.size(1) < 2:
                continue
            out = self.model(input_ids, labels=input_ids)
            # out.loss is per-token mean NLL; multiply back to total nats
            n_tok = input_ids.size(1) - 1
            total_nll += out.loss.item() * n_tok
            total_toks += n_tok
        if total_toks == 0:
            return float("nan")
        return float(math.exp(total_nll / total_toks))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_curves(curves: Iterable[FrontierCurve], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in curves], f, indent=2)