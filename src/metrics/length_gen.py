"""
Length-generalization metric (Section 6.3).

Train at L_train, evaluate at L_test, report PPL(L_test) / PPL(L_train).
Values near 1 → good length generalization. Values >> 1 → PE-bottlenecked.

Predicted outcome under H4:
    - RoPE(θ=500K) ≈ 1.0 across L_test ∈ {2×, 4×} training length
    - RoPE(θ=10K)   grows with L_test
    - NoPE          grows sharply for continuous / hybrid, stays flat for masked

Evaluation strategy:
    - For each L_test, generate N sequences at that length using the same seeds
    - Compute judge perplexity (so cross-length comparisons are well-defined)
    - Bootstrap error bars over seeds
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from ..samplers import DiffusionSampler
from .frontier import JudgePerplexity


@dataclass
class LengthGenResult:
    method: str
    tokenizer_name: str
    pe_type: str
    l_train: int
    per_length_ppl: dict[int, float]        # L_test -> judge PPL
    per_length_ratio: dict[int, float]      # L_test -> PPL(L_test) / PPL(L_train)
    per_length_ppl_by_tau: dict[int, dict[float, float]]  # keyed by L then τ
    temperatures: list[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON doesn't like int keys
        d["per_length_ppl"] = {str(k): v for k, v in d["per_length_ppl"].items()}
        d["per_length_ratio"] = {str(k): v for k, v in d["per_length_ratio"].items()}
        d["per_length_ppl_by_tau"] = {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in d["per_length_ppl_by_tau"].items()
        }
        return d


def length_generalization(
    sampler: DiffusionSampler,
    l_train: int,
    l_test_values: Sequence[int],
    temperatures: Sequence[float],
    nfe: int,
    pe_type: str,
    judge: JudgePerplexity,
    n_sequences_per_cell: int = 32,
    seed: int = 0,
) -> LengthGenResult:
    """
    Sweep (L_test × τ) for a fixed NFE and method.

    The ratio metric is computed against the τ-averaged PPL at L_train, so it's
    robust to a single unlucky temperature.
    """
    ppl_by_tau: dict[int, dict[float, float]] = {}
    for L in l_test_values:
        ppl_by_tau[L] = {}
        for tau in temperatures:
            seqs = sampler.sample(
                n_sequences=n_sequences_per_cell,
                seq_length=L,
                nfe=nfe,
                temperature=tau,
                seed=seed,
            )
            ppl = judge.perplexity_from_token_ids(
                seqs, source_tokenizer=sampler.tokenizer,
            )
            ppl_by_tau[L][tau] = ppl
            print(f"  L={L}  τ={tau:.3f}  judge_ppl={ppl:.2f}")

    # Aggregate: geometric mean across τ (robust to outlier-τ behavior)
    per_length_ppl: dict[int, float] = {}
    for L, by_tau in ppl_by_tau.items():
        logs = [math.log(v) for v in by_tau.values() if v and math.isfinite(v)]
        if logs:
            per_length_ppl[L] = float(math.exp(sum(logs) / len(logs)))
        else:
            per_length_ppl[L] = float("nan")

    ref = per_length_ppl.get(l_train, float("nan"))
    per_length_ratio = {L: (v / ref if ref and math.isfinite(ref) else float("nan"))
                        for L, v in per_length_ppl.items()}

    return LengthGenResult(
        method=sampler.config.model_name,
        tokenizer_name=sampler.config.tokenizer_name,
        pe_type=pe_type,
        l_train=l_train,
        per_length_ppl=per_length_ppl,
        per_length_ratio=per_length_ratio,
        per_length_ppl_by_tau=ppl_by_tau,
        temperatures=list(temperatures),
    )