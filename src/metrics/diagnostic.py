"""
Diagnostic metrics (Section 6.2).

These are for debugging and understanding, not for headline numbers. They're
useful when a frontier looks bad and you want to know *why*: is the model
collapsing at sequence boundaries? Are attention maps going diffuse under NoPE?
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..samplers import DiffusionSampler


# ---------------------------------------------------------------------------
# Per-position entropy H(x_i | noisy context)
# ---------------------------------------------------------------------------
@dataclass
class PerPositionEntropy:
    method: str
    noise_level: float
    seq_length: int
    mean_per_position: list[float]    # length = seq_length
    std_per_position: list[float]
    n_batches: int

    def to_dict(self) -> dict:
        return asdict(self)


def per_position_entropy(
    sampler: DiffusionSampler,
    token_sequences: Sequence[Sequence[int]],
    noise_levels: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    mask_fraction_at: dict[float, float] | None = None,
    batch_size: int = 8,
) -> list[PerPositionEntropy]:
    """
    For each noise level t, compute the model's predictive entropy at every
    position of the input sequence, averaged over many real-document conditions.

    Protocol:
      1. Take real sequences (from data.iter_token_sequences).
      2. At each noise level t, construct a mask with Bernoulli probability
         `mask_fraction_at[t]` (defaults to identity: t% positions masked).
      3. Run sampler.logits_at(x, t, mask) → [B, L, V].
      4. Compute softmax entropy at every position, average across batch.

    Reveals whether failures are spatially concentrated (e.g. edges only) or
    uniform — the difference matters for interpretation.
    """
    if mask_fraction_at is None:
        mask_fraction_at = {t: t for t in noise_levels}

    device = sampler.device
    seqs = torch.tensor(list(token_sequences), dtype=torch.long)
    N, L = seqs.shape

    results: list[PerPositionEntropy] = []
    for t in noise_levels:
        frac = mask_fraction_at[t]
        rng = torch.Generator().manual_seed(0)
        running_sum = torch.zeros(L)
        running_sqsum = torch.zeros(L)
        n_batches = 0
        for start in range(0, N, batch_size):
            x = seqs[start:start + batch_size]
            mask = (torch.rand(x.shape, generator=rng) < frac)
            logits = sampler.logits_at(x, noise_level=t, mask=mask)
            probs = F.softmax(logits.float(), dim=-1)
            # H = -Σ p log p   (clamp for numerical safety)
            H = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
            Hm = H.mean(dim=0).cpu()      # [L]
            running_sum += Hm
            running_sqsum += Hm ** 2
            n_batches += 1
        mean = (running_sum / n_batches).tolist()
        var = (running_sqsum / n_batches - (running_sum / n_batches) ** 2)
        std = var.clamp_min(0).sqrt().tolist()
        results.append(PerPositionEntropy(
            method=sampler.config.model_name,
            noise_level=t,
            seq_length=L,
            mean_per_position=mean,
            std_per_position=std,
            n_batches=n_batches,
        ))
    return results


# ---------------------------------------------------------------------------
# Attention-map diffuseness
# ---------------------------------------------------------------------------
@dataclass
class AttentionDiffuseness:
    method: str
    per_layer_mean_entropy: list[float]   # layer -> avg over heads, positions
    per_layer_per_head: list[list[float]]  # [layer][head]
    max_entropy: float                     # log(seq_length) = uniform

    def to_dict(self) -> dict:
        return asdict(self)


def attention_diffuseness(
    sampler: DiffusionSampler,
    token_sequences: Sequence[Sequence[int]],
    noise_level: float = 0.0,
    batch_size: int = 4,
) -> AttentionDiffuseness | None:
    """
    For each attention head, compute the mean entropy of its attention
    distribution across query positions. High values → diffuse / uniform, low
    → sharp / localized.

    Motivated by the DroPE / Kazemnejad finding that NoPE transformers have
    bounded attention-gradient norms at init, which manifests as persistently
    diffuse attention. If true for diffusion LMs, a NoPE CANDI should show
    higher per-head entropy than a RoPE CANDI.
    """
    seqs = torch.tensor(list(token_sequences), dtype=torch.long)
    N, L = seqs.shape

    # Warm-up a batch to find shape
    first = sampler.attention_maps(seqs[:batch_size], noise_level=noise_level)
    if first is None:
        print("[diag] attention_maps not supported by this sampler; skipping.")
        return None

    n_layers = len(first)
    n_heads = first[0].shape[1] if first[0].ndim >= 3 else 0

    per_layer_sum = [torch.zeros(n_heads) for _ in range(n_layers)]
    n_batches = 0
    for start in range(0, N, batch_size):
        x = seqs[start:start + batch_size]
        atts = sampler.attention_maps(x, noise_level=noise_level)
        if atts is None:
            continue
        for li, a in enumerate(atts):
            # a: [B, H, L, L]  attention weights (rows sum to 1 over keys)
            a = a.float().clamp_min(1e-12)
            H = -(a * a.log()).sum(-1)           # [B, H, L]   entropy over keys
            H_mean_hb = H.mean(dim=(0, 2))        # [H]        across batch & queries
            per_layer_sum[li] += H_mean_hb.cpu()
        n_batches += 1

    per_head = [(x / max(n_batches, 1)).tolist() for x in per_layer_sum]
    per_layer = [float(np.mean(x)) for x in per_head]
    return AttentionDiffuseness(
        method=sampler.config.model_name,
        per_layer_mean_entropy=per_layer,
        per_layer_per_head=per_head,
        max_entropy=float(np.log(L)),
    )
