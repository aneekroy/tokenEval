"""
Token-identifiability estimators.

Implements the two quantities from CANDI (Pynadath et al. 2026):

    ρ(σ) = P( argmax(one_hot + Gaussian) ≠ correct_index )
         = ∫ [1 - Φ(s/σ)^(|V|-1)] · N(s; 1, σ²) ds       (equation 5)

    r(σ) = E[ #{incorrect j : noisy[j] > noisy[i]} / (|V|-1) ]
         = Φ(-1/(σ√2))                                   (equation 6)

ρ depends on |V|, r does not — this asymmetry is the "temporal dissonance"
CANDI identifies. These estimators let you confirm the analytical predictions
empirically before committing any training compute.

Both analytical and Monte Carlo estimators are provided. At |V| in the hundreds
of thousands, the analytical integrand for ρ has a very sharp transition and
quadrature becomes unreliable; we use log-space quadrature there and fall back
to MC with a large sample count.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy import integrate, stats


# Above this |V|, analytical quadrature for ρ is known to be unreliable — the
# integrand becomes a very thin spike that quad's adaptive strategy can miss.
# We warn in that regime and recommend MC.
_RHO_QUADRATURE_V_LIMIT = 100_000


# ---------------------------------------------------------------------------
# Analytical r(σ)   —   equation 6 of CANDI
# ---------------------------------------------------------------------------
def r_analytical(sigma: float | np.ndarray) -> float | np.ndarray:
    """Continuous rank degradation, |V|-independent."""
    return stats.norm.cdf(-1.0 / (np.asarray(sigma) * np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Analytical ρ(σ, |V|)   —   equation 5 of CANDI
# ---------------------------------------------------------------------------
def rho_analytical(sigma: float, vocab_size: int, use_log: bool = True) -> float:
    """
    Discrete identity corruption rate.

    Computes  1 - ∫ Φ(s/σ)^(V-1) · N(s; 1, σ²) ds.

    For large V and moderate σ, Φ(s/σ)^(V-1) is 0 for most s and 1 for a thin
    band. We integrate in log space over that band for numerical stability.

    ⚠️ At |V| > ~10⁵ the integrand is a very thin spike and quad can miss the
    band between ~σ·Φ⁻¹(1/V) and ~σ+∞. A warning is emitted in that regime;
    prefer `estimate_corruption_mc` for reliable values there.
    """
    V = int(vocab_size)
    if V < 2:
        return 0.0

    if V > _RHO_QUADRATURE_V_LIMIT:
        warnings.warn(
            f"rho_analytical(σ={sigma}, |V|={V}): quadrature may be unreliable "
            f"at |V| > {_RHO_QUADRATURE_V_LIMIT:,} because the integrand is a "
            f"very thin spike. Use estimate_corruption_mc for large-vocab cases.",
            RuntimeWarning,
            stacklevel=2,
        )

    V_minus_1 = V - 1

    if use_log:
        # integrand(s) = exp[ (V-1) * log Φ(s/σ) + log N(s; 1, σ²) ]
        def log_integrand(s: float) -> float:
            log_phi = stats.norm.logcdf(s / sigma)
            log_pdf = stats.norm.logpdf(s, loc=1.0, scale=sigma)
            return V_minus_1 * log_phi + log_pdf

        def integrand(s: float) -> float:
            return float(np.exp(log_integrand(s)))
    else:
        def integrand(s: float) -> float:
            phi = stats.norm.cdf(s / sigma)
            return (phi ** V_minus_1) * stats.norm.pdf(s, loc=1.0, scale=sigma)

    # Integrate over a wide range. The correct-token marginal is N(1, σ²), so
    # most mass is within 1 ± 8σ.
    lo = 1.0 - 8.0 * sigma
    hi = 1.0 + 8.0 * sigma
    prob_correct, _ = integrate.quad(integrand, lo, hi, limit=200)
    prob_correct = float(np.clip(prob_correct, 0.0, 1.0))
    return 1.0 - prob_correct


# ---------------------------------------------------------------------------
# Monte Carlo estimators (GPU-friendly)
# ---------------------------------------------------------------------------
@dataclass
class CorruptionEstimate:
    sigma: float
    vocab_size: int
    rho_mc: float
    rho_mc_se: float        # standard error of the estimate
    r_mc: float
    r_mc_se: float
    rho_theory: float
    r_theory: float
    n_samples: int


def estimate_corruption_mc(
    sigma: float,
    vocab_size: int,
    n_samples: int = 10_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    batch_size: int | None = None,
) -> CorruptionEstimate:
    """
    Monte Carlo estimate of ρ(σ) and r(σ) at a single σ and |V|.

    Sampling scheme:
        - correct[n] ~ N(1, σ²)
        - incorrect[n, j] ~ N(0, σ²)   for j = 1 ... V-1
        - ρ : fraction of samples where any incorrect > correct
        - r : mean fraction of incorrect exceeding correct

    Memory: n_samples × vocab_size floats. At V=256K and 10K samples this is
    ~10GB; we batch automatically. At V=50K it fits in ~2GB per batch.
    """
    V = int(vocab_size)
    gen = torch.Generator(device=device).manual_seed(seed)

    # Auto-batch to stay under ~4GB / batch
    if batch_size is None:
        bytes_per = V * 4  # float32
        target_bytes = 4 * 1024 ** 3
        batch_size = max(1, min(n_samples, target_bytes // bytes_per))

    rho_hits = 0
    r_sum = 0.0
    r_sq_sum = 0.0

    n_done = 0
    while n_done < n_samples:
        bsz = min(batch_size, n_samples - n_done)
        correct = torch.randn(bsz, 1, generator=gen, device=device) * sigma + 1.0
        # We only need the MAX of incorrect and the COUNT exceeding correct.
        # To avoid allocating [bsz, V-1] we chunk over V.
        V_CHUNK = 16_384
        max_incorrect = torch.full((bsz, 1), float("-inf"), device=device)
        count_exceed = torch.zeros(bsz, 1, device=device)
        remaining = V - 1
        while remaining > 0:
            chunk = min(V_CHUNK, remaining)
            x = torch.randn(bsz, chunk, generator=gen, device=device) * sigma
            max_incorrect = torch.maximum(
                max_incorrect, x.max(dim=1, keepdim=True).values
            )
            count_exceed += (x > correct).float().sum(dim=1, keepdim=True)
            remaining -= chunk

        rho_hits += int((max_incorrect > correct).sum().item())
        frac = (count_exceed / max(V - 1, 1)).squeeze(1)
        r_sum += float(frac.sum().item())
        r_sq_sum += float((frac ** 2).sum().item())
        n_done += bsz

    rho = rho_hits / n_samples
    rho_se = float(np.sqrt(rho * (1 - rho) / n_samples))
    r_mean = r_sum / n_samples
    r_var = max(0.0, r_sq_sum / n_samples - r_mean ** 2)
    r_se = float(np.sqrt(r_var / n_samples))

    # Analytical references (r is exact; rho may fail at very large V in quadrature)
    try:
        # Suppress the high-V warning here; the whole point of calling this is
        # that we have the MC value to fall back on.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rho_th = rho_analytical(sigma, V)
    except Exception:
        rho_th = float("nan")
    r_th = float(r_analytical(sigma))

    return CorruptionEstimate(
        sigma=sigma,
        vocab_size=V,
        rho_mc=rho,
        rho_mc_se=rho_se,
        r_mc=r_mean,
        r_mc_se=r_se,
        rho_theory=rho_th,
        r_theory=r_th,
        n_samples=n_samples,
    )


def sweep_corruption(
    sigmas: Sequence[float],
    vocab_sizes: Sequence[int],
    n_samples: int = 10_000,
    **kwargs,
) -> list[CorruptionEstimate]:
    """Full sweep over (σ, |V|) — used to reproduce CANDI Figure 3."""
    results: list[CorruptionEstimate] = []
    for V in vocab_sizes:
        for sigma in sigmas:
            est = estimate_corruption_mc(
                sigma=float(sigma), vocab_size=int(V), n_samples=n_samples, **kwargs
            )
            results.append(est)
    return results