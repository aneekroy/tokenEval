"""
Smoke test.

Verifies the harness works end-to-end on a tiny synthetic corpus in under
a minute. Run this before committing a long full sweep — it catches:

- Tokenizer loading / auth problems (Gemma, Llama-2 gating)
- Fertility iterator bugs (off-by-one at chunk boundaries)
- ρ(σ) / r(σ) estimators matching CANDI's paper values at |V|=50
- JSON serialization round-trips for all dataclasses

Usage (from repo root):
    python scripts/smoke.py

Exit code 0 on success, 1 on any failure. Prints a compact summary.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

# Make src importable when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tokenizers_bench import (
    compute_fertility,
    load_tokenizer,
    run_fertility_suite,
    TOKENIZER_REGISTRY,
)
from src.corruption import (
    r_analytical,
    rho_analytical,
    estimate_corruption_mc,
)
from src.metrics.frontier import EnglishWordValidator, _sample_entropy


# ---------------------------------------------------------------------------
# Tiny synthetic corpus
# ---------------------------------------------------------------------------
TINY_ENGLISH = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is the foundation of every language model.",
    "Diffusion models generate sequences through iterative denoising.",
    "A small vocabulary is easy; a large one is harder to denoise.",
    "Positional embeddings like RoPE and NoPE encode different biases.",
] * 20     # 100 sentences

TINY_HINDI = [
    "यह एक परीक्षण वाक्य है।",
    "प्राकृतिक भाषा प्रसंस्करण दिलचस्प है।",
    "टोकनकरण विभिन्न भाषाओं में अलग-अलग काम करता है।",
] * 20


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        print(f"  ✓ {name}")
        PASSED.append(name)
    except AssertionError as e:
        print(f"  ✗ {name}: {e}")
        FAILED.append((name, str(e)))
    except Exception as e:
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        FAILED.append((name, f"{type(e).__name__}: {e}"))


# -- Tokenizer registry sanity
def test_registry_has_expected_tokenizers():
    expected = {"byt5", "llama2", "gpt2", "qwen25", "gemma3", "llada"}
    assert expected.issubset(TOKENIZER_REGISTRY.keys()), \
        f"registry missing entries: {expected - TOKENIZER_REGISTRY.keys()}"


# -- Load one fast tokenizer (GPT-2 is ungated, always available)
def test_gpt2_loads_and_encodes():
    tok = load_tokenizer("gpt2")
    ids = tok.encode("hello world", add_special_tokens=False)
    assert len(ids) >= 2
    decoded = tok.decode(ids)
    assert "hello" in decoded.lower()


# -- Fertility on tiny corpus
def test_fertility_gpt2_english():
    tok = load_tokenizer("gpt2")
    stats = compute_fertility(tok, TINY_ENGLISH, "gpt2", "tiny_english")
    assert not stats.is_empty
    assert stats.n_documents == len(TINY_ENGLISH)
    assert stats.n_tokens > 0
    # GPT-2 English fertility should be roughly 1.2–1.4 tokens/word
    assert 1.0 < stats.tokens_per_word_mean < 1.8, \
        f"unreasonable GPT-2 English fertility: {stats.tokens_per_word_mean}"
    assert stats.tokens_per_word_p99 >= stats.tokens_per_word_mean
    # Unk rate should be very low on clean English
    assert stats.unk_rate < 0.01


def test_fertility_byt5_english_is_high():
    tok = load_tokenizer("byt5")
    stats = compute_fertility(tok, TINY_ENGLISH, "byt5", "tiny_english")
    # ByT5 is byte-level; English words are ~5 bytes → fertility ~5
    assert stats.tokens_per_word_mean > 3.0, \
        f"ByT5 fertility suspiciously low: {stats.tokens_per_word_mean}"


def test_fertility_empty_corpus_returns_stats():
    """Regression for the previous behavior of raising on empty corpora,
    which killed the whole tokenizer × corpus grid in run_fertility_suite."""
    tok = load_tokenizer("gpt2")
    stats = compute_fertility(tok, [], "gpt2", "empty_corpus")
    assert stats.is_empty
    assert stats.n_documents == 0
    assert stats.tokens_per_word_mean == 0.0


# -- ρ(σ) and r(σ) against CANDI paper values at |V|=50
def test_r_analytical_matches_paper():
    # CANDI Fig. 3 at |V|=50, the σ where ρ=0.5 gives r≈0.0659.
    # We verify the r formula independently: if r=0.0659 then σ solves
    # Φ(-1/(σ√2)) = 0.0659 → −1/(σ√2) = Φ⁻¹(0.0659) ≈ -1.508
    # → σ ≈ 1 / (1.508 √2) ≈ 0.469
    sigma = 0.469
    r = r_analytical(sigma)
    assert abs(r - 0.0659) < 0.005, f"r({sigma})={r:.4f}; expected ~0.0659"


def test_rho_analytical_monotone_in_V():
    # At fixed σ, larger |V| → more discrete corruption
    sigma = 0.5
    rhos = [rho_analytical(sigma, V) for V in [5, 50, 500, 5000]]
    for a, b in zip(rhos, rhos[1:]):
        assert a < b + 1e-6, f"ρ not monotone in |V|: {rhos}"


def test_mc_estimator_agrees_with_theory():
    # |V|=50 is well within quadrature's stability range.
    # Explicit seed=0 so this doesn't flake if the default ever changes.
    est = estimate_corruption_mc(
        sigma=1.0, vocab_size=50, n_samples=5000, device="cpu", seed=0,
    )
    assert abs(est.rho_mc - est.rho_theory) < 0.03, \
        f"ρ MC={est.rho_mc:.3f} vs theory={est.rho_theory:.3f}"
    assert abs(est.r_mc - est.r_theory) < 0.01, \
        f"r MC={est.r_mc:.3f} vs theory={est.r_theory:.3f}"


# -- Metric utilities
def test_sample_entropy_uniform_is_log_V():
    # Uniform over 10 symbols → entropy = log(10) ≈ 2.303 nats
    seqs = [list(range(10)) * 100]
    H = _sample_entropy(seqs)
    assert abs(H - math.log(10)) < 0.01, f"H={H} not close to log(10)={math.log(10):.3f}"


def test_sample_entropy_degenerate_is_zero():
    seqs = [[7] * 1000]
    H = _sample_entropy(seqs)
    assert H < 1e-6


def test_english_word_validator():
    v = EnglishWordValidator()
    # "the" and "and" should be in any dictionary, real or fallback
    assert v.is_valid("the")
    assert v.is_valid("and")
    # Gibberish should not
    assert not v.is_valid("zxqvwptbk")


# -- Full fertility suite on 2 tokenizers x 1 corpus (end-to-end)
def test_run_fertility_suite_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        results = run_fertility_suite(
            corpora={"tiny_english": TINY_ENGLISH},
            tokenizer_names=["gpt2", "byt5"],   # both ungated, small, fast
            out_dir=out,
        )
        # Artifacts
        assert (out / "fertility.json").exists()
        assert (out / "fertility_report.md").exists()
        # Round-trip
        blob = json.loads((out / "fertility.json").read_text())
        assert "gpt2" in blob and "byt5" in blob
        assert blob["gpt2"]["tiny_english"]["n_documents"] == len(TINY_ENGLISH)
        # Cross-check byT5 > gpt2 on tokens/word for English
        byt5_fert = results["byt5"]["tiny_english"].tokens_per_word_mean
        gpt2_fert = results["gpt2"]["tiny_english"].tokens_per_word_mean
        assert byt5_fert > gpt2_fert, \
            f"byT5 ({byt5_fert}) should have higher fertility than GPT-2 ({gpt2_fert})"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print("=== diffusion_tokenizer_bench smoke test ===\n")

    print("[1] Tokenizer registry & loading")
    _run("registry has expected tokenizers", test_registry_has_expected_tokenizers)
    _run("gpt2 loads and encodes", test_gpt2_loads_and_encodes)

    print("\n[2] Fertility")
    _run("gpt2 English fertility in sane range", test_fertility_gpt2_english)
    _run("byt5 English fertility > gpt2", test_fertility_byt5_english_is_high)
    _run("empty corpus returns stats (no raise)", test_fertility_empty_corpus_returns_stats)

    print("\n[3] Corruption estimators")
    _run("r analytical matches paper", test_r_analytical_matches_paper)
    _run("rho analytical monotone in |V|", test_rho_analytical_monotone_in_V)
    _run("MC estimator agrees with theory", test_mc_estimator_agrees_with_theory)

    print("\n[4] Metric utilities")
    _run("sample entropy uniform = log(V)", test_sample_entropy_uniform_is_log_V)
    _run("sample entropy degenerate = 0", test_sample_entropy_degenerate_is_zero)
    _run("English word validator", test_english_word_validator)

    print("\n[5] End-to-end")
    _run("fertility suite end-to-end", test_run_fertility_suite_end_to_end)

    print(f"\n=== {len(PASSED)}/{len(PASSED) + len(FAILED)} tests passed ===")
    if FAILED:
        print("\nFailures:")
        for name, err in FAILED:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())