# Vocabulary Scaling and Positional-Embedding Ablations for Discrete, Continuous, and Hybrid Diffusion Language Models

## 1. Framing

CANDI (Pynadath et al., ICLR/under-review 2026) argues that continuous diffusion underperforms on discrete data because of **temporal dissonance**: discrete identity corruption ρ(t) scales with vocabulary size |V| (equation 5 — a Gaussian-CDF expression whose tail involves (Φ(s/σ))^(|V|-1)), while continuous rank degradation r(t) does not (equation 6 — Φ(-1/(σ√2)) is independent of |V|). The two corruption axes therefore decouple in an adversarial way as |V| grows: you either condition on clean anchors or learn a meaningful score function, never both.

This is a testable scaling law, and the natural next question — which you're raising — is how it interacts with two axes that the paper does not systematically control:

1. **Tokenizer choice.** CANDI tests Text8 (|V|≈27) and GPT-2 BPE (|V|=50,257). That's two points. Modern LM tokenizers span roughly three orders of magnitude (character-level 27 → ByT5 256 → Llama-2 32K → GPT-2 50K → Qwen 152K → mT5 250K → Gemma 256K), and Indic tokenizers (MuRIL, IndicBERT, Sangraha-trained SentencePiece) add another axis.
2. **Positional encoding.** Bidirectional diffusion transformers are an understudied setting for PE. The NoPE length-generalization results (Kazemnejad et al.) were in causal LMs; Gala et al. (IndicTrans2 work) show NoPE fails in encoder-decoder settings; Barbero et al. ("Round and Round We Go") prove RoPE can build positional-attention circuits that NoPE structurally cannot. None of this has been pushed through to diffusion LMs, and hybrid diffusion's state heterogeneity (clean / masked / Gaussian-corrupted positions coexisting in a single input) is exactly the regime where PE choice should matter most.

The combination gives a principled 2-axis ablation grid with sharp, pre-registerable predictions.

---

## 2. Axes and Choices

### 2.1 Vocabulary size axis

Two sub-strategies — **controlled** and **realistic** — because they answer different questions.

**Controlled vocab sweep (isolates |V| effect):**
Train a family of SentencePiece BPE tokenizers on the same corpus (OpenWebText subset + a Hindi slice from Sangraha for the Indic variant) at vocab sizes {4K, 8K, 16K, 32K, 64K, 128K, 256K}. Every other tokenizer property (algorithm, pre-tokenization, training corpus) held fixed. This is the cleanest way to verify the ρ(t) vs r(t) scaling prediction — plot empirical argmax-corruption rate and rank-degradation against your trained σ(t) schedule and compare to equations 5 and 6.

**Realistic tokenizers (tests practical transfer):**
Pick five production tokenizers that span |V| with different design philosophies:

| Tokenizer | \|V\| | Script coverage | Notes |
|---|---|---|---|
| Character (Text8) | 27 | English lowercase | Anchor low end |
| ByT5 | 256 | Byte-level | No subword structure |
| GPT-2 BPE | 50,257 | Latin-heavy | CANDI's reference point |
| Llama-2 BPE | 32,000 | Latin + some Unicode | Common modern baseline |
| Qwen2.5 | 151,936 | Multilingual w/ CJK | Large non-Indic vocab |
| Gemma-2 / Gemma-3 | 256,000 | Broad multilingual | Near-upper-bound |
| Sangraha-SPM (custom) | ~128K | Hindi-focused | Indic angle |

The Sangraha tokenizer is the one that matters for your larger Indic thread — if temporal dissonance is real, Indic continuous diffusion should be disproportionately crippled vs. discrete, because Indic BPE vocabs are large relative to English equivalents at the same bit-per-token budget.

### 2.2 Diffusion-model axis

Five methods spanning the discrete–continuous spectrum. Keep the backbone transformer fixed so PE is the only architectural degree of freedom.

| Method | Class | Key reference | Repo status |
|---|---|---|---|
| MDLM | Masked discrete | Sahoo et al. 2024 | Stable, you've run it |
| SEDD (absorbing + uniform) | Score-entropy discrete | Lou et al. 2024 | Non-standard API, you've debugged it |
| LLaDA-8B | Scaled masked discrete | Nie et al. 2024/25 | You've benchmarked it |
| Continuous one-hot | Pure continuous | CANDI baseline | Simple VE-SDE on one-hot |
| CANDI | Hybrid | Pynadath et al. 2026 | Project code at the linked page |

Optional additions worth including if compute allows:
- **DUO** (Sahoo et al. 2025, uniform-state, curriculum-distilled) — directly relevant baseline from the CANDI paper.
- **Embedding diffusion** (Diffusion-LM / SSD-LM style) — CANDI shows it mode-collapses on OWT at |V|=50K; worth reproducing as a second failure mode alongside one-hot.
- **CADD or CCDD** — concurrent hybrid work cited in CANDI §7. Including one of these sharpens the claim about *why* hybrids help.

### 2.3 Positional-embedding axis

The five candidates below cover the meaningful design space. Results from your search and the literature suggest strong predictions for each.

| PE | Mechanism | Relevant prior finding |
|---|---|---|
| **NoPE** | Identity | Kazemnejad: strong length-gen in causal LMs; Gala et al.: fails in enc-dec; Barbero et al.: cannot construct certain attention circuits |
| **RoPE (θ=10K)** | Rotary, relative | De facto standard; Barbero: high-freq bands do positional attention |
| **RoPE (θ=500K)** | NTK / extended base | Llama-3 style; better long-context |
| **ALiBi** | Additive distance bias | Press et al.; simple length-gen |
| **Sinusoidal abs.** | Additive absolute | Original transformer; weak baseline |
| **Hybrid RNoPE** | Interleave NoPE/RoPE layers | Yang et al. 2025 — worth including as a simple "best of both" |

If compute is tight, the core contrast is **NoPE vs RoPE(10K) vs RoPE(500K)**; everything else is nice-to-have.

---

## 3. Hypotheses

State these ahead of time so the experiment is falsifiable.

**H1 — Temporal dissonance replicates across tokenizers.** For pure continuous one-hot diffusion, generative-frontier collapse (high perplexity or mode collapse across all temperatures) appears monotonically as |V| crosses a threshold. Predict threshold around |V| ∈ [10³, 10⁴] based on equation 5. *Falsifier:* continuous one-hot matches MDLM at |V|=50K or higher.

**H2 — Hybrid recovers linearly with |V|.** CANDI's frontier tracks or dominates MDLM at all tested vocab sizes, including Gemma-256K and Indic-128K. *Falsifier:* CANDI frontier degrades below MDLM at |V| > 100K.

**H3 — PE interacts with diffusion type more than with vocab.** NoPE underperforms RoPE for all diffusion methods because bidirectional attention (common to all diffusion LMs here) gives NoPE no implicit positional signal. Effect size is largest for continuous / hybrid because joint position updates amplify any positional miscalibration. *Falsifier:* NoPE matches RoPE for CANDI but fails for MDLM — would imply the opposite interaction.

**H4 — Length generalization is PE-bottlenecked only when diffusion is continuous.** Train at L=1024, evaluate frontier at L=2048, 4096. Predict:
- RoPE(500K) > RoPE(10K) > NoPE for continuous / hybrid;
- MDLM tolerates any PE because discrete ancestral updates don't exploit positional coordination anyway.

**H5 — Indic asymmetry.** Hybrid diffusion's advantage over masked diffusion is larger on Hindi (Sangraha-128K tokenizer) than on English (GPT-2 50K) at matched NFE, because the larger Indic vocab pushes pure-masked further into the regime where joint updates would help. This is directly relevant to your dLLM-eval / Indic thread.

---

## 4. Experimental Matrix

### 4.1 Full factorial (reference only — too expensive)

```
Tokenizers (7) × Diffusion methods (5) × PEs (5) × NFE (4) × Seeds (3) = 2100 evaluations
```

### 4.2 Phased plan (actually runnable)

**Phase 1 — Vocab scaling (controlled SPM family)**
Fix PE = RoPE(10K). Train {MDLM, SEDD-absorb, Continuous one-hot, CANDI} on a ~500M-token slice at vocab sizes {8K, 32K, 128K, 256K}. Sequence length 1024. Generative frontier at NFE ∈ {8, 16, 32, 64}. **Output:** empirical confirmation or refutation of H1, H2.

*Size:* 4 methods × 4 vocabs × 4 NFE × ~8 temperatures ≈ 512 evaluations. Training is the bottleneck — 16 runs total. At 110M parameters and ~1M steps each, ~2–4 GPU-days per run on a single A100/H100. Reuse checkpoints from your Qwen2.5-Math / Llama-3.1 infrastructure where possible for the transformer backbone.

**Phase 2 — PE ablation at fixed |V|**
Fix tokenizer = GPT-2 (|V|=50K, matches CANDI baseline). Train {MDLM, CANDI, Continuous one-hot} × {NoPE, RoPE(10K), RoPE(500K), ALiBi} = 12 models. **Output:** H3 empirical estimate.

*Size:* 12 training runs, same eval protocol. This is the cleanest ablation and should probably be the first thing run in full.

**Phase 3 — Length generalization**
Take Phase 2 checkpoints, evaluate frontier at L ∈ {1024, 2048, 4096}. No retraining. **Output:** H4.

**Phase 4 — Indic / large-|V| extension**
Phase 1 subset (MDLM + CANDI only, RoPE(10K)) rerun on:
- Sangraha Hindi + Sangraha-trained SPM-128K
- Gemma-256K tokenizer on multilingual slice
**Output:** H5 + establishes the Indic-diffusion story independently of temporal-dissonance framing.

**Phase 5 (stretch) — LLaDA / SEDD at realistic scale**
Only the PE-ablation subset (NoPE vs RoPE) on LLaDA-style 1B+ parameter scale, using your existing LLaDA/SEDD checkpoints as starting points. LoRA adapters over different PE swaps may be enough to capture the delta without full retraining — this is where your vLLM + LoRA stack is directly useful.

Realistic total: ~30–40 training runs in phases 1–3, maybe 6–8 more in 4–5. Most are 110M-parameter; LLaDA-scale is only at the end and can be LoRA-based.

---

## 5. Implementation Notes

### 5.1 Backbone and fair comparison

Use the DiT-style transformer from CANDI / MDLM / DUO (Peebles & Xie 2023), ~110M parameters, so you can directly anchor numbers against published frontiers. Single backbone across methods; the *only* per-method differences should be:
- Loss (masked CE vs score-entropy vs hybrid ELBO);
- Corruption kernel;
- Inference sampler.

Keep optimizer (AdamW), data pipeline, sequence length, and steps identical across runs.

### 5.2 PE swap plumbing

Implement PE as a drop-in module at attention:
- NoPE: identity.
- Sinusoidal: add to token embedding before layer 1.
- Learned absolute: same, trainable.
- RoPE: rotate q, k inside attention; θ as config.
- ALiBi: additive bias to attention logits.
- Hybrid (RNoPE): alternate layers.

Crucial subtlety for diffusion: the mask token in MDLM and the Gaussian-corrupted positions in CANDI still occupy positional slots. PE must be applied to them the same way as clean tokens, not skipped. Check this explicitly when porting — SEDD's codebase has some unusual handling here you've already debugged.


### 5.4 Reproducing the τ-sweep for frontiers

CANDI's §6.1 argument about the single-temperature trap is important: a single-point entropy comparison can flip rankings. Sweep τ ∈ [0.7, 1.0] at 0.025 steps (13 points) per (method, NFE) cell. This is ~20× the naïve eval budget but non-negotiable for fair comparisons.

### 5.5 Measuring the theoretical quantities directly

Before (or alongside) training, implement ρ(t) and r(t) as Monte-Carlo estimators. For each (tokenizer, noise schedule) pair, sample one-hots, add Gaussian noise, compute empirical argmax-corruption and fraction-of-incorrect-tokens-exceeding-correct. Plot against equations 5 and 6. This is ~30 lines of code and gives you:

1. A direct empirical test of the token-identifiability framework before touching any generative metric;
2. A diagnostic for when CANDI's linear ρ–r decoupling (figure 4 of the paper) actually holds under your exact noise schedule.

Skipping this and going straight to generation metrics is a common mistake — you lose the ability to attribute failure to dissonance vs. training dynamics vs. sampler bugs.

---

## 6. Evaluation

### 6.1 Primary metrics

- **Entropy–perplexity frontier** (OWT-scale) — sweep τ, report full curve, not single points.
- **%unique-words / %valid-words frontier** (Text8-scale) — for the character-level anchor.
- **Per-tokenizer generative perplexity** evaluated under a fixed auxiliary LM (e.g., GPT-2 Large for English, IndicBERT for Hindi) to keep the judge consistent across tokenizer experiments. This is critical — comparing perplexities across different tokenizers directly is meaningless.

### 6.2 Diagnostic metrics

- Empirical ρ(t), r(t) vs theoretical predictions (§5.5).
- Per-position entropy: plot H(x_i | noisy context) across positions i to see if continuous methods are collapsing everywhere or just at boundaries.
- Attention-map inspection: for NoPE runs, check if attention becomes diffuse (DroPE paper's finding about gradient norms predicts it should).

### 6.3 Length-generalization metric

Ratio of perplexity at L_test to perplexity at L_train, averaged across τ. Report (L=2048 / L=1024) and (L=4096 / L=1024). Expect ≈1 for well-generalizing configs, > 1 for failures.

---

## 7. Risks, Confounders, and Decisions to Pre-register

**Tokenizer fairness.** Training data tokenized with |V|=8K has ~3× as many tokens per document as |V|=256K. If you fix training *steps*, the 8K model sees more data; if you fix *tokens*, the 256K model trains longer in wall-clock. Decision: fix total *bytes* of training data and total *compute* (FLOPs), let step counts vary. Report both.

**PE initialization scale.** RoPE's θ and ALiBi's slope schedule both have default values tuned for causal LMs on web text. Bidirectional diffusion may prefer different settings. Run a small θ sweep for RoPE ({10K, 100K, 500K}) at one vocab size to check; if sensitivity is huge, expand.

**CANDI's σ schedule.** CANDI uses a linear r*(t) schedule with [r_min, r_max] = (something close to 0, something below 0.5). This schedule hyperparameter can interact with tokenizer — different |V| may want different r_max. Worth a small sweep, or at least a sensitivity check, before declaring CANDI "worse than MDLM" at some |V|.

**Scale mismatch with LLaDA/SEDD.** Your production LLaDA and SEDD checkpoints are 1B+ parameters trained on hundreds of billions of tokens. Phase 1–3 models are 110M on ~500M tokens. Don't conflate them. Phase 5 (LoRA-over-LLaDA) is where you can test whether the story holds at production scale, but be explicit about the scale gap in any write-up.

**Multi-seed variance.** Frontier curves are noisy. Minimum 3 seeds per cell for the headline claims (Phase 2 PE ablation especially). Phase 1 can get away with 1 seed for the scaling trend, but the H3 PE comparison needs error bars.

**Evaluator LM confounds.** Using GPT-2 as the perplexity judge for a Gemma-tokenized model's outputs re-tokenizes at the boundary. This is fine as long as it's the same judge across methods within a tokenizer cell. Never compare raw perplexities across tokenizer cells directly.

---

## 8. Deliverables Aligned with Your Existing Threads

This plan produces three outputs that slot into your ongoing work:

1. **dLLM-eval extension.** The frontier-sweep infrastructure built for Phase 1 is a direct extension of the dLLM-eval pipelines you're already running on SEDD/LLaDA. The PE ablation becomes a configurable axis in that harness.

2. **Indic-diffusion research note.** Phase 4 (Indic + large-|V|) is the nucleus of a standalone paper on *why* masked diffusion currently dominates Indic LMs and what the hybrid-diffusion path looks like. Ties into the continual pre-training of LLaDA-8B on Sangraha that you've been scoping.

---

## 9. Compute Budget Estimate

Rough order of magnitude at 110M-parameter DiT:

| Phase | Training runs | Single-run cost | Subtotal (A100-days) |
|---|---|---|---|
| 1 (vocab sweep) | 16 | ~3 days | 48 |
| 2 (PE ablation) | 12 | ~3 days | 36 |
| 3 (length-gen eval) | 0 training, eval only | — | ~2 |
| 4 (Indic / Gemma) | 4 | ~4 days (larger vocab emb) | 16 |
| 5 (LLaDA LoRA) | 4 | ~2 days LoRA | 8 |
| **Total** | **36 runs** | — | **~110 A100-days** |

Evaluation adds maybe ~20% on top. Total is ~130 A100-days. On the LCS2 megatron/sentinel servers this is ~3–4 weeks wall-clock if 4 GPUs are usable in parallel.

If compute is tighter, cut Phase 4 to just MDLM+CANDI on Sangraha (skip Gemma), and skip Phase 5 entirely — the core scientific story lives in Phases 1–3.

---

## 10. Execution Order (Suggested)

1. **Week 0:** Implement ρ(t), r(t) Monte-Carlo estimators. Verify equations 5 and 6 reproduce figure 3 of CANDI paper.
2. **Week 1:** Implement PE swap module, verify numerical equivalence to reference RoPE/ALiBi implementations on a toy task.
3. **Week 2–3:** Phase 2 (PE ablation at fixed vocab) — this is the tightest scientific contribution and should be prioritized.
4. **Week 3–5:** Phase 1 (vocab sweep) in parallel with Phase 2 eval.
5. **Week 5–6:** Phase 3 (length-gen eval).
6. **Week 6–8:** Phase 4 (Indic).
7. **Week 8+:** Phase 5 (LLaDA scale) if time permits.

Write-up target: a ~10-page empirical study suitable for ICLR or a diffusion-focused venue. The PE + diffusion + vocab story is cohesive enough to stand alone; H5 (Indic asymmetry) is the novelty angle that makes it more than "CANDI replication with extra axes."
