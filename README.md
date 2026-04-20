# diffusion_tokenizer_bench

Evaluation harness for tokenizer × diffusion-model interactions, built around the hypotheses in the [vocab-scaling & PE ablation plan](vocabulary_tokenizer.md). Implements the metrics in sections 6.1–6.3 of that plan.

Two things this package does, cleanly separated:

1. **Tokenizer-intrinsic comparison** (§ tokenizers_bench.py) — fertility, compression, coverage, segmentation examples across ByT5 → Llama-2 → GPT-2 → Qwen2.5 → Gemma-3. No model required.
2. **Diffusion-model evaluation** (§ metrics/) — entropy-perplexity frontiers, %unique/%valid words, fixed-judge perplexity, per-position entropy, attention diffuseness, length-generalization ratios — each evaluated per (model, NFE, τ) cell.

`samplers.py` defines a single `DiffusionSampler` protocol; the **MDLM adapter is fully implemented** as a reference; the **LLaDA adapter** uses `AutoModel` + a low-confidence remasking loop; **SEDD** and **CANDI** adapters are stubs with clearly marked integration points where you paste in the existing inference code from each upstream repo.

---

## Install

```bash
cd diffusion_tokenizer_bench
pip install -r requirements.txt
# For English validity in Text8 metrics:
python -c "import nltk; nltk.download('words')"
```

Gated tokenizers require HF auth. Set `hf_token` in the config or `export HF_TOKEN=...`. Alternatively, use the ungated mirror `NousResearch/Llama-2-7b-hf` which is baked into `TOKENIZER_REGISTRY["llama2"]` by default.

---

## Quick start

```bash
# 1. Tokenizer-only comparison (fast, ~10 min on one CPU)
python -m src.run fertility --config configs/default.yaml

# 2. Verify the theoretical corruption curves against Monte Carlo
python -m src.run corruption --config configs/default.yaml

# 3. Diffusion-model evaluation (slow; needs GPU + loaded checkpoints)
python -m src.run frontier --config configs/default.yaml

# 4. Diagnostics and length-gen
python -m src.run diagnostics --config configs/default.yaml
python -m src.run length-gen  --config configs/default.yaml
```

---

## Data Setup

Three corpora, each with a factory function in `src/data.py`:

| Corpus | Purpose | Source | Approx size |
|---|---|---|---|
| **OpenWebText** | Primary benchmark, large-vocab regime | `Skylion007/openwebtext` (streaming) | ~40GB uncompressed; we subsample |
| **Text8** | Character-level anchor (|V|=27) | `afmck/text8` | ~100MB, one long string |
| **Sangraha Hindi** *(optional)* | Indic stress test | `ai4bharat/sangraha` (verified.hin) | ~50GB full; we subsample |

Data factories return *callables* (not iterators) so the same sample set can be consumed by multiple metrics. For reproducibility, each factory accepts `n_docs` and `seed`.

**For fertility**: we iterate raw strings once per tokenizer, accumulating per-word token counts. Words are split by whitespace + punctuation (`\w+|[^\w\s]`), which works for English and Devanagari. CJK scripts would need a language-specific segmenter — the package reports bytes/token as a script-agnostic fallback.

**For diffusion evaluation**: `iter_token_sequences` concatenates documents with EOS separators and slices into fixed-length windows of 1024 tokens (matching the MDLM/CANDI/SEDD convention). This is the "packing" strategy from Diffusion-LM.

**Data volumes used per metric**:
- Fertility: 5000 OWT docs, 2000 Text8 slices, 3000 Hindi docs (configurable)
- Frontier: 64 generated sequences × 11 temperatures × 4 NFEs = 2816 generations per model
- Per-position entropy: 32 real sequences × 5 noise levels = 160 forward passes per model
- Length-gen: 32 generations × 3 lengths × 3 temperatures = 288 generations per model

---

## Tokenizers

`TOKENIZER_REGISTRY` in `src/tokenizers_bench.py` defines five default tokenizers spanning three orders of magnitude in |V|. All loaded via `AutoTokenizer.from_pretrained(..., use_fast=True)`.

| Key | \|V\| | Family | HF repo | Notes |
|---|---:|---|---|---|
| `byt5` | 256 (effective) | byte | `google/byt5-small` | Byte-level; fertility is bytes/word. |
| `llama2` | 32,000 | BPE (SentencePiece) | `NousResearch/Llama-2-7b-hf` | Ungated mirror of `meta-llama/Llama-2-7b-hf`. |
| `gpt2` | 50,257 | byte-level BPE | `gpt2` | Reference tokenizer in CANDI / MDLM / SEDD. |
| `qwen25` | 151,936 | byte-level BPE | `Qwen/Qwen2.5-7B` | Multilingual, strong CJK coverage. |
| `gemma3` | 262,144 | BPE (SentencePiece) | `google/gemma-3-4b-pt` | Gated; broad multilingual. |

**Why these five?** They span the (|V|, family, script coverage) design space without the confounders of simultaneously varying training corpus / date. For the Indic thread, you can append a `sangraha-spm` entry pointing to a custom SentencePiece trained on Sangraha — the loader handles local paths.

**Fertility semantics**:
- `tokens_per_word_mean`: the canonical "fertility" number (Rust et al. 2021).
- `tokens_per_word_p95` / `p99`: reveal rare-word pathology (compound words, Devanagari morphology). Gemma and Qwen should have much lower p99 than GPT-2 on Hindi.
- `bytes_per_token`: compression ratio, script-agnostic. Higher = denser encoding.
- `unk_rate`: fraction of tokens mapping to UNK. For byte tokenizers this is ~0; for subword tokenizers on OOV scripts it can spike.

**Segmentation examples** (automatic): the harness picks specific stress words — `antidisestablishmentarianism`, `सूर्यमंदिर`, `北京`, `🙂` — and records how each tokenizer splits them. This qualitative view sometimes tells you more than the mean fertility does.

---

## Evaluation Metrics

### 6.1 — Primary Metrics

#### Entropy-perplexity frontier (`metrics/frontier.py`)

For each (method, NFE) cell, sweep τ ∈ {0.700, 0.725, …, 1.000} (11 points by default), generate 64 sequences at length 1024, and record:

- **Entropy** (*H*): Shannon entropy of the empirical token distribution over concatenated samples. Measures diversity. Computed in nats.
- **Generative perplexity** (*gen_ppl*): exp of mean NLL of generated sequences under the generating model itself. This is what CANDI figures plot on the y-axis. **Caveat**: not comparable across tokenizers — different models assign different log-probabilities over different symbol sets.
- **Judge perplexity** (*judge_ppl*): generations decoded to strings, re-tokenized by a fixed judge (GPT-2 Large for English, any causal Hindi LM for Hindi), PPL computed under the judge. **This is the metric you use for cross-tokenizer or cross-method comparisons.**

Why sweep τ and not report a single point: CANDI §6.1 shows that small τ changes can flip method rankings. The frontier — full curve across all τ — is what's comparable. A method "wins" only if its curve dominates at every entropy level in the target range.

Output: one `FrontierCurve` per NFE, containing one `FrontierPoint` per τ.

#### Text8 %unique / %valid (`metrics/frontier.py::text8_word_frontier`)

For character-level evaluation. Decode each generated token sequence to a string, lowercase, extract words with `[a-z]+`. Report:

- **%unique**: |distinct word types| / |total words|. High values → model isn't collapsing to a few tokens.
- **%valid**: fraction of word tokens in an English dictionary. Uses `nltk.corpus.words` (preferred), falls back to `/usr/share/dict/words`, falls back to a tiny baked-in list with a warning.

Unlike OWT entropy, these metrics are **tokenizer-dependent by design** — that's the point of running Text8 as the char-level anchor. Compare within a tokenizer across models, not across tokenizers.

#### Judge perplexity (`JudgePerplexity`)

A utility class that wraps a fixed judge LM and computes PPL on any generated sequence by the decode-and-re-encode protocol. This is the single most important cross-tokenizer comparison:

```python
judge = JudgePerplexity(judge_id="gpt2-large", device="cuda")
ppl = judge.perplexity_from_token_ids(generations, source_tokenizer_name="qwen25")
```

The judge is loaded once and reused across all (model, τ, NFE) cells — one-time cost of ~3s.

**Gotcha**: if the generating model's tokenizer can encode characters the judge cannot, the re-encode will truncate or emit unknown bytes. GPT-2 Large is fine for English + Latin-1, not for Devanagari — use `ai4bharat/IndicBART` or `sarvamai/sarvam-1` as the judge for Hindi.

### 6.2 — Diagnostic Metrics

#### Per-position entropy (`metrics/diagnostic.py::per_position_entropy`)

For each noise level *t* ∈ {0.0, 0.25, 0.5, 0.75, 0.9}:
1. Sample a Bernoulli mask with probability *t* over positions.
2. Run the model to get logits at every position.
3. Compute softmax entropy *H(x_i | context_i)* at every position *i*.
4. Average across 32 real sequences.

Returns a `[seq_length]` vector per noise level, with standard deviation across batches.

**What to look for**:
- **Edge collapse**: entropy spikes at positions 0 and L-1 → PE miscalibration at sequence boundaries, typical under ALiBi with attention windows.
- **Uniform collapse at high t**: entropy near log(|V|) everywhere → model has no signal to condition on, temporal dissonance in action.
- **Stable across positions**: what you want.

#### Attention diffuseness (`metrics/diagnostic.py::attention_diffuseness`)

For each attention head, compute the mean entropy of its attention distribution across query positions. High → diffuse / uniform (information soup); low → sharp / localized.

Motivated by the DroPE / Kazemnejad finding that NoPE transformers have bounded attention-gradient norms at init, which manifests as persistently high per-head entropy. **Prediction**: NoPE CANDI runs will show higher mean entropy across layers than RoPE CANDI.

Returns `per_layer_mean_entropy` and `per_layer_per_head` matrices. Plot as a heatmap (layer × head) to see if one or two heads are carrying positional information.

Requires the sampler's underlying model to support `output_attentions=True` — works for MDLM and LLaDA (HF-backed), needs a small patch for SEDD (the `DDiT` backbone supports it with a config flag).

### 6.3 — Length Generalization (`metrics/length_gen.py`)

For each model trained at L_train and each test length L_test ∈ {1024, 2048, 4096}:

1. Generate N sequences at length L_test using NFE=32 and a sweep over τ ∈ {0.8, 0.9, 1.0}.
2. Compute judge PPL at each (L_test, τ) pair.
3. Take the geometric mean across τ → `per_length_ppl[L_test]`.
4. Report `per_length_ratio[L_test] = per_length_ppl[L_test] / per_length_ppl[L_train]`.

Values near 1 → good length generalization. Values much > 1 → PE-bottlenecked.

Geometric rather than arithmetic mean across τ because PPL is log-normal; geometric mean is the right aggregator for multiplicative quantities.

---

## Expected Computational Cost

On one A100 80GB:

| Command | Wall clock |
|---|---|
| `fertility` (5 tokenizers × 3 corpora) | 10–25 min (tokenizer loading dominates for Gemma) |
| `corruption` (72 cells × 20K MC samples) | 5–15 min (GPU) |
| `frontier` for MDLM-OWT (1 model, 44 τ cells) | ~2 hours |
| `frontier` for LLaDA-8B (1 model, 44 τ cells) | ~8 hours (8B model, sequential sampling) |
| `diagnostics` | ~20 min per model |
| `length-gen` (4 models × 9 cells) | ~6 hours |

Parallelize across (model, NFE) via independent invocations — the YAML config lets you filter down to one model by editing the `models` list.

---

## Integrating LLaDA / SEDD / CANDI

The adapters in `src/samplers.py` have clearly-marked `TODO: INTEGRATE` blocks. Each needs roughly:

- **LLaDA**: already working via AutoModel. If your checkpoint's `mask_token_id` differs from 126336, set it on `LLaDASampler.__init__`.
- **SEDD**: import your already-working `get_pc_sampler` from the Score-Entropy repo and assign to `self._sampling_fn`. Contract: `self._sampling_fn(batch_size, steps, temperature) -> LongTensor[batch, seq_length]`.
- **CANDI**: clone `patrickpynadath1/candi-lander`, import the hybrid sampler, wire `_step` to the repo's reverse-ODE + masked-ancestral update (eqs. 14–15 of the paper).

The rest of the harness is agnostic to model internals — once `sample` and `logits_at` return the right shapes, all metrics light up.

---

## Output Layout

After a full run:

```
{out_dir}/
├── fertility/
│   ├── fertility.json                    # all stats, machine-readable
│   └── fertility_report.md               # human-readable table + examples
├── corruption/
│   └── corruption_estimates.json         # (σ, |V|) → ρ_mc, ρ_theory, r_mc, r_theory
├── frontier/
│   ├── mdlm-owt-110m_owt_frontier.json
│   ├── llada-8b_owt_frontier.json
│   ├── ...
│   └── <model>_text8_frontier.json
├── diagnostics/
│   ├── <model>_per_position_entropy.json
│   └── <model>_attention.json
└── length_gen/
    └── <model>_length_gen.json
```

All JSON; plots not included — the raw results are small enough (< 10MB total) that you can load them in a notebook and plot with matplotlib / plotly locally. A `scripts/plot.py` template is in the repo.

---

## Hypotheses This Harness Tests

Copied from the ablation plan for convenience:

1. **H1** — Pure continuous one-hot diffusion's frontier collapses as |V| crosses some threshold in [10³, 10⁴]. Test: corruption + frontier subcommands at vocab sizes 5 / 256 / 50K / 256K.
2. **H2** — CANDI's frontier tracks or dominates MDLM at all |V|, including Gemma-256K. Test: same as H1 but extended up to Indic / Gemma scale.
3. **H3** — NoPE underperforms RoPE for all diffusion methods; gap largest for continuous / hybrid. Test: retrain + run length-gen + diagnostics.
4. **H4** — Length-gen ratios near 1 only for RoPE(500K); NoPE grows sharply for continuous / hybrid. Test: length-gen subcommand across PE variants.
5. **H5** — Hybrid's advantage over masked is larger on Hindi than English at matched NFE. Test: fertility (Hindi) + frontier (native + Hindi judge).
