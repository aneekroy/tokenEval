"""
Orchestrator CLI.

Four entry points:

    python -m src.run fertility    --config configs/default.yaml
    python -m src.run corruption   --config configs/default.yaml
    python -m src.run frontier     --config configs/default.yaml
    python -m src.run length-gen   --config configs/default.yaml

Each reads a YAML config (see configs/default.yaml) and writes JSON + Markdown
to the configured output directory. Logs to W&B in offline mode if `wandb` is
installed and `use_wandb: true` in the config.

Typical workflow for a new server:
    1. `python -m src.run fertility`       # quick tokenizer sanity check
    2. `python -m src.run corruption`      # verify ρ/r estimators match theory
    3. `python -m src.run frontier ...`    # diffusion-model evaluation (slow)
    4. `python -m src.run length-gen ...`  # length-generalization sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .tokenizers_bench import run_fertility_suite
from .data import owt_factory, text8_factory, sangraha_hindi_factory
from .corruption import sweep_corruption
from .samplers import SamplerConfig, build_sampler
from .metrics import (
    entropy_perplexity_frontier,
    text8_word_frontier,
    JudgePerplexity,
    save_curves,
    per_position_entropy,
    attention_diffuseness,
    length_generalization,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Allow env var expansion in paths
    def expand(v):
        if isinstance(v, str):
            return os.path.expandvars(os.path.expanduser(v))
        if isinstance(v, dict):
            return {k: expand(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v
    return expand(cfg)


# ---------------------------------------------------------------------------
# Subcommand: fertility
# ---------------------------------------------------------------------------
def cmd_fertility(cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"]) / "fertility"
    tokenizer_names = cfg["fertility"]["tokenizers"]

    corpora = {}
    data_cfg = cfg.get("data", {})
    if data_cfg.get("owt", {}).get("enabled", True):
        docs = list(owt_factory(
            n_docs=data_cfg["owt"].get("n_docs", 2000),
            hf_cache=data_cfg.get("hf_cache"),
            keep_fraction=data_cfg["owt"].get("keep_fraction", 1.0),
        )())
        if docs:
            corpora["owt"] = docs
        else:
            print("[warn] OWT returned 0 docs; skipping from fertility suite")
    if data_cfg.get("text8", {}).get("enabled", True):
        docs = list(text8_factory(
            path=data_cfg["text8"].get("path"),
            n_docs=data_cfg["text8"].get("n_docs", 2000),
        )())
        if docs:
            corpora["text8"] = docs
        else:
            print("[warn] Text8 returned 0 docs; skipping from fertility suite")
    if data_cfg.get("hindi", {}).get("enabled", False):
        docs = list(sangraha_hindi_factory(
            n_docs=data_cfg["hindi"].get("n_docs", 1000),
            hf_cache=data_cfg.get("hf_cache"),
        )())
        if docs:
            corpora["hindi"] = docs
        else:
            print("[warn] Sangraha Hindi returned 0 docs; skipping from fertility suite")

    if not corpora:
        print("[error] no corpora available; nothing to do")
        return

    results = run_fertility_suite(
        corpora=corpora,
        tokenizer_names=tokenizer_names,
        out_dir=out_dir,
        hf_token=cfg.get("hf_token"),
    )
    print(f"\n[done] Fertility results written to {out_dir}")


# ---------------------------------------------------------------------------
# Subcommand: corruption
# ---------------------------------------------------------------------------
def cmd_corruption(cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"]) / "corruption"
    out_dir.mkdir(parents=True, exist_ok=True)

    cc = cfg["corruption"]
    sigmas = cc["sigmas"]
    vocab_sizes = cc["vocab_sizes"]
    n_samples = cc.get("n_samples", 10_000)

    print(f"[corruption] sweeping σ × |V|  = {len(sigmas)} × {len(vocab_sizes)}")
    results = sweep_corruption(
        sigmas=sigmas,
        vocab_sizes=vocab_sizes,
        n_samples=n_samples,
    )
    dump = [asdict(r) for r in results]
    with open(out_dir / "corruption_estimates.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[done] Corruption estimates in {out_dir / 'corruption_estimates.json'}")

    # Quick text summary
    print("\n| |V| |   σ   | ρ_mc  | ρ_th  | r_mc  | r_th  |")
    print("|----:|------:|------:|------:|------:|------:|")
    for r in results:
        rho_th = f"{r.rho_theory:.3f}" if r.rho_theory == r.rho_theory else "   nan"
        print(
            f"| {r.vocab_size:>4d} | {r.sigma:>5.2f} | "
            f"{r.rho_mc:.3f} | {rho_th} | "
            f"{r.r_mc:.3f} | {r.r_theory:.3f} |"
        )


# ---------------------------------------------------------------------------
# Subcommand: frontier
# ---------------------------------------------------------------------------
def cmd_frontier(cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"]) / "frontier"
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = cfg["frontier"]
    judge = None
    if fc.get("use_judge", True):
        judge = JudgePerplexity(
            judge_id=fc.get("judge_id", "gpt2-large"),
            device=fc.get("device", "cuda"),
        )

    for model_spec in fc["models"]:
        print(f"\n=== Frontier: {model_spec['name']} ===")
        sampler_cfg = SamplerConfig(
            model_name=model_spec["name"],
            checkpoint_path=model_spec["checkpoint"],
            tokenizer_name=model_spec["tokenizer"],
            device=fc.get("device", "cuda"),
            dtype=fc.get("dtype", "bfloat16"),
        )
        try:
            sampler = build_sampler(model_spec["kind"], sampler_cfg)
        except NotImplementedError as e:
            print(f"[skip] {model_spec['name']}: {e}")
            continue

        # OWT frontier
        if fc.get("owt", {}).get("enabled", True):
            curves = entropy_perplexity_frontier(
                sampler,
                nfe_values=fc["owt"]["nfe_values"],
                temperatures=fc["owt"]["temperatures"],
                seq_length=fc["owt"].get("seq_length", 1024),
                n_sequences=fc["owt"].get("n_sequences", 64),
                seed=fc.get("seed", 0),
                judge_sampler=judge,
            )
            save_curves(
                curves,
                out_dir / f"{model_spec['name']}_owt_frontier.json",
            )

        # Text8 word-frontier
        if fc.get("text8", {}).get("enabled", False):
            curves = text8_word_frontier(
                sampler,
                nfe_values=fc["text8"]["nfe_values"],
                temperatures=fc["text8"]["temperatures"],
                source_tokenizer=sampler.tokenizer,
                seq_length=fc["text8"].get("seq_length", 1024),
                n_sequences=fc["text8"].get("n_sequences", 64),
                seed=fc.get("seed", 0),
            )
            save_curves(
                curves,
                out_dir / f"{model_spec['name']}_text8_frontier.json",
            )


# ---------------------------------------------------------------------------
# Subcommand: diagnostics
# ---------------------------------------------------------------------------
def cmd_diagnostics(cfg: dict) -> None:
    from .data import iter_token_sequences, owt_factory
    out_dir = Path(cfg["out_dir"]) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    dc = cfg["diagnostics"]
    for model_spec in dc["models"]:
        print(f"\n=== Diagnostics: {model_spec['name']} ===")
        sampler_cfg = SamplerConfig(
            model_name=model_spec["name"],
            checkpoint_path=model_spec["checkpoint"],
            tokenizer_name=model_spec["tokenizer"],
            device=dc.get("device", "cuda"),
            dtype=dc.get("dtype", "bfloat16"),
        )
        try:
            sampler = build_sampler(model_spec["kind"], sampler_cfg)
        except NotImplementedError as e:
            print(f"[skip] {model_spec['name']}: {e}")
            continue

        # Condition data: take N real OWT sequences tokenized with the model's tokenizer
        seq_len = dc.get("seq_length", 512)
        n_seq = dc.get("n_sequences", 32)
        docs = list(iter_token_sequences(
            owt_factory(n_docs=2000),
            tokenizer_name=model_spec["tokenizer"],
            seq_length=seq_len,
            max_sequences=n_seq,
        ))

        # 1. Per-position entropy
        ppe = per_position_entropy(
            sampler,
            token_sequences=docs,
            noise_levels=dc.get("noise_levels", [0.0, 0.25, 0.5, 0.75]),
        )
        with open(out_dir / f"{model_spec['name']}_per_position_entropy.json", "w") as f:
            json.dump([p.to_dict() for p in ppe], f, indent=2)

        # 2. Attention diffuseness
        if dc.get("attention", True):
            ad = attention_diffuseness(sampler, token_sequences=docs[:8])
            if ad is not None:
                with open(out_dir / f"{model_spec['name']}_attention.json", "w") as f:
                    json.dump(ad.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Subcommand: length-gen
# ---------------------------------------------------------------------------
def cmd_length_gen(cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"]) / "length_gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    lg = cfg["length_gen"]
    judge = JudgePerplexity(
        judge_id=lg.get("judge_id", "gpt2-large"),
        device=lg.get("device", "cuda"),
    )

    for model_spec in lg["models"]:
        print(f"\n=== Length gen: {model_spec['name']} ({model_spec.get('pe', '?')}) ===")
        sampler_cfg = SamplerConfig(
            model_name=model_spec["name"],
            checkpoint_path=model_spec["checkpoint"],
            tokenizer_name=model_spec["tokenizer"],
            device=lg.get("device", "cuda"),
            dtype=lg.get("dtype", "bfloat16"),
        )
        try:
            sampler = build_sampler(model_spec["kind"], sampler_cfg)
        except NotImplementedError as e:
            print(f"[skip] {model_spec['name']}: {e}")
            continue

        result = length_generalization(
            sampler,
            l_train=lg["l_train"],
            l_test_values=lg["l_test"],
            temperatures=lg["temperatures"],
            nfe=lg["nfe"],
            pe_type=model_spec.get("pe", "unknown"),
            judge=judge,
            n_sequences_per_cell=lg.get("n_sequences", 32),
            seed=lg.get("seed", 0),
        )
        with open(out_dir / f"{model_spec['name']}_length_gen.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(
            f"  {model_spec['name']}: ratios = "
            + ", ".join(f"L{L}:{v:.2f}" for L, v in result.per_length_ratio.items())
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
COMMANDS = {
    "fertility": cmd_fertility,
    "corruption": cmd_corruption,
    "frontier": cmd_frontier,
    "diagnostics": cmd_diagnostics,
    "length-gen": cmd_length_gen,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    COMMANDS[args.command](cfg)


if __name__ == "__main__":
    main()