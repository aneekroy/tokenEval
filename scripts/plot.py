"""
Plot helper for diffusion_tokenizer_bench outputs.

Reads the JSON dumps written by src/run.py and produces figures matching
the CANDI paper's style (entropy-PPL frontier, corruption divergence, etc).
Each function is independent — you can call one or all.

Usage:
    python scripts/plot.py --out-dir /path/to/run --plots all

Figures are saved as PNG into {out_dir}/figures/. Matplotlib is the only
dependency beyond stdlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
# Consistent colour palette across plots so the same method is the same colour
# in every figure. Maps method-name-prefix → colour.
METHOD_COLORS = {
    "mdlm":  "#d62728",   # red
    "llada": "#1f77b4",   # blue
    "sedd":  "#9467bd",   # purple
    "candi": "#2ca02c",   # green
    "cont":  "#e377c2",   # pink (continuous one-hot baseline)
}

NFE_MARKERS = {8: "o", 16: "s", 32: "^", 64: "D", 128: "v"}


def _color_for(method: str) -> str:
    for prefix, c in METHOD_COLORS.items():
        if method.startswith(prefix):
            return c
    return "#555555"


def _ensure_fig_dir(out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ---------------------------------------------------------------------------
# Frontier: entropy-perplexity (OWT)
# ---------------------------------------------------------------------------
def plot_owt_frontiers(out_dir: Path) -> None:
    """
    Reproduce CANDI Fig. 7: entropy-perplexity frontier, one subplot per NFE,
    one curve per method. Temperature varies along each curve.
    """
    frontier_dir = out_dir / "frontier"
    curve_files = sorted(frontier_dir.glob("*_owt_frontier.json"))
    if not curve_files:
        print(f"[plot] no OWT frontier files in {frontier_dir}; skipping")
        return

    # Gather: dict[nfe] -> list of (method, tau_sorted, H_sorted, ppl_sorted, judge_sorted)
    by_nfe: dict[int, list[dict]] = {}
    for path in curve_files:
        curves = json.loads(path.read_text())
        for curve in curves:
            nfe = curve["nfe"]
            pts = curve["points"]
            pts = [p for p in pts if p.get("entropy") is not None]
            # Sort by temperature so the line traces monotonically
            pts.sort(key=lambda p: p["temperature"])
            by_nfe.setdefault(nfe, []).append({
                "method": curve["method"],
                "H": [p["entropy"] for p in pts],
                "ppl": [p.get("gen_perplexity") for p in pts],
                "judge_ppl": [p.get("judge_perplexity") for p in pts],
                "tau": [p["temperature"] for p in pts],
            })

    n_nfe = len(by_nfe)
    fig, axes = plt.subplots(1, n_nfe, figsize=(4.2 * n_nfe, 4.0), sharey=True)
    if n_nfe == 1:
        axes = [axes]
    for ax, (nfe, series) in zip(axes, sorted(by_nfe.items())):
        for s in series:
            ppl_for_y = [p for p in s["ppl"] if p is not None]
            if len(ppl_for_y) == 0:
                continue
            ax.plot(
                s["H"], s["ppl"],
                marker=NFE_MARKERS.get(nfe, "o"),
                color=_color_for(s["method"]),
                label=s["method"],
                linewidth=1.5, markersize=5,
            )
        ax.set_title(f"NFE = {nfe}")
        ax.set_xlabel("Sample entropy (nats)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Generative perplexity (log scale)")
    # Dedupe legend
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(),
               loc="upper center", ncol=min(len(uniq), 5),
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    fig_dir = _ensure_fig_dir(out_dir)
    out_path = fig_dir / "owt_frontier_gen_ppl.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

    # Judge-PPL variant
    if any(any(x is not None for x in s["judge_ppl"]) for series in by_nfe.values() for s in series):
        fig, axes = plt.subplots(1, n_nfe, figsize=(4.2 * n_nfe, 4.0), sharey=True)
        if n_nfe == 1:
            axes = [axes]
        for ax, (nfe, series) in zip(axes, sorted(by_nfe.items())):
            for s in series:
                valid = [(h, p) for h, p in zip(s["H"], s["judge_ppl"]) if p is not None]
                if not valid:
                    continue
                xs, ys = zip(*valid)
                ax.plot(xs, ys,
                        marker=NFE_MARKERS.get(nfe, "o"),
                        color=_color_for(s["method"]),
                        label=s["method"], linewidth=1.5, markersize=5)
            ax.set_title(f"NFE = {nfe}")
            ax.set_xlabel("Sample entropy (nats)")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Judge perplexity (GPT-2 Large)")
        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(),
                   loc="upper center", ncol=min(len(uniq), 5),
                   bbox_to_anchor=(0.5, 1.04), frameon=False)
        fig.tight_layout()
        out_path = fig_dir / "owt_frontier_judge_ppl.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Frontier: %unique / %valid (Text8)
# ---------------------------------------------------------------------------
def plot_text8_frontiers(out_dir: Path) -> None:
    """CANDI Fig. 6 style: %unique on x-axis, %valid on y-axis, per NFE."""
    frontier_dir = out_dir / "frontier"
    files = sorted(frontier_dir.glob("*_text8_frontier.json"))
    if not files:
        print(f"[plot] no Text8 frontier files in {frontier_dir}; skipping")
        return

    by_nfe: dict[int, list[dict]] = {}
    for path in files:
        curves = json.loads(path.read_text())
        for curve in curves:
            nfe = curve["nfe"]
            pts = [p for p in curve["points"]
                   if p.get("pct_unique_words") is not None]
            pts.sort(key=lambda p: p["temperature"])
            by_nfe.setdefault(nfe, []).append({
                "method": curve["method"],
                "unique": [p["pct_unique_words"] for p in pts],
                "valid":  [p["pct_valid_words"] for p in pts],
            })

    n = len(by_nfe)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (nfe, series) in zip(axes, sorted(by_nfe.items())):
        for s in series:
            ax.plot(s["unique"], s["valid"],
                    marker=NFE_MARKERS.get(nfe, "o"),
                    color=_color_for(s["method"]),
                    label=s["method"], linewidth=1.5, markersize=5)
        ax.set_title(f"NFE = {nfe}")
        ax.set_xlabel("% unique words")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% valid words")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(),
               loc="upper center", ncol=min(len(uniq), 5),
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    fig_dir = _ensure_fig_dir(out_dir)
    out_path = fig_dir / "text8_frontier.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Corruption: ρ(σ) and r(σ) vs |V|
# ---------------------------------------------------------------------------
def plot_corruption(out_dir: Path) -> None:
    """
    Reproduce CANDI Fig. 3: ρ(σ) and r(σ) vs σ, one subplot per vocab size.
    ρ scales with |V|, r does not — that divergence is the headline.
    """
    path = out_dir / "corruption" / "corruption_estimates.json"
    if not path.exists():
        print(f"[plot] no {path}; skipping")
        return
    data = json.loads(path.read_text())

    # Group by vocab
    by_V: dict[int, list[dict]] = {}
    for row in data:
        by_V.setdefault(row["vocab_size"], []).append(row)
    for V in by_V:
        by_V[V].sort(key=lambda r: r["sigma"])

    V_sorted = sorted(by_V)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(V_sorted)))

    # Figure 1: ρ(σ) across |V| (shows the |V|-dependence)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for V, color in zip(V_sorted, cmap):
        rows = by_V[V]
        sigmas = [r["sigma"] for r in rows]
        rho_mc = [r["rho_mc"] for r in rows]
        rho_th = [r["rho_theory"] for r in rows]
        ax.plot(sigmas, rho_mc, color=color, marker="o",
                label=f"|V|={V:,}", linewidth=1.6, markersize=4)
        # Overlay theory as dashed where available
        rho_th_valid = [(s, t) for s, t in zip(sigmas, rho_th)
                        if t == t and t is not None]
        if rho_th_valid:
            xs, ys = zip(*rho_th_valid)
            ax.plot(xs, ys, color=color, linestyle="--", alpha=0.5,
                    linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("ρ(σ)  —  discrete identity corruption")
    ax.set_title("ρ(σ) vs |V|  (solid: MC, dashed: analytic)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig_dir = _ensure_fig_dir(out_dir)
    out_path = fig_dir / "corruption_rho.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

    # Figure 2: r(σ) — should be |V|-independent
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for V, color in zip(V_sorted, cmap):
        rows = by_V[V]
        sigmas = [r["sigma"] for r in rows]
        r_mc = [r["r_mc"] for r in rows]
        ax.plot(sigmas, r_mc, color=color, marker="o",
                label=f"|V|={V:,}", linewidth=1.6, markersize=4, alpha=0.8)
    # Theory overlay — pick any row's r_theory since it's V-independent
    r_th_sigmas = [r["sigma"] for r in by_V[V_sorted[0]]]
    r_th_vals = [r["r_theory"] for r in by_V[V_sorted[0]]]
    ax.plot(r_th_sigmas, r_th_vals, color="black", linestyle="--",
            label="Φ(−1/(σ√2))", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("r(σ)  —  continuous rank degradation")
    ax.set_title("r(σ) is |V|-independent (CANDI Eq. 6)")
    ax.set_ylim(-0.02, 0.52)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = fig_dir / "corruption_r.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

    # Figure 3: the divergence — ρ vs r parametric plot at fixed σ sweep
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for V, color in zip(V_sorted, cmap):
        rows = by_V[V]
        rho = [r["rho_mc"] for r in rows]
        rv = [r["r_mc"] for r in rows]
        ax.plot(rho, rv, color=color, marker="o",
                label=f"|V|={V:,}", linewidth=1.6, markersize=4)
    ax.set_xlabel("ρ(σ)  —  discrete corruption")
    ax.set_ylabel("r(σ)  —  continuous degradation")
    ax.set_title("Temporal dissonance: the gap between ρ and r widens with |V|")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.52)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = fig_dir / "corruption_rho_vs_r.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Per-position entropy
# ---------------------------------------------------------------------------
def plot_per_position_entropy(out_dir: Path) -> None:
    diag_dir = out_dir / "diagnostics"
    files = sorted(diag_dir.glob("*_per_position_entropy.json"))
    if not files:
        print(f"[plot] no per-position entropy files; skipping")
        return
    for path in files:
        data = json.loads(path.read_text())
        if not data:
            continue
        method = data[0]["method"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for entry in data:
            mean = np.array(entry["mean_per_position"])
            std = np.array(entry["std_per_position"])
            xs = np.arange(len(mean))
            ax.plot(xs, mean, label=f"t={entry['noise_level']:.2f}")
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("H(x_i | context_i)  (nats)")
        ax.set_title(f"Per-position entropy — {method}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig_dir = _ensure_fig_dir(out_dir)
        out_path = fig_dir / f"per_position_entropy_{method}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Attention diffuseness heatmap
# ---------------------------------------------------------------------------
def plot_attention(out_dir: Path) -> None:
    diag_dir = out_dir / "diagnostics"
    files = sorted(diag_dir.glob("*_attention.json"))
    if not files:
        print(f"[plot] no attention files; skipping")
        return
    for path in files:
        data = json.loads(path.read_text())
        method = data["method"]
        per_head = np.array(data["per_layer_per_head"])   # [L, H]
        H_max = data["max_entropy"]
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            per_head.T / H_max, aspect="auto",
            cmap="magma", vmin=0, vmax=1,
            origin="lower",
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        ax.set_title(
            f"Attention entropy / log(L) — {method}\n"
            f"(0 = sharp, 1 = uniform)"
        )
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig_dir = _ensure_fig_dir(out_dir)
        out_path = fig_dir / f"attention_{method}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Length-generalization
# ---------------------------------------------------------------------------
def plot_length_gen(out_dir: Path) -> None:
    lg_dir = out_dir / "length_gen"
    files = sorted(lg_dir.glob("*_length_gen.json"))
    if not files:
        print(f"[plot] no length-gen files; skipping")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path in files:
        data = json.loads(path.read_text())
        ratios = data["per_length_ratio"]
        Ls = sorted(int(k) for k in ratios)
        ys = [ratios[str(L)] for L in Ls]
        method = data["method"]
        pe = data.get("pe_type", "?")
        ls = "--" if pe == "nope" else "-"
        ax.plot(Ls, ys, marker="o", linestyle=ls,
                color=_color_for(method),
                label=f"{method} ({pe})", linewidth=1.5)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Test sequence length")
    ax.set_ylabel("Judge-PPL(L_test) / Judge-PPL(L_train)")
    ax.set_title("Length generalization — ratio near 1 = good")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig_dir = _ensure_fig_dir(out_dir)
    out_path = fig_dir / "length_gen.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Fertility bar-chart summary
# ---------------------------------------------------------------------------
def plot_fertility(out_dir: Path) -> None:
    path = out_dir / "fertility" / "fertility.json"
    if not path.exists():
        print(f"[plot] no {path}; skipping")
        return
    data = json.loads(path.read_text())
    tokenizers = sorted(data)
    corpora = sorted({c for v in data.values() for c in v})

    # Tokens/word grouped bar chart, one group per corpus
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(tokenizers) * len(corpora)), 4.5))
    width = 0.8 / max(len(corpora), 1)
    xs = np.arange(len(tokenizers))
    for i, corpus in enumerate(corpora):
        ys = [data[t].get(corpus, {}).get("tokens_per_word_mean", 0)
              for t in tokenizers]
        ax.bar(xs + i * width, ys, width=width, label=corpus)
    ax.set_xticks(xs + width * (len(corpora) - 1) / 2)
    ax.set_xticklabels(tokenizers, rotation=15)
    ax.set_ylabel("Tokens / word (mean)")
    ax.set_title("Fertility by tokenizer × corpus")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig_dir = _ensure_fig_dir(out_dir)
    out_path = fig_dir / "fertility_tokens_per_word.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_PLOTS = {
    "fertility": plot_fertility,
    "corruption": plot_corruption,
    "owt": plot_owt_frontiers,
    "text8": plot_text8_frontiers,
    "per_position": plot_per_position_entropy,
    "attention": plot_attention,
    "length_gen": plot_length_gen,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="Root directory passed to src/run.py")
    ap.add_argument("--plots", nargs="+", default=["all"],
                    choices=["all"] + list(ALL_PLOTS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    names = list(ALL_PLOTS) if "all" in args.plots else args.plots
    for name in names:
        print(f"\n--- {name} ---")
        try:
            ALL_PLOTS[name](out_dir)
        except Exception as e:
            print(f"[plot:{name}] failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
