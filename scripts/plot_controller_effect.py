#!/usr/bin/env python3
"""Plot the paper's archived RAG-controller effects; no new experiment is run."""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen3.5:" + size for size in ("0.8b", "2b", "4b", "9b", "27b")]
DATASETS = [("hotpotqa", "HotpotQA"), ("2wiki", "2WikiMultihopQA"), ("musique", "MuSiQue")]


def main():
    with (ROOT / "paper/analysis/bootstrap_ci.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["effect"] == "controller_rag"]
    effects = {(r["dataset"], r["model"]): r for r in rows}
    with (ROOT / "data/runs/main-20260930/summary.csv").open(encoding="utf-8", newline="") as handle:
        scores = {(r["dataset"], r["model"], r["condition"]): float(r["em"]) for r in csv.DictReader(handle)}
    expected = {(ds, model) for ds, _ in DATASETS for model in MODELS}
    assert len(rows) == len(effects) == 15 and set(effects) == expected, "Incomplete or duplicated effect rows"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharex=True, sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS):
        ax.axvspan(-25, 0, color="#f9f3ed", zorder=0)
        ax.axvspan(0, 18, color="#eef6fa", zorder=0)
        ax.axvline(0, color="#707d88", linewidth=1, zorder=1)
        for y, model in enumerate(MODELS):
            row = effects[dataset, model]
            estimate, low, high = (float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
            assert all(math.isfinite(v) for v in (estimate, low, high)) and low <= estimate <= high
            delta = scores[dataset, model, "rag_laya"] - scores[dataset, model, "rag_llm"]
            assert math.isclose(estimate, delta, abs_tol=1e-9), "Effect does not match original scores"
            color = "#226485" if estimate > 0 else "#99603c"
            excludes_zero = low > 0 or high < 0
            ax.errorbar(estimate, y, xerr=[[estimate-low], [high-estimate]], fmt="o", color=color,
                        markersize=7, capsize=4, linewidth=1.8, markeredgewidth=1.6,
                        markerfacecolor=color if excludes_zero else "white", zorder=3)
            ax.annotate(f"{estimate:+.1f}", (estimate, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", color=color, fontsize=10, weight="bold")
        ax.set_title(title, pad=22, fontsize=12)
        ax.set_xlim(-24, 18)
        ax.set_xticks([-20, -10, 0, 10])
        ax.set_yticks(range(5), [m.split(":")[1].upper() for m in MODELS])
        ax.set_ylim(4.5, -0.6)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.grid(axis="y", color="white", linewidth=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Qwen3.5 model size", labelpad=12)
    fig.suptitle("External control changes accuracy differently across model sizes", y=0.99, fontsize=15, weight="bold")
    fig.supxlabel("EM difference: RAG + Laya minus RAG + LLM (percentage points)", y=0.115, fontsize=11)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#475569", label="95% CI excludes zero"),
        Line2D([], [], marker="o", linestyle="", color="#475569", markerfacecolor="white", label="95% CI includes zero"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.24, wspace=0.12)
    output = ROOT / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        options = {"dpi": 180} if suffix == "png" else {"metadata": {"CreationDate": None}}
        fig.savefig(output / f"controller_effect_by_scale.{suffix}", **options)
    plt.close(fig)
    print("Verified and plotted all 15 RAG-controller effects and paired 95% confidence intervals.")


if __name__ == "__main__":
    main()
