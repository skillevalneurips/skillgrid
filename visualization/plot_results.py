#!/usr/bin/env python3
"""Generate publication-quality plots from experiment results.

Usage:
    python visualization/plot_results.py --results outputs/results.json --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_success_by_origin_policy(results: list[dict], output_dir: Path) -> None:
    """Heatmap: skill_origin x runtime_policy -> success_rate."""
    origins = sorted(set(r["skill_origin"] for r in results))
    policies = sorted(set(r["runtime_policy"] for r in results))

    matrix = np.zeros((len(origins), len(policies)))
    counts = np.zeros_like(matrix)

    for r in results:
        i = origins.index(r["skill_origin"])
        j = policies.index(r["runtime_policy"])
        matrix[i, j] += r["success_rate"]
        counts[i, j] += 1

    mask = counts > 0
    matrix[mask] /= counts[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies)
    ax.set_yticks(range(len(origins)))
    ax.set_yticklabels(origins)
    ax.set_xlabel("Runtime Policy")
    ax.set_ylabel("Skill Origin")
    ax.set_title("Success Rate: Skill Origin x Runtime Policy")

    for i in range(len(origins)):
        for j in range(len(policies)):
            text = f"{matrix[i, j]:.1%}" if counts[i, j] > 0 else "N/A"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color="white" if matrix[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Success Rate")
    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_origin_policy.pdf", dpi=150)
    fig.savefig(output_dir / "heatmap_origin_policy.png", dpi=150)
    plt.close(fig)


def plot_per_pattern_bars(results: list[dict], output_dir: Path) -> None:
    """Grouped bar chart: success rate per composition pattern (SL/PO/FP)."""
    experiments = []
    for r in results:
        if r.get("per_pattern"):
            experiments.append(r)

    if not experiments:
        return

    patterns = ["SL", "PO", "FP"]
    labels = [f"{r['dataset_id']}/{r['skill_origin']}" for r in experiments]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    for i, p in enumerate(patterns):
        values = [r["per_pattern"].get(p, 0) for r in experiments]
        ax.bar(x + i * width, values, width, label=p)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Composition Pattern")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(title="Pattern")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "bars_per_pattern.pdf", dpi=150)
    fig.savefig(output_dir / "bars_per_pattern.png", dpi=150)
    plt.close(fig)


def plot_cost_vs_success(results: list[dict], output_dir: Path) -> None:
    """Scatter: avg_cost vs success_rate, colored by model."""
    models = sorted(set(r["model_id"] for r in results))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    model_color = dict(zip(models, colors))

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        ax.scatter(
            r["avg_cost"], r["success_rate"],
            c=[model_color[r["model_id"]]],
            s=80, alpha=0.7, edgecolors="black", linewidth=0.5,
        )

    for model, color in model_color.items():
        ax.scatter([], [], c=[color], label=model, s=80, edgecolors="black")

    ax.set_xlabel("Average Cost per Episode ($)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Cost-Effectiveness of Skill Configurations")
    ax.legend(title="Model")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "scatter_cost_success.pdf", dpi=150)
    fig.savefig(output_dir / "scatter_cost_success.png", dpi=150)
    plt.close(fig)


def plot_recovery_rates(results: list[dict], output_dir: Path) -> None:
    """Bar chart of recovery rates across experiment configurations."""
    labels = [
        f"{r['dataset_id']}/{r['skill_origin']}/{r['runtime_policy']}"
        for r in results
    ]
    rates = [r["recovery_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    colors = plt.cm.RdYlGn([r for r in rates])
    ax.barh(range(len(labels)), rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Recovery Rate")
    ax.set_title("Error Recovery Rate by Configuration")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "recovery_rates.pdf", dpi=150)
    fig.savefig(output_dir / "recovery_rates.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument("--results", type=str, default="outputs/results.json")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)
    print(f"Loaded {len(results)} experiment results.")

    plot_success_by_origin_policy(results, output_dir)
    plot_per_pattern_bars(results, output_dir)
    plot_cost_vs_success(results, output_dir)
    plot_recovery_rates(results, output_dir)

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
