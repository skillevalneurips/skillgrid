"""Write conversational recommendation taxonomy results as LaTeX tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("Reddit v2", "ReDial")
DEFAULT_RESULT_SETS = {
    "GPT-4o-mini": {
        "Reddit v2": Path(
            "datasets/conversational_rec/outputs/"
            "gpt4o_mini_redditv2_td_axis/results.json"
        ),
        "ReDial": Path(
            "datasets/conversational_rec/outputs/"
            "gpt4o_mini_redial_td_axis/results.json"
        ),
    },
    "GPT-5-mini": {
        "Reddit v2": Path(
            "datasets/conversational_rec/outputs/"
            "gpt5_mini_redditv2_td_axis/results.json"
        ),
        "ReDial": Path(
            "datasets/conversational_rec/outputs/"
            "gpt5_mini_redial_td_axis/results.json"
        ),
    },
    "Qwen3-4B": {
        "Reddit v2": Path(
            "datasets/conversational_rec/outputs/"
            "qwen3_4b_redditv2_td_axis/results.json"
        ),
        "ReDial": Path(
            "datasets/conversational_rec/outputs/"
            "qwen3_4b_redial_td_axis/results.json"
        ),
    },
    "Qwen2.5-7B": {
        "Reddit v2": Path(
            "datasets/conversational_rec/outputs/"
            "qwen25_7b_redditv2_td_axis/results.json"
        ),
        "ReDial": Path(
            "datasets/conversational_rec/outputs/"
            "qwen25_7b_redial_td_axis/results.json"
        ),
    },
}
DEFAULT_OUTPUT = Path(
    "69734baaf1ddea8fb4f4f3f4/figures/rec_results.tex"
)


@dataclass(frozen=True)
class Cell:
    visibility: str
    retrieval: str
    evolution: str


AXIS_SPECS = [
    {
        "key": "visibility",
        "title": "Skill Visibility",
        "setting": "Visibility",
        "caption": (
            "Recommendation results along the skill visibility axis. "
            "Retrieval is held at NR and evolution is held at FR."
        ),
        "rows": [
            ("NL", Cell("NL", "NR", "FR"), None),
            ("LB", Cell("LB", "NR", "FR"), None),
            ("FL", Cell("FL", "NR", "FR"), None),
        ],
    },
    {
        "key": "retrieval",
        "title": "Runtime Skill Selection",
        "setting": "Retrieval",
        "caption": (
            "Recommendation results along the runtime skill selection axis. "
            "Visibility is held at FL and evolution is held at FR."
        ),
        "rows": [
            ("NR", Cell("FL", "NR", "FR"), None),
            ("RR", Cell("FL", "RR", "FR"), None),
            ("PR", Cell("FL", "PR", "FR"), None),
        ],
    },
    {
        "key": "evolution",
        "title": "Skill Evolution",
        "setting": "Evolution",
        "caption": (
            "Recommendation results along the skill evolution axis. "
            "Visibility is held at FL and retrieval is held at NR. "
            "$\\Delta$ reports BU minus FR in percentage points."
        ),
        "rows": [
            ("FR", Cell("FL", "NR", "FR"), None),
            ("BU", Cell("FL", "NR", "BU"), None),
            ("$\\Delta$", Cell("FL", "NR", "BU"), Cell("FL", "NR", "FR")),
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table for conversational recommendation results."
    )
    parser.add_argument(
        "--redditv2-results",
        type=Path,
        default=None,
        help=(
            "Backward-compatible single-model Reddit v2 results path. "
            "If provided, use with --redial-results and --model."
        ),
    )
    parser.add_argument(
        "--redial-results",
        type=Path,
        default=None,
        help=(
            "Backward-compatible single-model ReDial results path. "
            "If provided, use with --redditv2-results and --model."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="GPT-4o-mini")
    parser.add_argument("--origin", default="TD")
    parser.add_argument(
        "--table-ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values to include in the table.",
    )
    args = parser.parse_args()
    table_ks = parse_ks(args.table_ks)

    result_sets = default_result_sets()
    if args.redditv2_results is not None or args.redial_results is not None:
        if args.redditv2_results is None or args.redial_results is None:
            parser.error("--redditv2-results and --redial-results must be provided together")
        result_sets = {
            args.model: {
                "Reddit v2": args.redditv2_results,
                "ReDial": args.redial_results,
            }
        }

    results = {
        model: {
            dataset: load_results(path)
            for dataset, path in dataset_paths.items()
        }
        for model, dataset_paths in result_sets.items()
    }
    table = render_table(
        results=results,
        origin=args.origin,
        table_ks=table_ks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


def default_result_sets() -> dict[str, dict[str, Path]]:
    return {
        model: dict(dataset_paths)
        for model, dataset_paths in DEFAULT_RESULT_SETS.items()
    }


def load_results(path: Path) -> dict[Cell, dict[str, Any]]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("results", [])
    else:
        rows = raw

    out: dict[Cell, dict[str, Any]] = {}
    for row in rows:
        metrics = row.get("metadata", {}).get("dataset_metrics", {})
        cell = Cell(
            str(row.get("visibility", "")),
            str(row.get("retrieval", "")),
            str(row.get("evolution", "")),
        )
        by_k: dict[int, dict[str, float]] = {}
        for key, value in metrics.items():
            metric_name = None
            k_text = ""
            for prefix, name in [
                ("mean_hit_at_", "hit"),
                ("mean_recall_at_", "recall"),
                ("mean_ndcg_at_", "ndcg"),
            ]:
                if key.startswith(prefix):
                    metric_name = name
                    k_text = key.removeprefix(prefix)
                    break
            if metric_name is not None and k_text.isdigit():
                by_k.setdefault(int(k_text), {})[metric_name] = float(value)
        out[cell] = {
            "hit": float(metrics.get("mean_hit_at_k", row.get("success_rate", 0.0))),
            "recall": float(metrics.get("mean_recall_at_k", 0.0)),
            "ndcg": float(metrics.get("mean_ndcg_at_k", 0.0)),
            "n": float(metrics.get("gold_score_count", 0.0)),
            "by_k": by_k,
        }
    return out


def render_table(
    *,
    results: dict[str, dict[str, dict[Cell, dict[str, Any]]]],
    origin: str,
    table_ks: list[int],
) -> str:
    return "\n\n".join(
        render_axis_table(
            axis=axis,
            results=results,
            origin=origin,
            table_ks=table_ks,
        )
        for axis in AXIS_SPECS
    ) + "\n"


def render_axis_table(
    *,
    axis: dict[str, Any],
    results: dict[str, dict[str, dict[Cell, dict[str, Any]]]],
    origin: str,
    table_ks: list[int],
) -> str:
    datasets = list(DATASETS)
    cols_per_dataset = 3 * len(table_ks)
    header_cols = "l l " + " ".join(["c" * cols_per_dataset for _ in datasets])
    metric_header = " ".join(
        f"& H@{k} & R@{k} & N@{k}" for k in table_ks
    )
    lines = [
        "% Auto-generated by datasets/conversational_rec/write_rec_results_table.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{\\textbf{{{axis['title']}.}} {axis['caption']} "
        f"Models use trace-derived ({origin}) skills. "
        "H, R, and N denote Hit, Recall, and NDCG; metrics are percentages. "
        "Missing entries indicate incomplete or unavailable runs.}",
        f"\\label{{tab:rec-{axis['key']}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{header_cols}}}",
        "\\toprule",
        f"\\textbf{{Model}} & \\textbf{{{axis['setting']}}} "
        + " ".join(
            f"& \\multicolumn{{{cols_per_dataset}}}{{c}}{{\\textbf{{{escape_tex(name)}}}}}"
            for name in datasets
        )
        + " \\\\",
        " & "
        + " ".join(metric_header for _ in datasets)
        + " \\\\",
        "\\midrule",
    ]

    for model_idx, (model, model_results) in enumerate(results.items()):
        if model_idx > 0:
            lines.append("\\midrule")
        for row_idx, (setting, cell, delta_from) in enumerate(axis["rows"]):
            lines.append(
                render_row(
                    model if row_idx == 0 else "",
                    setting,
                    cell,
                    model_results,
                    table_ks,
                    datasets=datasets,
                    delta_from=delta_from,
                )
            )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_row(
    model: str,
    setting: str,
    cell: Cell,
    results: dict[str, dict[Cell, dict[str, Any]]],
    table_ks: list[int],
    datasets: list[str],
    delta_from: Cell | None = None,
) -> str:
    values = []
    for dataset in datasets:
        dataset_results = results.get(dataset, {})
        metrics = dataset_results.get(cell)
        baseline = dataset_results.get(delta_from) if delta_from is not None else None
        if metrics is None or (delta_from is not None and baseline is None):
            values.extend(["--"] * (3 * len(table_ks)))
        else:
            for k in table_ks:
                for name in ["hit", "recall", "ndcg"]:
                    value = metric_for_k(metrics, name, k)
                    if delta_from is not None:
                        base = metric_for_k(baseline, name, k)
                        value = None if value is None or base is None else value - base
                    values.append(fmt_delta(value) if delta_from is not None else fmt_pct(value))
    return (
        f"{escape_tex(model)} & {setting} "
        + " ".join(f"& {value}" for value in values)
        + " \\\\"
    )


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.1f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:+.1f}"


def metric_for_k(metrics: dict[str, Any], name: str, k: int) -> float | None:
    by_k = metrics.get("by_k", {})
    if isinstance(by_k, dict) and k in by_k and name in by_k[k]:
        return float(by_k[k][name])
    if not by_k and k == 10:
        return float(metrics.get(name, 0.0))
    return None


def parse_ks(value: str) -> list[int]:
    ks = sorted({
        int(part.strip())
        for part in value.split(",")
        if part.strip() and int(part.strip()) > 0
    })
    return ks or [10]


def escape_tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


if __name__ == "__main__":
    main()
