#!/usr/bin/env python3
"""Generate LaTeX tables from experiment results.

Usage:
    python visualization/generate_tables.py --results outputs/results.json --output-dir outputs/tables
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main_results_table(results: list[dict]) -> str:
    """Main results table: rows = dataset x model, columns = origin x policy."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main benchmark results: success rate (\%) across taxonomy dimensions.}",
        r"\label{tab:main-results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll" + "c" * 9 + "}",
        r"\toprule",
    ]

    origins = ["SD", "TD", "FO"]
    policies = ["OB", "RR", "PV"]
    header_top = r" & & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textbf{{{o}}}}}" for o in origins
    ) + r" \\"
    header_bot = r"\textbf{Dataset} & \textbf{Model} & " + " & ".join(
        [p for o in origins for p in policies]
    ) + r" \\"
    lines.append(header_top)
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}")
    lines.append(header_bot)
    lines.append(r"\midrule")

    grouped: dict[tuple[str, str], dict[tuple[str, str], float]] = defaultdict(dict)
    for r in results:
        key = (r["dataset_id"], r["model_id"])
        cell = (r["skill_origin"], r["runtime_policy"])
        grouped[key][cell] = r["success_rate"]

    for (ds, model), cells in sorted(grouped.items()):
        row = f"{ds} & {model}"
        for o in origins:
            for p in policies:
                val = cells.get((o, p))
                row += f" & {val*100:.1f}" if val is not None else " & --"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def pattern_breakdown_table(results: list[dict]) -> str:
    """Per-pattern breakdown table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Success rate (\%) by composition pattern.}",
        r"\label{tab:pattern-breakdown}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Config} & \textbf{SL} & \textbf{PO} & \textbf{FP} \\",
        r"\midrule",
    ]

    for r in results:
        pp = r.get("per_pattern", {})
        if not pp:
            continue
        config = f"{r['skill_origin']}/{r['runtime_policy']}"
        sl = f"{pp.get('SL', 0)*100:.1f}" if "SL" in pp else "--"
        po = f"{pp.get('PO', 0)*100:.1f}" if "PO" in pp else "--"
        fp = f"{pp.get('FP', 0)*100:.1f}" if "FP" in pp else "--"
        lines.append(f"{r['dataset_id']} & {config} & {sl} & {po} & {fp} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
    parser.add_argument("--results", type=str, default="outputs/results.json")
    parser.add_argument("--output-dir", type=str, default="outputs/tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results)

    main_table = main_results_table(results)
    (output_dir / "main_results.tex").write_text(main_table)
    print(f"Main results table -> {output_dir / 'main_results.tex'}")

    pattern_table = pattern_breakdown_table(results)
    (output_dir / "pattern_breakdown.tex").write_text(pattern_table)
    print(f"Pattern breakdown table -> {output_dir / 'pattern_breakdown.tex'}")


if __name__ == "__main__":
    main()
