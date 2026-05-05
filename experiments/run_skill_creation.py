#!/usr/bin/env python3
"""Run skill creation experiments: compare top-down, bottom-up, and hybrid.

Usage:
    python experiments/run_skill_creation.py --dataset gsm8k --model openai \
        --config configs/models/gpt4o.yaml --strategy top_down
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skilleval.skills.creation.top_down import TopDownCreator
from skilleval.skills.creation.bottom_up import BottomUpCreator
from skilleval.skills.creation.hybrid import HybridCreator
from skilleval.utils.logging import setup_logging
from skilleval.utils.io import save_json
from experiments.common import build_dataset, build_model, warn_compatibility_issues

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Creation Experiment")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="top_down",
                        choices=["top_down", "bottom_up", "hybrid", "all"])
    parser.add_argument("--output-dir", type=str, default="outputs/skill_creation")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    warn_compatibility_issues()

    model = build_model(args.model, args.config)

    dataset = build_dataset(args.dataset, args.dataset_config)

    strategies = [args.strategy] if args.strategy != "all" else ["top_down", "bottom_up", "hybrid"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        logger.info("Running skill creation: strategy=%s, dataset=%s", strategy, args.dataset)

        if strategy == "top_down":
            creator = TopDownCreator(model)
            library = creator.create(dataset)
        elif strategy == "bottom_up":
            creator_bu = BottomUpCreator(model)
            library = creator_bu.create([])  # needs traces; empty for demo
        elif strategy == "hybrid":
            creator_h = HybridCreator(model)
            library = creator_h.create(dataset)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        summary = library.summary()
        logger.info("Strategy %s: %s", strategy, summary)

        library.save(output_dir / f"library_{strategy}_{args.dataset}.json")
        save_json(summary, output_dir / f"summary_{strategy}_{args.dataset}.json")

    logger.info("Skill creation experiments complete.")


if __name__ == "__main__":
    main()
