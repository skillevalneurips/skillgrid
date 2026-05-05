#!/usr/bin/env python3
"""Run baseline experiments: no skills, direct model + tools.

Usage:
    python experiments/run_baseline.py --dataset gsm8k --model openai --config configs/models/gpt4o.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skilleval.core.types import RuntimePolicy, SkillOrigin
from skilleval.evaluation.metrics import MetricsComputer
from skilleval.evaluation.reporters import ConsoleReporter, JSONReporter
from skilleval.skills.library import SkillLibrary
from skilleval.agents.executor import SkillAgent
from skilleval.utils.logging import setup_logging
from experiments.common import build_dataset, build_model, warn_compatibility_issues

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillEval Baseline Experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. gsm8k)")
    parser.add_argument("--model", type=str, default="openai", help="Model provider")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML path")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset config YAML path")
    parser.add_argument("--max-episodes", type=int, default=10, help="Max tasks to evaluate")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    warn_compatibility_issues()

    model = build_model(args.model, args.config)
    logger.info("Model: %s", model.name)

    dataset = build_dataset(args.dataset, args.dataset_config)
    logger.info("Dataset: %s (%d tasks)", dataset.name, len(dataset))

    empty_library = SkillLibrary()
    agent = SkillAgent(
        model=model,
        library=empty_library,
        protocol="tool_using",
        policy="OB",
        env_tools=dataset.get_tools(),
    )

    tasks = dataset.tasks()[:args.max_episodes]
    logger.info("Running baseline on %d tasks...", len(tasks))
    traces = agent.solve_batch(tasks, max_steps=args.max_steps)

    metrics = MetricsComputer()
    result = metrics.compute(
        traces=traces,
        tasks=tasks,
        experiment_id=f"baseline_{args.dataset}_{model.model_id}",
        model_id=model.model_id,
        dataset_id=args.dataset,
        skill_origin=SkillOrigin.SPEC_DERIVED,
        runtime_policy=RuntimePolicy.ORACLE_BUNDLE,
    )

    ConsoleReporter.report([result])
    reporter = JSONReporter(args.output_dir)
    reporter.save([result], f"baseline_{args.dataset}_{model.model_id}.json")
    reporter.save_traces(traces, f"baseline_{args.dataset}_{model.model_id}_traces.json")


if __name__ == "__main__":
    main()
