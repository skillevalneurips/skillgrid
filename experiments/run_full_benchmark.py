#!/usr/bin/env python3
"""Run the full benchmark grid across all taxonomy dimensions.

Usage:
    python experiments/run_full_benchmark.py --config configs/experiments/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry, model_registry
from skilleval.core.types import RuntimePolicy, SkillOrigin
from skilleval.evaluation.evaluator import Evaluator
from skilleval.evaluation.reporters import ConsoleReporter, CSVReporter, JSONReporter
from skilleval.utils.logging import setup_logging
from experiments.common import (
    MODEL_CONFIG_MAP,
    resolve_dataset_config_path,
    warn_compatibility_issues,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full SkillEval Benchmark")
    parser.add_argument("--config", type=str, default="configs/experiments/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    output_dir = args.output_dir or config.get("project.output_dir", "outputs")
    setup_logging(args.log_level, log_file=Path(output_dir) / "experiment.log")
    warn_compatibility_issues()

    dataset_names = config.get("experiment.datasets", ["gsm8k"])
    model_names = config.get("experiment.models", ["gpt-4o"])
    origin_names = config.get("experiment.skill_origins", ["SD", "TD", "FO"])
    policy_names = config.get("experiment.runtime_policies", ["OB", "RR", "PV"])
    max_episodes = config.get("experiment.max_episodes", 100)
    max_steps = config.get("experiment.max_steps_per_episode", 50)

    datasets = []
    for ds_name in dataset_names:
        try:
            ds_cfg = Config.from_yaml(resolve_dataset_config_path(ds_name))
            ds = dataset_registry.build(ds_name, ds_cfg)
            datasets.append(ds)
        except Exception as e:
            logger.warning("Could not load dataset %s: %s", ds_name, e)

    models = []
    for m_name in model_names:
        if m_name in MODEL_CONFIG_MAP:
            provider, cfg_path = MODEL_CONFIG_MAP[m_name]
            try:
                m_cfg = Config.from_yaml(cfg_path)
                m = model_registry.build(provider, m_cfg)
                models.append(m)
            except Exception as e:
                logger.warning("Could not load model %s: %s", m_name, e)

    origins = [SkillOrigin(o) for o in origin_names]
    policies = [RuntimePolicy(p) for p in policy_names]

    logger.info(
        "Benchmark grid: %d datasets x %d models x %d origins x %d policies = %d cells",
        len(datasets), len(models), len(origins), len(policies),
        len(datasets) * len(models) * len(origins) * len(policies),
    )

    evaluator = Evaluator(config)
    results = evaluator.run_grid(
        datasets=datasets,
        models=models,
        skill_origins=origins,
        runtime_policies=policies,
        max_episodes=max_episodes,
        max_steps=max_steps,
    )

    ConsoleReporter.report(results)
    json_reporter = JSONReporter(output_dir)
    json_reporter.save(results)
    csv_reporter = CSVReporter(output_dir)
    csv_reporter.save(results)

    logger.info("Full benchmark complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
