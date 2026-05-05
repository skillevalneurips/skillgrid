#!/usr/bin/env python3
"""Run skill composition experiments: compare protocols and runtime policies.

Usage:
    python experiments/run_composition.py --dataset gsm8k --model openai \
        --config configs/models/gpt4o.yaml --protocol tool_using --policy OB
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skilleval.core.types import RuntimePolicy, SkillOrigin
from skilleval.agents.executor import SkillAgent
from skilleval.evaluation.metrics import MetricsComputer
from skilleval.evaluation.reporters import ConsoleReporter, JSONReporter
from skilleval.skills.creation.top_down import TopDownCreator
from skilleval.utils.logging import setup_logging
from experiments.common import build_dataset, build_model, warn_compatibility_issues

logger = logging.getLogger(__name__)

PROTOCOLS = ["tool_using", "in_context", "anthropic_style"]
POLICIES = ["OB", "RR", "PV"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Composition Experiment")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--protocol", type=str, default="all", choices=PROTOCOLS + ["all"])
    parser.add_argument("--policy", type=str, default="all", choices=POLICIES + ["all"])
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/composition")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    warn_compatibility_issues()

    model = build_model(args.model, args.config)

    dataset = build_dataset(args.dataset, args.dataset_config)

    creator = TopDownCreator(model)
    library = creator.create(dataset)

    protocols = PROTOCOLS if args.protocol == "all" else [args.protocol]
    policies = POLICIES if args.policy == "all" else [args.policy]

    all_results = []
    tasks = dataset.tasks()[:args.max_episodes]

    for protocol in protocols:
        for policy in policies:
            logger.info("Running: protocol=%s, policy=%s", protocol, policy)
            try:
                agent = SkillAgent(
                    model=model,
                    library=library,
                    protocol=protocol,
                    policy=policy,
                    env_tools=dataset.get_tools(),
                )
                traces = agent.solve_batch(tasks, max_steps=args.max_steps)
                metrics = MetricsComputer()
                result = metrics.compute(
                    traces=traces,
                    tasks=tasks,
                    experiment_id=f"comp_{args.dataset}_{protocol}_{policy}",
                    model_id=model.model_id,
                    dataset_id=args.dataset,
                    skill_origin=SkillOrigin.SPEC_DERIVED,
                    runtime_policy=RuntimePolicy(policy),
                )
                all_results.append(result)
            except Exception as e:
                logger.error("Failed: protocol=%s, policy=%s: %s", protocol, policy, e)

    ConsoleReporter.report(all_results)
    reporter = JSONReporter(args.output_dir)
    reporter.save(all_results, f"composition_{args.dataset}_{model.model_id}.json")


if __name__ == "__main__":
    main()
