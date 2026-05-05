"""Conversational recommendation evaluation entry point.

Usage:
    python datasets/conversational_rec/run.py --model-provider openai \
        --model-config configs/models/gpt4o_mini.yaml
    python datasets/conversational_rec/run.py --library-path /path/to/library \
        --only-origin TD

Default mode is axis_sweep, matching the AIME runner: it runs the three
one-axis sweeps and deduplicates the overlapping FL/NR/FR baseline into
6 unique experiments per model/dataset/origin. Use --mode full_grid for
the full 3 x 3 x 2 = 18-cell taxonomy grid.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import build_model, warn_compatibility_issues
from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry
from skilleval.core.types import SkillOrigin, SkillRetrieval, SkillVisibility, UpdateStrategy
from skilleval.evaluation.evaluator import Evaluator
from skilleval.evaluation.reporters import ConsoleReporter, CSVReporter, JSONReporter
from skilleval.skills.library import SkillLibrary
from skilleval.utils.logging import setup_logging

import skilleval.datasets.conversational_rec  # noqa: F401

logger = logging.getLogger(__name__)

ORIGIN_MAP = {"SD": SkillOrigin.SPEC_DERIVED, "TD": SkillOrigin.TRACE_DERIVED}
AXIS_CHOICES = ("visibility", "retrieval", "evolution")
MODEL_PRESETS = {
    "gpt4o-mini": ("openai", "datasets/conversational_rec/models/gpt4o_mini.yaml"),
    "gpt5-mini": ("openai", "datasets/conversational_rec/models/gpt5_mini_2025_08_07.yaml"),
    "gpt-5-mini-2025-08-07": (
        "openai",
        "datasets/conversational_rec/models/gpt5_mini_2025_08_07.yaml",
    ),
    "qwen3-4b": ("vllm", "datasets/conversational_rec/models/qwen3_4b_instruct_vllm.yaml"),
    "qwen2.5-7b": ("vllm", "datasets/conversational_rec/models/qwen25_7b_instruct_vllm.yaml"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational recommendation evaluation")
    parser.add_argument("--config", type=str, default="datasets/conversational_rec/config.yaml")
    parser.add_argument("--mode", type=str, default="axis_sweep", choices=("single_axis", "axis_sweep", "full_grid"))
    parser.add_argument("--model", type=str, default="gpt4o-mini")
    parser.add_argument("--model-provider", type=str, default="openai")
    parser.add_argument("--model-config", type=str, default="configs/models/gpt4o_mini.yaml")
    parser.add_argument("--only-origin", type=str, default="SD", choices=list(ORIGIN_MAP) + ["all"])
    parser.add_argument("--only-axis", type=str, default="visibility", choices=AXIS_CHOICES)
    parser.add_argument("--protocol", type=str, default=None, choices=("in_context", "react", "tool_using", "anthropic_style"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="datasets/conversational_rec/outputs")
    parser.add_argument(
        "--library-path",
        type=str,
        default=None,
        help=(
            "Path to an existing skill library directory. The path should be "
            "the parent directory containing manifest.json and/or skills/."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    warn_compatibility_issues()

    config = Config.from_yaml(args.config)
    if args.protocol:
        config.set("experiment.protocol", args.protocol)
    dataset = dataset_registry.build(config.get("dataset.name"), config)
    logger.info("Dataset summary: %s", dataset.summary())

    max_episodes = args.max_episodes or int(config.get("experiment.max_episodes", 20))
    max_steps = int(config.get("experiment.max_steps_per_episode", 4))
    num_rounds = int(config.get("experiment.num_evolution_rounds", 1))

    results = []
    for provider, model_config in _resolve_models(args):
        model = build_model(provider, model_config)
        logger.info("Model: %s", model.name)
        evaluator = Evaluator(config)
        for origin in _resolve_origins(args.only_origin):
            if args.library_path:
                library = SkillLibrary.load_auto(args.library_path)
                logger.info(
                    "Library loaded from %s: %d skills",
                    args.library_path,
                    library.size,
                )
            else:
                library = evaluator.create_library_once(dataset, model, skill_origin=origin)
                logger.info("Library (%s) created: %d skills", origin.value, library.size)
            results.extend(_run_mode(
                args.mode,
                args.only_axis,
                evaluator,
                dataset,
                model,
                origin,
                library,
                max_episodes,
                max_steps,
                num_rounds,
            ))

    # Axis helpers overlap at FL/NR/FR across full sweeps; dedupe defensively.
    unique = {r.experiment_id: r for r in results}
    out_dir = Path(args.output_dir)
    JSONReporter(out_dir).save(list(unique.values()))
    CSVReporter(out_dir).save(list(unique.values()))
    ConsoleReporter.report(list(unique.values()))


def _resolve_models(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.model == "all":
        return list(MODEL_PRESETS.values())
    if args.model in MODEL_PRESETS:
        return [MODEL_PRESETS[args.model]]
    return [(args.model_provider, args.model_config)]


def _resolve_origins(value: str) -> list[SkillOrigin]:
    if value == "all":
        return list(ORIGIN_MAP.values())
    return [ORIGIN_MAP[value]]


def _run_mode(
    mode: str,
    only_axis: str,
    evaluator: Evaluator,
    dataset: object,
    model: object,
    origin: SkillOrigin,
    library: object,
    max_episodes: int,
    max_steps: int,
    num_rounds: int,
):
    if mode == "full_grid":
        results = []
        for visibility in SkillVisibility:
            for retrieval in SkillRetrieval:
                for evolution in (UpdateStrategy.FROZEN, UpdateStrategy.BATCH_UPDATE):
                    results.append(evaluator.run_single(
                        dataset,
                        model,
                        visibility=visibility,
                        retrieval=retrieval,
                        evolution=evolution,
                        skill_origin=origin,
                        max_episodes=max_episodes,
                        max_steps=max_steps,
                        num_evolution_rounds=num_rounds,
                        skill_library=library,
                    ))
        return results

    if mode == "axis_sweep":
        return [
            *evaluator.run_visibility_axis(
                dataset, model,
                skill_origin=origin,
                max_episodes=max_episodes,
                max_steps=max_steps,
                skill_library=library,
            ),
            *evaluator.run_retrieval_axis(
                dataset, model,
                skill_origin=origin,
                max_episodes=max_episodes,
                max_steps=max_steps,
                skill_library=library,
            ),
            *evaluator.run_evolution_axis(
                dataset, model,
                skill_origin=origin,
                max_episodes=max_episodes,
                max_steps=max_steps,
                num_evolution_rounds=num_rounds,
                skill_library=library,
            ),
        ]

    if only_axis == "visibility":
        return evaluator.run_visibility_axis(
            dataset, model,
            skill_origin=origin,
            max_episodes=max_episodes,
            max_steps=max_steps,
            skill_library=library,
        )
    if only_axis == "retrieval":
        return evaluator.run_retrieval_axis(
            dataset, model,
            skill_origin=origin,
            max_episodes=max_episodes,
            max_steps=max_steps,
            skill_library=library,
        )
    return evaluator.run_evolution_axis(
        dataset, model,
        skill_origin=origin,
        max_episodes=max_episodes,
        max_steps=max_steps,
        num_evolution_rounds=num_rounds,
        skill_library=library,
    )


if __name__ == "__main__":
    main()
