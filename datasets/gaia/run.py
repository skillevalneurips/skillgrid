"""GAIA 3-axis evaluation entry point.

Discovers all model definitions in datasets/gaia/models/*.py and runs
the full taxonomy grid (Visibility x Retrieval x Evolution) for each model.

Skills are generated ONCE per model and reused across all experiments.

Usage:
    python datasets/gaia/run.py                    # run all models
    python datasets/gaia/run.py --model gpt4o_mini  # run one model
    python datasets/gaia/run.py --only-origin TD    # TD skills only
    python datasets/gaia/run.py --only-axis retrieval
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    # Fall back to manual loading if python-dotenv not installed
    _env_path = ROOT / ".env"
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry
from skilleval.core.types import SkillOrigin
from skilleval.debug.wrapper import DebugModel
from skilleval.evaluation.evaluator import Evaluator
from skilleval.evaluation.reporters import ConsoleReporter, CSVReporter, JSONReporter

import skilleval.datasets.gaia  # noqa: F401

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_DIR = Path("outputs/debug") / RUN_ID
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

_log_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
_stderr = logging.StreamHandler()
_stderr.setFormatter(_log_fmt)
_file = logging.FileHandler(DEBUG_DIR / "run.log", encoding="utf-8")
_file.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_stderr, _file], force=True)
logger = logging.getLogger(__name__)
logger.info("Debug output dir: %s", DEBUG_DIR)


def discover_models(models_dir: Path, filter_name: str | None = None) -> list[tuple[str, any]]:
    """Discover model files in the models/ directory.

    Each model file must have:
      - MODEL_NAME: str  (human-readable name for results)
      - create_model() -> BaseModel

    Returns list of (model_name, model_instance) tuples.
    """
    models = []
    for model_file in sorted(models_dir.glob("*.py")):
        if model_file.name.startswith("_"):
            continue

        stem = model_file.stem
        if filter_name and stem != filter_name:
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"gaia_models.{stem}", model_file,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            model_name = getattr(module, "MODEL_NAME", stem)
            model = module.create_model()
            models.append((model_name, model))
            logger.info("Loaded model: %s (%s)", model_name, model_file.name)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", model_file.name, exc)

    return models


ORIGIN_MAP = {"SD": SkillOrigin.SPEC_DERIVED, "TD": SkillOrigin.TRACE_DERIVED}
AXIS_CHOICES = ("visibility", "retrieval", "evolution")


def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (filename without .py)")
    parser.add_argument(
        "--only-origin", type=str, default=None, choices=list(ORIGIN_MAP),
        help="Run only this skill origin (SD or TD). Default: both.",
    )
    parser.add_argument(
        "--only-axis", type=str, default=None, choices=AXIS_CHOICES,
        help="Run only this axis (visibility, retrieval, or evolution). "
             "Default: all three.",
    )
    parser.add_argument(
        "--library-path", type=str, default=None,
        help="Path to existing skill library (skips skill generation).",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Override max_episodes from config (number of tasks to evaluate).",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    config = Config.from_yaml(config_path)

    dataset = dataset_registry.build(config.get("dataset.name"), config)
    dataset.load()

    models_dir = Path(__file__).parent / "models"
    models = discover_models(models_dir, filter_name=args.model)

    if not models:
        logger.error("No models found in %s", models_dir)
        sys.exit(1)

    max_episodes = args.max_episodes if args.max_episodes is not None else int(config.get("experiment.max_episodes", 50))
    max_steps = int(config.get("experiment.max_steps_per_episode", 16))
    num_rounds = int(config.get("experiment.num_evolution_rounds", 1))

    if args.only_origin:
        origins_to_run = [ORIGIN_MAP[args.only_origin]]
    else:
        origins_to_run = [SkillOrigin.SPEC_DERIVED, SkillOrigin.TRACE_DERIVED]

    axes_to_run = [args.only_axis] if args.only_axis else list(AXIS_CHOICES)

    all_results = []

    for model_name, model in models:
        logger.info("=" * 60)
        logger.info("Running model: %s", model_name)
        logger.info("=" * 60)

        model = DebugModel(model, log_dir=DEBUG_DIR)
        evaluator = Evaluator(config)

        for origin in origins_to_run:
            logger.info("--- origin=%s ---", origin.value)

            if args.library_path:
                from skilleval.skills.library import SkillLibrary
                library = SkillLibrary.load_auto(args.library_path)
                logger.info(
                    "Library loaded from %s: %d skills",
                    args.library_path, library.size,
                )
            else:
                library = evaluator.create_library_once(
                    dataset, model, skill_origin=origin,
                )
                logger.info(
                    "Library (%s) created: %d skills",
                    origin.value, library.size,
                )

            if "visibility" in axes_to_run:
                all_results.extend(evaluator.run_visibility_axis(
                    dataset, model,
                    skill_origin=origin,
                    max_episodes=max_episodes, max_steps=max_steps,
                    skill_library=library,
                ))
            if "retrieval" in axes_to_run:
                all_results.extend(evaluator.run_retrieval_axis(
                    dataset, model,
                    skill_origin=origin,
                    max_episodes=max_episodes, max_steps=max_steps,
                    skill_library=library,
                ))
            if "evolution" in axes_to_run:
                all_results.extend(evaluator.run_evolution_axis(
                    dataset, model,
                    skill_origin=origin,
                    max_episodes=max_episodes, max_steps=max_steps,
                    num_evolution_rounds=num_rounds,
                    skill_library=library,
                ))

    seen: set[str] = set()
    unique_results = []
    for r in all_results:
        if r.experiment_id not in seen:
            seen.add(r.experiment_id)
            unique_results.append(r)

    out_dir = Path(__file__).parent / "outputs"
    JSONReporter(str(out_dir)).save(unique_results)
    CSVReporter(str(out_dir)).save(unique_results)
    ConsoleReporter.report(unique_results)


if __name__ == "__main__":
    main()
