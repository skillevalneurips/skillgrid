"""Discover per-dataset SkillRecipe overrides.

Looks for ``datasets/<dataset_name>/skills/recipe.py`` with a
``build_recipe()`` entry point. Falls back to a default ``SkillRecipe()``
if no override is present, so core callers can always assume a recipe.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from skilleval.skills.creation.recipe import SkillRecipe

logger = logging.getLogger(__name__)


def load_dataset_recipe(
    dataset_name: str,
    datasets_root: Path | None = None,
) -> SkillRecipe:
    """Resolve a ``SkillRecipe`` for the named dataset.

    Search order:
      1. ``<datasets_root>/<dataset_name>/skills/recipe.py:build_recipe()``
      2. default ``SkillRecipe()`` if no file exists

    ``datasets_root`` defaults to ``<project_root>/datasets``.
    """
    if datasets_root is None:
        # project_root = .../skillbench-eval/; this file lives at
        # .../skillbench-eval/skilleval/skills/creation/loader.py
        datasets_root = Path(__file__).resolve().parents[3] / "datasets"

    recipe_path = datasets_root / dataset_name / "skills" / "recipe.py"
    if not recipe_path.exists():
        logger.info(
            "No per-dataset recipe for %s (expected at %s) — using defaults.",
            dataset_name, recipe_path,
        )
        return SkillRecipe()

    try:
        spec = importlib.util.spec_from_file_location(
            f"{dataset_name}_skill_recipe", recipe_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning("Failed to load recipe %s: %s — using defaults.", recipe_path, exc)
        return SkillRecipe()

    if not hasattr(module, "build_recipe"):
        logger.warning(
            "Recipe %s has no build_recipe() — using defaults.", recipe_path,
        )
        return SkillRecipe()

    try:
        recipe = module.build_recipe()
    except Exception as exc:
        logger.warning(
            "build_recipe() raised: %s — using defaults.", exc,
        )
        return SkillRecipe()

    if not isinstance(recipe, SkillRecipe):
        logger.warning(
            "build_recipe() in %s returned %s, expected SkillRecipe — using defaults.",
            recipe_path, type(recipe).__name__,
        )
        return SkillRecipe()

    logger.info(
        "Loaded recipe for %s: seed_skills=%d, sd_prompt=%s, td_prompt=%s",
        dataset_name, len(recipe.seed_skills),
        "custom" if recipe.sd_prompt else "default",
        "custom" if recipe.td_prompt else "default",
    )
    return recipe
