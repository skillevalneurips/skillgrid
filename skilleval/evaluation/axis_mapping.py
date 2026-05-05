"""Map paper Table 2 taxonomy axes to concrete runtime objects.

Axis 1 (Visibility) determines what skills the agent sees at episode start.
Axis 2 (Retrieval) determines how skills are used during execution.
Axis 3 (Evolution) determines whether the library is updated between rounds.

These axes are independent — Axis 1 is resolved before the agent is created,
Axis 2 controls runtime behavior, Axis 3 controls between-round updates.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import Any

from skilleval.core.types import (
    SkillRetrieval,
    SkillVisibility,
    TaskInstance,
    UpdateStrategy,
)
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


def resolve_initial_library(
    visibility: SkillVisibility,
    full_library: SkillLibrary,
    task: TaskInstance | None,
    lb_sample_size: int = 1,
    lb_seed: int = 0,
    lb_selection: str = "random",
) -> SkillLibrary:
    """Axis 1: Determine the skill library visible to the agent at episode start.

    - NL: Empty library (0 skills). Agent solves with backbone only.
    - LB: Limited bundle. By default, randomly sample ``lb_sample_size`` skills
      deterministically from ``lb_seed`` and ``task.task_id``. Configs may opt
      into semantic relevance ranking.
    - FL: Full library. Agent sees all skills.
    """
    if visibility == SkillVisibility.NO_LIBRARY:
        return SkillLibrary()

    if visibility == SkillVisibility.FULL_LIBRARY:
        return full_library

    # LIMITED_BUNDLE: task-relevant subset.
    all_skills = full_library.list_skills()
    if not all_skills:
        return SkillLibrary()

    k = max(1, lb_sample_size)
    if lb_selection == "random" or task is None:
        sampled = _sample_skills(all_skills, task, k, lb_seed)
    else:
        sampled = _rank_skills_by_relevance(task, all_skills, k)
        if len(sampled) < min(k, len(all_skills)):
            already = {s.skill_id for s in sampled}
            rest = [s for s in all_skills if s.skill_id not in already]
            sampled.extend(_sample_skills(rest, task, k - len(sampled), lb_seed))

    bundle = SkillLibrary()
    for skill in sampled:
        bundle.add(skill)
    return bundle


def _sample_skills(
    skills: list[Any], task: TaskInstance | None, k: int, seed_value: int
) -> list[Any]:
    seed_source = f"{seed_value}:{getattr(task, 'task_id', '')}".encode()
    seed = int(hashlib.md5(seed_source).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return rng.sample(skills, min(k, len(skills)))


def _rank_skills_by_relevance(
    task: TaskInstance, skills: list[Any], k: int
) -> list[Any]:
    query = task.instruction
    query_tokens = _tokens(query)
    scored: list[tuple[float, str, Any]] = []
    for skill in skills:
        text = " ".join([
            getattr(skill.spec, "name", ""),
            getattr(skill.spec, "description", ""),
            " ".join(getattr(skill.spec, "tool_calls", []) or []),
        ])
        score = _keyword_relevance(query_tokens, _tokens(text))
        scored.append((score, skill.skill_id, skill))

    scored.sort(key=lambda item: (-item[0], item[1]))
    relevant = [skill for score, _, skill in scored if score > 0]
    return relevant[:k]


def _keyword_relevance(query_tokens: set[str], skill_tokens: set[str]) -> float:
    if not query_tokens or not skill_tokens:
        return 0.0
    overlap = query_tokens & skill_tokens
    return len(overlap) / max(len(query_tokens), 1)


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in _STOPWORDS
    }


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "you", "are",
    "was", "were", "have", "has", "had", "but", "not", "use", "using",
    "recommend", "recommendation", "movie", "movies", "film", "films",
}


def resolve_retrieval_policy(retrieval: SkillRetrieval) -> str:
    """Axis 2: Map Retrieval setting to runtime policy code.

    - NR: No retrieval. Agent uses Axis 1 skills as-is.
    - RR: Semantic retrieval at each step.
    - PR: Planner agent picks which skill to use.
    """
    mapping = {
        SkillRetrieval.NO_RETRIEVAL: "NR",
        SkillRetrieval.RETRIEVE_ROUTE: "RR",
        SkillRetrieval.PLAN_RETRIEVE: "PV",
    }
    return mapping[retrieval]


def validate_axis_combination(
    visibility: SkillVisibility,
    retrieval: SkillRetrieval,
    evolution: UpdateStrategy,
) -> list[str]:
    """Return warnings for degenerate axis combinations."""
    warnings: list[str] = []

    if visibility == SkillVisibility.NO_LIBRARY:
        if retrieval != SkillRetrieval.NO_RETRIEVAL:
            warnings.append(
                f"NL + {retrieval.value}: nothing to retrieve from an empty "
                "library; retrieval will return no skills."
            )
        if evolution == UpdateStrategy.BATCH_UPDATE:
            warnings.append(
                "NL + BU: library starts empty; BU will attempt to "
                "bootstrap skills from traces alone."
            )

    return warnings
