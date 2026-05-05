"""Core data types, enums, and dataclasses for the SkillEval framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Taxonomy enums (aligned with paper Section 4)
# ---------------------------------------------------------------------------

# Cat 1: Skill Creation
class SkillOrigin(str, Enum):
    """Cat 1 – Skill Creation: How the skill library is constructed."""
    SPEC_DERIVED = "SD"        # LLM writes skills from dataset + tool specs
    TRACE_DERIVED = "TD"       # LLM writes skills from probe traces


# Cat 2: Runtime Policy
class RuntimePolicy(str, Enum):
    """Cat 2 – Runtime Policy: Runtime skill access and composition policy."""
    NO_RETRIEVAL = "NR"     # Pass-through: agent uses Axis 1 skills as-is
    ORACLE_BUNDLE = "OB"    # Pre-selected bundle of skills (legacy)
    RETRIEVE_ROUTE = "RR"   # Agent retrieves from library (may include distractors)
    PLAN_VERIFY = "PV"      # Agent plans and verifies step-by-step


# Cat 3: Skill Evolution
class UpdateStrategy(str, Enum):
    """Cat 3 – Skill Evolution: Library update strategy across episodes."""
    FROZEN = "FR"        # No updates allowed
    BATCH_UPDATE = "BU"  # Updates between rounds using traces
    ONLINE_UPDATE = "OU" # Online updates during evaluation


# ---------------------------------------------------------------------------
# Paper Table 2 – Three orthogonal evaluation axes
# ---------------------------------------------------------------------------

class SkillVisibility(str, Enum):
    """Axis 1 – Skill Visibility: What the agent sees at episode start."""
    NO_LIBRARY = "NL"       # No pre-loaded skills; backbone only
    LIMITED_BUNDLE = "LB"   # Curated subset of episode-relevant skills
    FULL_LIBRARY = "FL"     # Complete skill library available


class SkillRetrieval(str, Enum):
    """Axis 2 – Skill Retrieval: How skills are selected during an episode."""
    NO_RETRIEVAL = "NR"     # Agent uses only initial bundle, no further retrieval
    RETRIEVE_ROUTE = "RR"   # On-demand retrieval from library during execution
    PLAN_RETRIEVE = "PR"    # Planner retrieves and routes; separate executor acts


class TaskDomain(str, Enum):
    """Supported task domains."""
    GENERAL_REASONING = "general_reasoning"
    CONVERSATIONAL_REC = "conversational_rec"
    EXECUTION_ENV = "execution_env"
    MATH_REASONING = "math_reasoning"
    WEB_NAVIGATION = "web_navigation"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskInstance:
    """A single benchmark task instance."""
    task_id: str
    domain: TaskDomain
    instruction: str
    composition_pattern: str | None = None
    tools_required: list[str] = field(default_factory=list)
    gold_answer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain.value,
            "composition_pattern": self.composition_pattern,
            "instruction": self.instruction,
            "tools_required": self.tools_required,
            "gold_answer": self.gold_answer,
            "metadata": self.metadata,
        }


@dataclass
class SkillSpec:
    """Specification for a single reusable skill."""
    skill_id: str
    name: str
    description: str
    origin: SkillOrigin
    level: int = 1  # 1=atomic, 2=macro, 3=programmatic
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    template: str = ""
    fallback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "origin": self.origin.value,
            "level": self.level,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "tool_calls": self.tool_calls,
            "template": self.template,
            "fallback": self.fallback,
            "metadata": self.metadata,
        }


@dataclass
class TraceEntry:
    """A single step in an interaction trace."""
    step: int
    action: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    skill_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "observation": self.observation,
            "state": self.state,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "skill_used": self.skill_used,
        }


@dataclass
class EpisodeTrace:
    """Complete trace for one task episode."""
    task_id: str
    model_id: str
    entries: list[TraceEntry] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    success: bool = False
    final_answer: str | None = None
    task_instruction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.entries)

    @property
    def num_errors(self) -> int:
        return sum(1 for e in self.entries if not e.success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "entries": [e.to_dict() for e in self.entries],
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "final_answer": self.final_answer,
            "task_instruction": self.task_instruction,
            "num_steps": self.num_steps,
            "num_errors": self.num_errors,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    experiment_id: str
    model_id: str
    dataset_id: str
    skill_origin: SkillOrigin
    runtime_policy: RuntimePolicy
    # Paper Table 2 taxonomy axes (optional for backward compat)
    visibility: SkillVisibility | None = None
    retrieval: SkillRetrieval | None = None
    evolution: UpdateStrategy | None = None
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_cost: float = 0.0
    avg_tokens: int = 0
    recovery_rate: float = 0.0
    selection_accuracy: float = 0.0
    per_pattern: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "skill_origin": self.skill_origin.value,
            "runtime_policy": self.runtime_policy.value,
            "visibility": self.visibility.value if self.visibility else None,
            "retrieval": self.retrieval.value if self.retrieval else None,
            "evolution": self.evolution.value if self.evolution else None,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_cost": self.avg_cost,
            "avg_tokens": self.avg_tokens,
            "recovery_rate": self.recovery_rate,
            "selection_accuracy": self.selection_accuracy,
            "per_pattern": self.per_pattern,
            "metadata": self.metadata,
        }
        return d
