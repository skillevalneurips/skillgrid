"""Main agent executor that ties together model, skills, protocol, and policy."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from skilleval.agents.base import BaseAgent
from skilleval.core.config import Config
from skilleval.core.types import (
    EpisodeTrace,
    RuntimePolicy,
    TaskInstance,
    TraceEntry,
)
from skilleval.datasets.base import BaseDataset
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary
from skilleval.skills.protocols.anthropic_style import AnthropicStyleProtocol
from skilleval.skills.protocols.in_context import InContextProtocol
from skilleval.skills.protocols.gaia_react import GaiaReActProtocol
from skilleval.skills.protocols.react import ReActProtocol
from skilleval.skills.protocols.tool_using import ToolUsingProtocol
from skilleval.skills.runtime.no_retrieval import NoRetrievalPolicy
from skilleval.skills.runtime.plan_verify import PlanVerifyPolicy
from skilleval.skills.runtime.retrieve_route import RetrieveRoutePolicy

logger = logging.getLogger(__name__)


class SkillAgent(BaseAgent):
    """Configurable agent that combines any protocol with any runtime policy.

    Args:
        model: The LLM backend.
        library: The skill library.
        protocol: "tool_using", "in_context", or "anthropic_style".
        policy: "NR", "RR", or "PV".
        env_tools: Tool definitions from the dataset.
    """

    def __init__(
        self,
        model: BaseModel,
        library: SkillLibrary,
        protocol: str = "tool_using",
        policy: str = "NR",
        env_tools: list[dict[str, Any]] | None = None,
        full_library: SkillLibrary | None = None,
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
        config: Config | None = None,
        answer_format: str | None = None,
    ) -> None:
        super().__init__(model, library)
        self.env_tools = env_tools or []
        self.full_library = full_library
        self.tool_executors = tool_executors or {}
        self.config = config or Config()
        self.answer_format = answer_format

        self._protocol = self._make_protocol(protocol, model)
        self._policy = self._make_policy(policy, model)

    def solve(
        self,
        task: TaskInstance,
        max_steps: int = 50,
        **kwargs: Any,
    ) -> EpisodeTrace:
        """Run one episode: select skills -> execute protocol -> collect trace."""
        self.model.reset_counters()
        t0 = time.perf_counter()

        selected_skills = self._select_skills(task)
        logger.info(
            "AGENT.solve task=%s protocol=%s policy=%s visible_skills=%s",
            task.task_id,
            self._protocol.__class__.__name__,
            self._policy.__class__.__name__,
            [getattr(s, "skill_id", "?") for s in selected_skills],
        )
        task_library = SkillLibrary()
        for s in selected_skills:
            task_library.add(s)

        raw_steps = self._run_protocol(task, task_library, max_steps)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        entries = self._convert_to_trace_entries(raw_steps)
        # success is False here; the evaluator scores traces against gold
        # via dataset.evaluate_prediction() and overrides this field.
        trace = EpisodeTrace(
            task_id=task.task_id,
            model_id=self.model.model_id,
            entries=entries,
            total_cost=self.model.total_cost,
            total_tokens=self.model.total_tokens,
            success=False,
            final_answer=self._extract_final_answer(raw_steps),
            task_instruction=task.instruction,
            metadata={
                "protocol": self._protocol.__class__.__name__,
                "policy": self._policy.__class__.__name__,
                "elapsed_ms": elapsed_ms,
                "num_skills_selected": len(selected_skills),
            },
        )
        logger.info(
            "AGENT.solve done task=%s steps=%d final_answer=%r",
            task.task_id, len(entries), trace.final_answer,
        )
        return trace

    # -- internals -----------------------------------------------------------

    def _select_skills(self, task: TaskInstance) -> list[Any]:
        """Select skills for this task based on Axis 2 (Retrieval).

        Axis 1 (Visibility) already determined self.library.
        Axis 2 determines how skills are selected at runtime:
          - NR: pass through whatever Axis 1 gave us
          - RR: re-retrieve from the visible library using embeddings
          - PV: planner sees all visible skills (handled in run_episode)
        """
        if isinstance(self._policy, NoRetrievalPolicy):
            # NR: pass through Axis 1 library as-is
            return self.library.list_skills()

        if isinstance(self._policy, RetrieveRoutePolicy):
            # RR: semantic retrieval from the visible library
            return self._policy.select_skills(task, self.library)

        if isinstance(self._policy, PlanVerifyPolicy):
            # PV: planner sees all visible skills
            return self.library.list_skills()

        return self.library.list_skills()

    def _run_protocol(
        self, task: TaskInstance, library: SkillLibrary, max_steps: int
    ) -> list[dict[str, Any]]:
        """Execute the episode using the appropriate protocol.

        PV and RR have their own run_episode() methods.
        NR (and any other policy) uses the standard protocol.
        """
        if isinstance(self._policy, PlanVerifyPolicy):
            return self._policy.run_episode(task, library, max_steps=max_steps)

        if isinstance(self._policy, RetrieveRoutePolicy):
            return self._policy.run_episode(
                task, library, self.env_tools,
                max_steps=max_steps, tool_executors=self.tool_executors,
                protocol=self._protocol,
                answer_format=self.answer_format,
            )

        # NR and fallback: standard protocol execution
        if isinstance(self._protocol, ToolUsingProtocol):
            return self._protocol.run_episode(
                task.instruction, library, self.env_tools,
                max_steps=max_steps, tool_executors=self.tool_executors,
            )
        elif isinstance(self._protocol, InContextProtocol):
            return self._protocol.run_episode(
                task.instruction, library, self.env_tools, max_turns=max_steps,
                answer_format=self.answer_format,
            )
        elif isinstance(self._protocol, GaiaReActProtocol):
            return self._protocol.run_episode(
                task.instruction, library, self.env_tools, max_turns=max_steps,
                answer_format=self.answer_format,
                task_id=task.task_id,
            )
        elif isinstance(self._protocol, ReActProtocol):
            return self._protocol.run_episode(
                task.instruction, library, self.env_tools, max_turns=max_steps,
                answer_format=self.answer_format,
            )
        elif isinstance(self._protocol, AnthropicStyleProtocol):
            return self._protocol.run_episode(
                task.instruction, library, self.env_tools, max_turns=max_steps,
            )
        return []

    @staticmethod
    def _convert_to_trace_entries(raw_steps: list[dict]) -> list[TraceEntry]:
        entries = []
        for s in raw_steps:
            entries.append(
                TraceEntry(
                    step=s.get("step", len(entries)),
                    action=s.get("action", "unknown"),
                    tool_name=s.get("tool_name"),
                    tool_args=s.get("arguments", {}),
                    observation=s.get("observation", s.get("text", "")),
                    success=s.get("status", "passed") != "failed",
                    error=s.get("error"),
                    skill_used=s.get("skill_id"),
                )
            )
        return entries

    @staticmethod
    def _extract_final_answer(raw_steps: list[dict]) -> str | None:
        """Return the agent's final-answer string if present.

        Looks for the last step with action == "final_answer" and returns
        its text. Falls back to the last non-error step's text/observation.
        """
        if not raw_steps:
            return None
        for step in reversed(raw_steps):
            if step.get("action") == "final_answer":
                return str(step.get("text") or step.get("observation") or "")
        last = raw_steps[-1]
        return str(last.get("text") or last.get("observation") or "") or None

    def _make_protocol(self, name: str, model: BaseModel) -> Any:
        react_visibility = self.config.get("experiment.react.visibility", "labels")
        react_verify = bool(self.config.get("experiment.react.enable_verify", True))
        gaia_timeout = int(self.config.get("experiment.gaia_react.python_timeout", 60))
        gaia_max_output = int(self.config.get("experiment.gaia_react.max_exec_output_chars", 12000))
        gaia_trace_dir = self.config.get("experiment.gaia_react.trace_dir", None)
        gaia_domain = self.config.get("experiment.gaia_react.domain", "gaia")
        protocols = {
            "tool_using": lambda: ToolUsingProtocol(model),
            "in_context": lambda: InContextProtocol(model),
            "anthropic_style": lambda: AnthropicStyleProtocol(model),
            "react": lambda: ReActProtocol(
                model,
                visibility=react_visibility,
                enable_verify=react_verify,
            ),
            "gaia_react": lambda: GaiaReActProtocol(
                model,
                visibility=react_visibility,
                enable_verify=react_verify,
                python_timeout=gaia_timeout,
                max_exec_output_chars=gaia_max_output,
                trace_dir=gaia_trace_dir,
                domain=gaia_domain,
            ),
        }
        if name not in protocols:
            raise ValueError(f"Unknown protocol: {name}. Choose from {list(protocols.keys())}")
        return protocols[name]()

    def _make_policy(self, name: str, model: BaseModel) -> Any:
        retrieval_bundle_size = int(self.config.get("experiment.retrieval_bundle_size", 3))
        rr_description_only = bool(
            self.config.get("experiment.retrieval.description_only", False)
        )
        policies = {
            "NR": lambda: NoRetrievalPolicy(),
            "RR": lambda: RetrieveRoutePolicy(
                model,
                retrieval_bundle_size=retrieval_bundle_size,
                description_only=rr_description_only,
            ),
            "PV": lambda: PlanVerifyPolicy(
                model,
                answer_format=self.answer_format,
                retrieval_bundle_size=retrieval_bundle_size,
                description_only=rr_description_only,
            ),
        }
        if name not in policies:
            raise ValueError(f"Unknown policy: {name}. Choose from {list(policies.keys())}")
        return policies[name]()
