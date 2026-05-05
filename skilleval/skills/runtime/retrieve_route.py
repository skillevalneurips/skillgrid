"""Runtime policy: Retrieve and Route (RR).

The agent retrieves candidate skills from the library (possibly among
distractors) and routes execution to the selected skills.

Step-aware: skills are re-retrieved at each step based on current context.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from typing import Any, Callable

from skilleval.core.registry import runtime_policy_registry
from skilleval.core.types import RuntimePolicy, TaskInstance
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary
from skilleval.skills.protocols.in_context import InContextProtocol
from skilleval.skills.protocols.react import ReActProtocol
from skilleval.skills.protocols.tool_using import ToolUsingProtocol

# Suppress non-critical warnings from sentence-transformers
warnings.filterwarnings("ignore", message=".*CodeCarbonCallback.*")
warnings.filterwarnings("ignore", message=".*Cupy.*")

logger = logging.getLogger(__name__)

# Semantic embeddings using SentenceTransformer for semantic similarity
_EMBEDDING_MODEL = None


def _get_embedding_model():
    """Lazy-load SentenceTransformer model on first use."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            # Suppress non-critical CodeCarbonCallback warnings
            import warnings
            warnings.filterwarnings("ignore", message=".*CodeCarbonCallback.*")

            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError as e:
            logger.error(
                "SentenceTransformer not installed. "
                "Install with: pip install sentence-transformers. Error: %s", e
            )
            raise
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise
    return _EMBEDDING_MODEL


def _semantic_embedding(text: str) -> list[float]:
    """Convert text to a semantic embedding vector using SentenceTransformer."""
    model = _get_embedding_model()
    return model.encode(text, convert_to_numpy=True).tolist()


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import math

    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


@runtime_policy_registry.register("RR")
class RetrieveRoutePolicy:
    """Retrieve relevant skills and inject the best one into the prompt.

    Uses semantic relevance scoring on skill templates to pick the best match.
    The top-1 skill is injected directly into the executor prompt (like PV),
    ensuring the skill is actually used rather than just shown in a catalog.
    """

    def __init__(
        self,
        model: BaseModel,
        retrieval_bundle_size: int = 1,  # Changed from 3 to 1 for forced usage
        distractor_ratio: float = 0.0,
        description_only: bool = False,
    ) -> None:
        self.model = model
        self.retrieval_bundle_size = retrieval_bundle_size
        self.distractor_ratio = distractor_ratio
        self.description_only = description_only

    @property
    def policy_type(self) -> RuntimePolicy:
        return RuntimePolicy.RETRIEVE_ROUTE

    def select_skills(
        self,
        task: TaskInstance,
        library: SkillLibrary,
    ) -> list[Any]:
        """Retrieve the most relevant skill(s) using semantic scoring.

        Scores each skill by semantic similarity between the skill's template
        and the task instruction. Returns top retrieval_bundle_size skills
        (default: 1), optionally padded with random distractors.

        Note: for forced skill usage via prompt injection, use run_episode().
        """
        candidates = self._retrieve_by_relevance(task, library)

        num_distractors = int(len(candidates) * self.distractor_ratio)
        if num_distractors > 0:
            all_skills = library.list_skills()
            candidate_ids = {s.skill_id for s in candidates}
            distractors = [s for s in all_skills if s.skill_id not in candidate_ids]
            import random
            rng = random.Random(hash(task.task_id))
            rng.shuffle(distractors)
            candidates.extend(distractors[:num_distractors])

        return candidates

    def run_episode(
        self,
        task: TaskInstance,
        library: SkillLibrary,
        env_tools: list[dict[str, Any]],
        max_steps: int = 50,
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
        protocol: Any | None = None,
        answer_format: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run episode with step-aware skill retrieval.

        Respects the active protocol: ``tool_using`` presents retrieved
        skills as function-calling tools; ``in_context`` injects them as
        text in the system prompt.
        """
        if isinstance(protocol, ReActProtocol):
            return self._run_episode_react(
                task, library, env_tools, max_steps, protocol,
                answer_format=answer_format,
            )
        if isinstance(protocol, InContextProtocol):
            return self._run_episode_in_context(
                task, library, env_tools, max_steps, protocol,
                answer_format=answer_format,
            )
        return self._run_episode_tool_using(
            task, library, env_tools, max_steps, tool_executors,
        )

    def _run_episode_tool_using(
        self,
        task: TaskInstance,
        library: SkillLibrary,
        env_tools: list[dict[str, Any]],
        max_steps: int,
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """RR with tool-using protocol (original behaviour)."""
        tool_executors = tool_executors or {}
        protocol = ToolUsingProtocol(self.model)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": protocol._system_prompt()},
            {"role": "user", "content": task.instruction},
        ]
        steps: list[dict[str, Any]] = []

        for step_idx in range(max_steps):
            current_context = self._build_context_from_messages(messages)
            context_for_retrieval = f"{task.instruction}\n\nCurrent progress: {current_context}"
            relevant_skills = self._retrieve_by_relevance_text(context_for_retrieval, library)

            env_tools_normalized = protocol._normalize_env_tools(env_tools)
            skill_tools = protocol.skills_to_tools(
                self._build_library_from_skills(relevant_skills)
            )
            all_tools = env_tools_normalized + skill_tools

            if all_tools:
                response = self.model.generate_with_tools(messages, all_tools)
            else:
                response = self.model.generate(messages)

            if not response.tool_calls:
                steps.append({"step": step_idx, "action": "final_answer", "text": response.text})
                break

            for tc in response.tool_calls:
                observation = protocol._execute_tool_call(tc, library, tool_executors)
                steps.append({
                    "step": step_idx,
                    "action": "tool_call",
                    "tool_name": tc["name"],
                    "arguments": tc["arguments"],
                    "observation": observation,
                })
                messages.append({"role": "assistant", "content": json.dumps(tc)})
                messages.append({"role": "user", "content": f"Tool result: {observation}"})

        return steps

    def _run_episode_react(
        self,
        task: TaskInstance,
        library: SkillLibrary,
        env_tools: list[dict[str, Any]],
        max_steps: int,
        protocol: ReActProtocol,
        answer_format: str | None = None,
    ) -> list[dict[str, Any]]:
        """RR with ReAct protocol: retrieve top skill, inject into prompt.

        Retrieves the most relevant skill via embedding similarity and injects
        it directly into the prompt (like PV), forcing the agent to use it
        rather than just showing it in a catalog.
        """
        relevant_skills = self._retrieve_by_relevance(task, library)
        
        if not relevant_skills:
            # No skills available, run without skill injection
            return protocol.run_episode(
                task.instruction, SkillLibrary(), env_tools,
                max_turns=max_steps, answer_format=answer_format,
            )
        
        # Get top skill
        top_skill = relevant_skills[0]
        logger.info(
            "RR: Retrieved top skill '%s' for task",
            top_skill.spec.name,
        )
        
        # Inject skill directly into task instruction (like PV executor)
        skill_injection = (
            f"\n\n**Recommended Skill**: {top_skill.spec.name}\n"
            f"**Description**: {top_skill.spec.description}\n\n"
            f"**Skill Guidance** (use this approach):\n{top_skill.spec.template}\n\n"
            "---\n"
        )
        augmented_instruction = skill_injection + task.instruction
        
        # Run with empty library (skill already injected in prompt)
        return protocol.run_episode(
            augmented_instruction, SkillLibrary(), env_tools,
            max_turns=max_steps, answer_format=answer_format,
        )

    def _run_episode_in_context(
        self,
        task: TaskInstance,
        library: SkillLibrary,
        env_tools: list[dict[str, Any]],
        max_steps: int,
        protocol: InContextProtocol,
        answer_format: str | None = None,
    ) -> list[dict[str, Any]]:
        """RR with in-context protocol: retrieve top skill, inject into prompt.

        Like the ReAct version, retrieves the best skill and injects it
        directly into the task instruction for forced usage.
        """
        relevant_skills = self._retrieve_by_relevance(task, library)
        selected_ids = [skill.skill_id for skill in relevant_skills]
        
        # Prepare augmented instruction with top skill injected
        augmented_instruction = task.instruction
        if relevant_skills:
            top_skill = relevant_skills[0]
            logger.info(
                "RR: Retrieved top skill '%s' for task",
                top_skill.spec.name,
            )
            skill_injection = (
                f"**Recommended Skill**: {top_skill.spec.name}\n"
                f"**Description**: {top_skill.spec.description}\n\n"
                f"**Skill Guidance** (use this approach):\n{top_skill.spec.template}\n\n"
                "---\n\n"
            )
            augmented_instruction = skill_injection + task.instruction

        # Build prompt with empty skill list (skill already in instruction)
        messages = protocol.build_prompt(
            augmented_instruction, [], env_tools,
            answer_format=answer_format,
        )
        steps: list[dict[str, Any]] = []

        for turn in range(max_steps):
            response = self.model.generate(messages)
            steps.append({
                "step": turn,
                "action": "generate",
                "text": response.text,
                "skill_id": selected_ids[0] if selected_ids else None,
                "selected_skill_ids": selected_ids,
                "tokens": response.input_tokens + response.output_tokens,
            })

            if protocol._is_final(response.text):
                break

            messages.append({"role": "assistant", "content": response.text})
            messages.append({
                "role": "user",
                "content": "Continue with the next step, or provide the final answer.",
            })

        return steps

    def _retrieve_by_relevance(
        self,
        task: TaskInstance,
        library: SkillLibrary,
    ) -> list[Any]:
        """Score and rank skills by semantic embedding similarity to the task."""
        return self._retrieve_by_relevance_text(task.instruction, library)

    def _retrieve_by_relevance_text(
        self,
        text: str,
        library: SkillLibrary,
    ) -> list[Any]:
        """Score and rank skills by semantic embedding similarity to arbitrary text.

        Used for both initial task instruction and step-wise context.
        """
        try:
            # Embed the text once
            text_embedding = _semantic_embedding(text)
            scored: list[tuple[float, Any]] = []

            for skill in library.list_skills():
                skill_text = self._skill_retrieval_text(skill)
                skill_embedding = _semantic_embedding(skill_text)

                # Semantic similarity via cosine distance of embeddings
                score = _cosine_similarity(text_embedding, skill_embedding)
                scored.append((score, skill))

            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [s for _, s in scored[:self.retrieval_bundle_size]]
            logger.info("RR selected skills: %s", [s.skill_id for s in selected])
            return selected
        except Exception as exc:
            logger.info("RR embedding retrieval failed; using lexical fallback: %s", exc)
            return self._retrieve_by_keyword(text, library)

    def _skill_retrieval_text(self, skill: Any) -> str:
        if self.description_only:
            return f"{skill.spec.name} {skill.spec.description}"
        return f"{skill.spec.name} {skill.spec.description} {skill.spec.template}"

    def _retrieve_by_keyword(self, text: str, library: SkillLibrary) -> list[Any]:
        query = _tokens(text)
        scored: list[tuple[float, str, Any]] = []
        for skill in library.list_skills():
            skill_tokens = _tokens(self._skill_retrieval_text(skill))
            score = len(query & skill_tokens) / max(len(query), 1)
            scored.append((score, skill.skill_id, skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [s for _, _, s in scored[:self.retrieval_bundle_size]]
        logger.info("RR selected skills: %s", [s.skill_id for s in selected])
        return selected

    @staticmethod
    def _build_context_from_messages(messages: list[dict[str, str]]) -> str:
        """Extract and concatenate recent conversation context.

        Takes the last few non-system messages to summarize current progress.
        """
        context_parts = []
        # Skip system message, take last 10 messages
        for msg in messages[-10:]:
            if msg.get("role") != "system":
                content = msg.get("content", "")
                # Truncate very long content
                if len(content) > 500:
                    content = content[:500] + "..."
                context_parts.append(content)
        return "\n".join(context_parts) if context_parts else ""

    @staticmethod
    def _build_library_from_skills(skills: list[Any]) -> SkillLibrary:
        """Wrap a list of skill objects into a SkillLibrary."""
        lib = SkillLibrary()
        for skill in skills:
            lib.add(skill)
        return lib

    def route(
        self,
        task: TaskInstance,
        candidates: list[Any],
    ) -> list[Any]:
        """Use the LLM to select which skills to apply.

        Override for embedding-based or learned routing.
        """
        if not candidates:
            return []

        skill_list = "\n".join(
            f"{i+1}. {s.spec.name}: {s.spec.description}"
            for i, s in enumerate(candidates)
        )
        prompt = (
            f"Task: {task.instruction}\n\n"
            f"Available skills:\n{skill_list}\n\n"
            "Which skills (by number) are most relevant? "
            "Reply with comma-separated numbers."
        )
        response = self.model.generate([{"role": "user", "content": prompt}])

        selected_indices = self._parse_selection(response.text, len(candidates))
        return [candidates[i] for i in selected_indices]

    @staticmethod
    def _parse_selection(text: str, max_idx: int) -> list[int]:
        import re
        numbers = re.findall(r"\d+", text)
        indices = []
        for n in numbers:
            idx = int(n) - 1
            if 0 <= idx < max_idx:
                indices.append(idx)
        return indices or [0]  # fallback to first if parsing fails


def _tokens(text: str) -> set[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "you", "are",
        "use", "using", "movie", "movies", "film", "films", "recommend",
        "recommendation",
    }
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in stopwords
    }
