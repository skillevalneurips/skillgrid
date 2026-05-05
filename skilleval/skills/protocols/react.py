"""Skill protocol 4: ReAct with toolless actions.

The agent operates in a Thought/Action/Observation loop. Actions are
answered by the harness (library lookup, critic LLM) rather than by
external tools, so this protocol works when the useful operations are
skill lookup and optional answer checking rather than external API calls.

Supported actions:

- ``<fetch_skill>name</fetch_skill>`` — harness returns the skill's
  template text as the observation. Lets the agent lazily load skills
  it needs instead of having every skill stuffed into the system prompt.
- ``<verify>candidate</verify>`` — a critic LLM judges the candidate
  answer against the original task and returns ``correct`` or a
  short critique. Gives the solver a genuine second opinion.
- ``<answer>value</answer>`` — final answer; terminates the episode.

Each action produces an observation that is fed back as a user message,
so every turn carries new information — unlike plain CoT with a
continue-nudge, which just restates the model's own prior reply.
"""

from __future__ import annotations

import logging
import json
import re
from typing import Any, Literal

from skilleval.core.registry import skill_protocol_registry
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)

Visibility = Literal["none", "labels", "full"]

# Ordered list of action tags recognised by the parser.  Answer must come
# first because it is terminal — once the model emits <answer> we stop,
# even if other action tags appear earlier in the reply.
_ACTION_TAGS = ("answer", "fetch_skill", "verify")

_CRITIC_SYSTEM = (
    "You are a strict verifier. Given a task and a candidate answer, "
    "decide whether the answer is correct. Respond with one line only:\n"
    "  CORRECT  — if the answer is right\n"
    "  WRONG: <one-sentence critique>  — if it is wrong\n"
    "Do not solve the task from scratch; just audit the candidate."
)


@skill_protocol_registry.register("react")
class ReActProtocol:
    """Toolless ReAct loop with fetch_skill / verify / answer actions."""

    def __init__(
        self,
        model: BaseModel,
        visibility: Visibility = "labels",
        enable_verify: bool = True,
        max_turns: int = 12,
    ) -> None:
        self.model = model
        self.visibility = visibility
        self.enable_verify = enable_verify
        self.max_turns = max_turns

    # -- prompt construction ------------------------------------------------

    def build_prompt(
        self,
        task_instruction: str,
        skills: list[Any],
        env_tools: list[dict[str, str]],
        answer_format: str | None = None,
    ) -> list[dict[str, str]]:
        allow_verify = self.enable_verify and not _is_json_recommendation_format(answer_format)
        actions_block = self._actions_block(skills, allow_verify=allow_verify)
        catalog_block = self._catalog_block(skills)
        examples_block = self._examples_block(answer_format, allow_verify=allow_verify)
        tool_block = self._tool_block(env_tools)

        system_parts = [
            "You solve tasks using a ReAct loop: Thought, then one "
            "Action tag, then wait for the Observation before continuing.",
            actions_block,
        ]
        if catalog_block:
            system_parts.append(catalog_block)
        if tool_block:
            system_parts.append(tool_block)
        system_parts.append(
            "Format every reply as:\n"
            "Thought: <your reasoning>\n"
            "Action: <one XML action tag>\n"
            "Emit exactly one action tag per reply. Do not answer without "
            "an <answer> tag; do not wrap the tag in code fences."
        )
        if _is_json_recommendation_format(answer_format):
            system_parts.append(
                "For the final answer, put the required JSON object inside "
                "the <answer>...</answer> tag and do not include prose in "
                "that tag."
            )
        # Examples teach <fetch_skill>; only include them when the agent
        # actually has a catalog to fetch from (i.e., not NL / not empty).
        if examples_block and catalog_block:
            system_parts.append(examples_block)
        if answer_format:
            system_parts.append(answer_format)

        system = "\n\n".join(p for p in system_parts if p)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_instruction},
        ]

    # -- episode loop -------------------------------------------------------

    def run_episode(
        self,
        task_instruction: str,
        library: SkillLibrary,
        env_tools: list[dict[str, str]],
        max_turns: int | None = None,
        answer_format: str | None = None,
    ) -> list[dict[str, Any]]:
        turns = max_turns or self.max_turns
        skills = library.list_skills()
        messages = self.build_prompt(
            task_instruction, skills, env_tools, answer_format=answer_format,
        )
        steps: list[dict[str, Any]] = []

        final_answer: str | None = None
        last_text = ""

        for turn in range(turns):
            response = self.model.generate(messages)
            text = response.text or ""
            last_text = text
            steps.append({
                "step": turn,
                "action": "generate",
                "text": text,
                "tokens": response.input_tokens + response.output_tokens,
            })

            kind, payload = self._parse_action(text)

            if kind == "answer":
                final_answer = payload.strip()
                break

            if kind == "fetch_skill":
                observation = self._resolve_fetch(payload, library)
                steps.append({
                    "step": turn,
                    "action": "fetch_skill",
                    "tool_name": payload,
                    "arguments": {"name": payload},
                    "observation": observation,
                })
                library.record_usage(payload)
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation:\n{observation}",
                })
                continue

            if kind == "verify" and allow_verify:
                observation = self._resolve_verify(task_instruction, payload, answer_format)
                steps.append({
                    "step": turn,
                    "action": "verify",
                    "tool_name": "verify",
                    "arguments": {"candidate": payload},
                    "observation": observation,
                })
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation:\n{observation}",
                })
                continue

            # No recognised action — nudge once, then break on repeated failure.
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    "Your last reply had no Action tag. Reply with exactly "
                    f"one of {self._nudge_actions(skills, allow_verify)}."
                ),
            })

        if final_answer is None:
            final_answer = self._salvage_answer(last_text, answer_format)

        steps.append({
            "step": len(steps),
            "action": "final_answer",
            "text": final_answer,
        })
        return steps

    # -- action resolution --------------------------------------------------

    def _resolve_fetch(self, name: str, library: SkillLibrary) -> str:
        skill = library.get(name.strip())
        if skill is None:
            # try loose match by name attribute
            for s in library.list_skills():
                if getattr(s.spec, "name", "") == name.strip():
                    skill = s
                    break
        if skill is None:
            return f"Skill '{name}' not found. Available: " + ", ".join(
                s.skill_id for s in library.list_skills()
            ) or "(library empty)"
        spec = skill.spec
        return (
            f"Skill: {spec.name}\n"
            f"Description: {spec.description}\n"
            f"Template:\n{spec.template}"
        )

    def _resolve_verify(
        self,
        task: str,
        candidate: str,
        answer_format: str | None = None,
    ) -> str:
        candidate_stripped = candidate.strip()
        if not candidate_stripped:
            return (
                "WRONG: You must provide a concrete candidate answer to verify."
            )
        if _is_math_answer_format(answer_format) and not re.search(
            r"-?\d+", candidate_stripped
        ):
            return (
                "WRONG: You must provide a computed numeric answer to verify, "
                "not a description of your approach."
            )
        try:
            resp = self.model.generate([
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": (
                    f"Task:\n{task}\n\n"
                    f"Candidate answer: {candidate_stripped}"
                )},
            ])
            verdict = (resp.text or "").strip().splitlines()[0].strip()
            return verdict or "CORRECT"
        except Exception as exc:
            logger.warning("Verify critic failed: %s", exc)
            return "CORRECT"  # fail-open so we don't kill the episode

    # -- parsing / rendering ------------------------------------------------

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, str]:
        """Return (action_kind, payload) for the last action tag in text.

        Ignores order except that an <answer> tag always wins (terminal).
        """
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            return "answer", answer_match.group(1)

        best_kind: str | None = None
        best_payload = ""
        best_end = -1
        for tag in ("fetch_skill", "verify"):
            for m in re.finditer(
                rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE,
            ):
                if m.end() > best_end:
                    best_end = m.end()
                    best_kind = tag
                    best_payload = m.group(1)
        return best_kind, best_payload

    @staticmethod
    def _salvage_answer(text: str, answer_format: str | None = None) -> str:
        """Best-effort extraction when the loop exits without <answer>."""
        if not text:
            return ""
        json_answer = _extract_json_object(text)
        if json_answer is not None:
            return json.dumps(json_answer, ensure_ascii=False)
        if _is_math_answer_format(answer_format):
            math_answer = _extract_math_answer(text)
            if math_answer is not None:
                return math_answer
        return _strip_action_prefixes(text)

    def _actions_block(self, skills: list[Any], allow_verify: bool = True) -> str:
        lines = ["## Available actions"]
        if self.visibility != "none" and skills:
            lines.append(
                "- <fetch_skill>name</fetch_skill>: load a skill template "
                "by name. The system returns the full template text."
            )
        if allow_verify:
            lines.append(
                "- <verify>candidate</verify>: ask a critic to audit your "
                "candidate answer. The critic returns CORRECT or a short "
                "critique starting with WRONG."
            )
        lines.append(
            "- <answer>value</answer>: submit your final answer. "
            "This ends the episode."
        )
        return "\n".join(lines)

    def _catalog_block(self, skills: list[Any]) -> str:
        if self.visibility == "none" or not skills:
            return ""
        # LB and FL both render as names + descriptions; templates are
        # fetched via <fetch_skill>. LB/FL differ only in how many skills
        # are visible (LB is a random subset applied upstream).
        lines = [
            "## Skill catalog (names and one-line descriptions — "
            "use <fetch_skill> to read the full template)",
        ]
        for s in skills:
            desc = (s.spec.description or "").replace("\n", " ")[:180]
            lines.append(f"- {s.spec.name}: {desc}")
        return "\n".join(lines)

    def _examples_block(
        self,
        answer_format: str | None = None,
        allow_verify: bool = True,
    ) -> str:
        if self.visibility == "none":
            return ""
        if _is_json_recommendation_format(answer_format):
            return (
                "## Example\n"
                "Task: The user liked tense puzzle thrillers and wants similar movies.\n"
                "Thought: I should use a recommendation skill and avoid repeating "
                "titles already mentioned in the conversation.\n"
                "Action: <fetch_skill>rank_contextual_movie_recommendations</fetch_skill>\n"
                "Observation:\n"
                "Skill: rank_contextual_movie_recommendations\n"
                "Description: Rank movies against the conversation context.\n"
                "Template:\n"
                "1. Infer preferences from the dialogue.\n"
                "2. Exclude already-mentioned movies.\n"
                "3. Return ranked JSON recommendations.\n"
                "Thought: I have enough guidance to produce the final JSON.\n"
                "Action: <answer>{\"recommendations\":[{\"title\":\"The Prestige\","
                "\"imdb_id\":\"tt0482571\"}]}</answer>\n\n"
                "## Now solve"
            )
        ex1 = (
            "Problem: How many 3-digit numbers contain at least one 7?\n"
            "Thought: \"At least one\" problems are cleanest via the complement. "
            "Let me load that skill.\n"
            "Action: <fetch_skill>counting_by_complement</fetch_skill>\n"
            "Observation:\n"
            "Skill: counting_by_complement\n"
            "Description: When asked for \"at least one X\", count cases with NO X "
            "and subtract from total.\n"
            "Template:\n"
            "1. Compute the total count.\n"
            "2. Compute the count with NO X.\n"
            "3. Answer = total - complement.\n"
            "Thought: Total 3-digit numbers = 900. With no 7: 8*9*9 = 648. "
            "Answer = 900 - 648 = 252.\n"
            "Action: <answer>252</answer>"
        )
        ex2 = (
            "Problem: In a sequence, each term is twice the previous. If the first "
            "term is 3, what is the 5th term?\n"
            "Thought: Sequence problem. I'll try the arithmetic-sequence skill.\n"
            "Action: <fetch_skill>arithmetic_sequence</fetch_skill>\n"
            "Observation:\n"
            "Skill: arithmetic_sequence\n"
            "Description: Use when consecutive terms differ by a constant amount "
            "(common difference).\n"
            "Template:\n"
            "1. Identify first term a_1 and common difference d.\n"
            "2. n-th term: a_n = a_1 + (n-1)*d.\n"
            "Thought: This needs a constant *difference*, but my terms have a constant "
            "*ratio* (x2). Wrong skill - switching.\n"
            "Action: <fetch_skill>geometric_sequence</fetch_skill>\n"
            "Observation:\n"
            "Skill: geometric_sequence\n"
            "Description: Use when each term is a constant multiple of the previous "
            "(common ratio).\n"
            "Template:\n"
            "1. Identify first term a_1 and common ratio r.\n"
            "2. n-th term: a_n = a_1 * r^(n-1).\n"
            "Thought: a_1 = 3, r = 2, n = 5. a_5 = 3 * 2^4 = 48.\n"
            "Action: <answer>48</answer>"
        )
        parts = ["## Examples", ex1, "---", ex2]
        if allow_verify:
            ex3 = (
                "Problem: What is the sum of the digits of 2^20?\n"
                "Thought: Large power + digit sum. Load the skill.\n"
                "Action: <fetch_skill>large_power_computation</fetch_skill>\n"
                "Observation:\n"
                "Skill: large_power_computation\n"
                "Description: For a^b where the result has several digits. Use "
                "repeated squaring and combine factors.\n"
                "Template:\n"
                "1. Build up powers by squaring: a^2, a^4, a^8, ...\n"
                "2. Combine to reach a^b.\n"
                "3. Extract the integer result.\n"
                "Thought: 2^10 = 1024, 2^20 = 1024^2 = 1,048,576. Digits: "
                "1+0+4+8+5+7+6 = 31.\n"
                "Action: <verify>31</verify>\n"
                "Observation: CORRECT\n"
                "Thought: Verified.\n"
                "Action: <answer>31</answer>"
            )
            parts.extend(["---", ex3])
        parts.append("## Now solve")
        return "\n\n".join(parts)

    @staticmethod
    def _tool_block(env_tools: list[dict[str, str]]) -> str:
        if not env_tools:
            return ""
        lines = ["## Environment tools (informational only — not callable in this protocol)"]
        for t in env_tools:
            lines.append(f"- {t.get('name', '?')}: {t.get('description', '')}")
        return "\n".join(lines)

    @staticmethod
    def _nudge_actions(skills: list[Any], allow_verify: bool) -> str:
        actions = []
        if skills:
            actions.append("<fetch_skill>name</fetch_skill>")
        if allow_verify:
            actions.append("<verify>candidate</verify>")
        actions.append("<answer>value</answer>")
        return ", ".join(actions)


def _is_json_recommendation_format(answer_format: str | None) -> bool:
    return bool(answer_format and "recommendations" in answer_format.lower())


def _is_math_answer_format(answer_format: str | None) -> bool:
    if not answer_format:
        return False
    lower = answer_format.lower()
    return "answer:" in lower or ("integer" in lower and "answer" in lower)


def _extract_math_answer(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"ANSWER:\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_action_prefixes(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = re.sub(
            r"^(?:Action:|Thought:|Observation:|Final answer:)\s*",
            "",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()
