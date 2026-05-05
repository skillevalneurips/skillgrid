"""GAIA skill-loop protocol.

Mirrors gaia/scripts/run_gaia_skills.py SkillLoopAgent exactly:

1. _select_action  — separate LLM call: pick one skill by name or "final"
2. _generate_code  — separate LLM call: write Python given the SKILL.md guide
3. _run_code       — execute in subprocess, capture stdout/stderr
4. Repeat until "final" or turn budget exhausted
5. _generate_final — synthesize final answer from all collected evidence

Skills come from the SkillLibrary (populated from gaia/skills/*/SKILL.md),
so Visibility / Retrieval / Evolution axes work unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from skilleval.core.registry import skill_protocol_registry
from skilleval.models.base import BaseModel
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)

# Default prompts (GAIA-flavoured); override via constructor for other datasets.
_DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent assistant helping with the GAIA benchmark. "
    "For each question, reasoning steps are helpful, but you MUST end your "
    "response with the final answer in the format: FINAL ANSWER: [answer]"
)

_DEFAULT_ACTION_SELECTOR_SYSTEM = (
    "You are a GAIA task controller. Decide the next action.\n"
    "Output exactly ONE token from the allowed list.\n"
    "Do not output anything else — no punctuation, no explanation."
)

_DEFAULT_CODE_GENERATOR_SYSTEM = (
    "You write Python code to solve GAIA questions using a skill guide.\n"
    "Return ONLY runnable Python source code.\n"
    "No markdown fences, no backticks, no explanation.\n"
    "Use print() for every value you want to see in the output."
)

# WebWalkerQA-specific prompts.
_WEBWALKER_SYSTEM_PROMPT = (
    "You are an intelligent web navigation agent helping with WebWalkerQA tasks. "
    "Each task gives you a root URL and a question. You must navigate the site "
    "by crawling pages and following links to find the answer. "
    "You MUST end your response with the final answer in the format: FINAL ANSWER: [answer]"
)

_WEBWALKER_ACTION_SELECTOR_SYSTEM = (
    "You are a WebWalkerQA task controller. Decide the next action.\n"
    "Output exactly ONE token from the allowed list.\n"
    "Do not output anything else — no punctuation, no explanation."
)

_WEBWALKER_CODE_GENERATOR_SYSTEM = (
    "You write Python code to navigate websites and answer WebWalkerQA questions "
    "using a skill guide. Start from the root URL provided in the task.\n"
    "Return ONLY runnable Python source code.\n"
    "No markdown fences, no backticks, no explanation.\n"
    "Use print() for every value you want to see in the output."
)

DOMAIN_PROMPTS = {
    "gaia": (
        _DEFAULT_SYSTEM_PROMPT,
        _DEFAULT_ACTION_SELECTOR_SYSTEM,
        _DEFAULT_CODE_GENERATOR_SYSTEM,
    ),
    "webwalkerqa": (
        _WEBWALKER_SYSTEM_PROMPT,
        _WEBWALKER_ACTION_SELECTOR_SYSTEM,
        _WEBWALKER_CODE_GENERATOR_SYSTEM,
    ),
}


@skill_protocol_registry.register("gaia_react")
class GaiaReActProtocol:
    """Skill-loop protocol matching run_gaia_skills.SkillLoopAgent."""

    def __init__(
        self,
        model: BaseModel,
        visibility: str = "labels",
        enable_verify: bool = True,
        max_turns: int = 16,
        python_timeout: int = 60,
        max_exec_output_chars: int = 12000,
        trace_dir: str | Path | None = None,
        domain: str = "gaia",
    ) -> None:
        self.model = model
        self.visibility = visibility
        self.enable_verify = enable_verify
        self.max_turns = max_turns
        self.python_timeout = python_timeout
        self.max_exec_output_chars = max_exec_output_chars
        self.trace_dir = Path(trace_dir) if trace_dir else None
        prompts = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["gaia"])
        self._system_prompt = prompts[0]
        self._action_selector_system = prompts[1]
        self._code_generator_system = prompts[2]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_episode(
        self,
        task_instruction: str,
        library: SkillLibrary,
        env_tools: list[dict[str, str]],   # unused — skills come from library
        max_turns: int | None = None,
        answer_format: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        turns = max_turns or self.max_turns
        skills = library.list_skills()

        run_log: list[str] = []
        reasoning_turns: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        final_answer: str | None = None

        for i in range(1, turns + 1):
            action = self._select_action(task_instruction, skills, run_log, i, turns)

            reasoning_turn: dict[str, Any] = {"turn": i, "action_kind": action}

            if action == "final":
                final_response = self._generate_final(
                    task_instruction, run_log, answer_format,
                )
                final_answer = self._extract_answer(final_response)
                steps.append({
                    "step": len(steps), "action": "final_answer", "text": final_answer,
                })
                reasoning_turn["final_response"] = final_response
                reasoning_turns.append(reasoning_turn)
                break

            # Look up the skill; fall back to partial name match
            skill = self._resolve_skill(action, skills)
            if skill is None:
                obs = (
                    f"status: error\nstdout:\n[empty]\n"
                    f"stderr:\nSkill '{action}' not found in library. "
                    f"Available: {', '.join(s.spec.name for s in skills) or '(none)'}"
                )
                run_log.append(f"action: {action}\n{obs}")
                steps.append({
                    "step": len(steps), "action": "skill_error",
                    "tool_name": action, "observation": obs,
                })
                reasoning_turn["observation"] = obs
                reasoning_turns.append(reasoning_turn)
                continue

            skill_text = skill.spec.template
            reasoning_turn["skill_name"] = skill.spec.name
            reasoning_turn["skill_content_preview"] = skill_text[:100]

            code, code_messages = self._generate_code(
                task_instruction, skill.spec.name, skill_text, run_log,
            )
            reasoning_turn["input_messages"] = code_messages
            reasoning_turn["code"] = code

            if not code.strip():
                obs = (
                    "status: error\nstdout:\n[empty]\n"
                    "stderr:\nNo executable Python code was produced."
                )
            else:
                obs = self._run_code(code)

            reasoning_turn["observation"] = obs
            run_log.append(f"action: {skill.spec.name}\n{obs}")
            steps.append({
                "step": len(steps),
                "action": "execute_skill",
                "tool_name": skill.spec.name,
                "arguments": {"code": code[:500]},
                "observation": obs,
            })
            reasoning_turns.append(reasoning_turn)

        else:
            # Budget exhausted without "final" — force finalization
            final_response = self._generate_final(
                task_instruction, run_log, answer_format,
            )
            final_answer = self._extract_answer(final_response)
            steps.append({
                "step": len(steps), "action": "final_answer", "text": final_answer,
            })
            reasoning_turns.append({
                "turn": turns + 1,
                "action_kind": "forced_final",
                "final_response": final_response,
            })

        self._write_reasoning_trace(
            task_id=task_id or "unknown",
            task_instruction=task_instruction,
            final_answer=final_answer,
            reasoning_turns=reasoning_turns,
        )
        return steps

    # ------------------------------------------------------------------
    # Three-phase LLM calls (mirrors SkillLoopAgent exactly)
    # ------------------------------------------------------------------

    def _select_action(
        self,
        question: str,
        skills: list[Any],
        run_log: list[str],
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Phase 1: ask the model which skill to use next, or 'final'."""
        skill_names = [s.spec.name for s in skills]
        allowed = skill_names + ["final"]
        allowed_str = ", ".join(allowed) if allowed else "final"

        history_text = "\n\n".join(run_log[-3:]) if run_log else "No prior skill runs."
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self._action_selector_system}\n"
                    f"Allowed actions: {allowed_str}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"GAIA Question:\n{question}\n\n"
                    f"Iteration: {iteration}/{max_iterations}\n"
                    f"Recent execution feedback:\n{history_text}\n\n"
                    f"Reply with exactly one token from: {allowed_str}"
                ),
            },
        ]
        raw = self._chat(messages)
        action = self._parse_action_token(raw, skill_names)
        if action:
            return action

        # Repair attempt
        repair = [
            {"role": "system", "content": f"Output exactly one token from: {allowed_str}"},
            {"role": "user", "content": f"Your prior reply was invalid: {raw!r}\nOutput exactly one token now."},
        ]
        raw2 = self._chat(repair)
        action2 = self._parse_action_token(raw2, skill_names)
        return action2 if action2 else "final"

    def _generate_code(
        self,
        question: str,
        skill_name: str,
        skill_text: str,
        run_log: list[str],
    ) -> tuple[str, list[dict[str, str]]]:
        """Phase 2: generate Python code using the skill guide."""
        history_text = "\n\n".join(run_log[-3:]) if run_log else "No prior skill runs."
        messages = [
            {"role": "system", "content": self._code_generator_system},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Selected skill: {skill_name}\n\n"
                    f"Skill guide (SKILL.md):\n{skill_text}\n\n"
                    f"Execution feedback from previous iterations:\n{history_text}\n\n"
                    "Now return Python code only."
                ),
            },
        ]
        raw = self._chat(messages)
        return self._extract_python_code(raw), messages

    def _generate_final(
        self,
        question: str,
        run_log: list[str],
        answer_format: str | None,
    ) -> str:
        """Phase 3: synthesize final answer from all evidence."""
        history_text = "\n\n".join(run_log) if run_log else "No tool outputs were collected."
        system = self._system_prompt
        if answer_format:
            system = f"{system}\n\n{answer_format}"
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Evidence from skill executions:\n{history_text}\n\n"
                    "Provide the answer now. End with: FINAL ANSWER: [answer]"
                ),
            },
        ]
        response = self._chat(messages)
        if not re.search(r"final\s*answer\s*[:\-]", response, flags=re.IGNORECASE):
            response = f"{response.strip()}\n\nFINAL ANSWER: {response.strip()}"
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, str]]) -> str:
        response = self.model.generate(messages)
        return (response.text or "").strip()

    def _parse_action_token(self, raw: str, skill_names: list[str]) -> str:
        text = (raw or "").strip().lower()
        first = re.match(r"([a-z_]+)", text)
        token = first.group(1) if first else ""
        if token == "final":
            return "final"
        # Match against skill names (exact, then prefix)
        for name in skill_names:
            if token == name.lower():
                return name
        for name in skill_names:
            if token.startswith(name.lower()) or name.lower().startswith(token):
                return name
        return ""

    def _resolve_skill(self, action: str, skills: list[Any]) -> Any | None:
        """Find skill by id or name, case-insensitive."""
        action_lower = action.lower()
        for s in skills:
            if s.skill_id.lower() == action_lower or s.spec.name.lower() == action_lower:
                return s
        return None

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Pull value after 'FINAL ANSWER:' or <answer> tag."""
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"final\s*answer\s*[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().splitlines()[0].strip()
        # Last non-empty line as fallback
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                return line
        return text.strip()

    @staticmethod
    def _extract_python_code(raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            candidates = [b.strip() for b in blocks if b.strip()]
            if candidates:
                return max(candidates, key=len)
        text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _run_code(self, code: str) -> str:
        """Execute Python in a subprocess, return formatted stdout/stderr."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as f:
            tmp_path = f.name
            f.write(code)

        try:
            compile(code, tmp_path, "exec")
        except SyntaxError as e:
            os.unlink(tmp_path)
            return (
                f"status: error\nexit_code: 1\n"
                f"stdout:\n[empty]\n"
                f"stderr:\nSyntaxError: {e.msg} (line {e.lineno})\n"
                f"hint: Regenerate complete, valid Python code."
            )

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.python_timeout,
                cwd=str(Path.cwd()),
            )
            status = "success" if result.returncode == 0 else "error"
            stdout = self._truncate(result.stdout or "")
            stderr = self._truncate(result.stderr or "")
            return (
                f"status: {status}\nexit_code: {result.returncode}\n"
                f"stdout:\n{stdout if stdout else '[empty]'}\n"
                f"stderr:\n{stderr if stderr else '[empty]'}"
            )
        except subprocess.TimeoutExpired as e:
            stdout = self._truncate((e.stdout or "").strip()) if e.stdout else "[empty]"
            stderr = self._truncate((e.stderr or "").strip()) if e.stderr else "[empty]"
            return (
                f"status: timeout\nexit_code: timeout_after_{self.python_timeout}s\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        except Exception as e:
            return (
                f"status: runner_error\nexit_code: runner_error\n"
                f"stdout:\n[empty]\nstderr:\n{e}"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_exec_output_chars:
            return text
        return text[: self.max_exec_output_chars] + "\n...[truncated]..."

    def _write_reasoning_trace(
        self,
        task_id: str,
        task_instruction: str,
        final_answer: str | None,
        reasoning_turns: list[dict[str, Any]],
    ) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace = {
            "task_id": task_id,
            "task_instruction": task_instruction,
            "model_id": getattr(self.model, "model_id", "unknown"),
            "num_turns": len(reasoning_turns),
            "final_answer": final_answer,
            "turns": reasoning_turns,
        }
        out_path = self.trace_dir / f"{task_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        logger.debug("Reasoning trace written to %s", out_path)
