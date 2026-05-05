"""Skill protocol 1: Tool-Using (agent orchestrator + direct tool calls).

The agent receives tools as function definitions and decides when/how
to call them.  Skills are expressed as tool-call sequences.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from skilleval.core.registry import skill_protocol_registry
from skilleval.models.base import BaseModel, ModelResponse
from skilleval.skills.library import SkillLibrary

logger = logging.getLogger(__name__)


@skill_protocol_registry.register("tool_using")
class ToolUsingProtocol:
    """Skills are surfaced as callable tools via the model's tool-use API.

    Each skill becomes a tool definition the model can invoke.
    The orchestrator handles sequencing and error handling.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def skills_to_tools(self, library: SkillLibrary) -> list[dict[str, Any]]:
        """Convert skills into OpenAI-style tool definitions."""
        tools = []
        for skill in library.list_skills():
            tools.append({
                "type": "function",
                "function": {
                    "name": skill.skill_id,
                    "description": skill.spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Input for this skill step",
                            },
                        },
                        "required": ["input"],
                    },
                },
            })
        return tools

    def run_episode(
        self,
        task_instruction: str,
        library: SkillLibrary,
        env_tools: list[dict[str, Any]],
        max_steps: int = 20,
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one task episode with tool-using protocol.

        Returns list of steps: [{action, tool_call, observation, ...}].

        ``tool_executors`` maps tool_name -> callable(args_dict) -> observation
        for real environment tools. Tools not in this dict fall back to a
        stub string so the agent can still reason.
        """
        tool_executors = tool_executors or {}
        all_tools = self._normalize_env_tools(env_tools) + self.skills_to_tools(library)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task_instruction},
        ]
        steps: list[dict[str, Any]] = []

        for step_idx in range(max_steps):
            # When there are no tools/skills available, use plain generate()
            # so providers that reject empty tools lists (e.g. OpenAI) don't 400.
            if all_tools:
                response = self.model.generate_with_tools(messages, all_tools)
            else:
                response = self.model.generate(messages)

            if not response.tool_calls:
                steps.append({"step": step_idx, "action": "final_answer", "text": response.text})
                break

            for tc in response.tool_calls:
                observation = self._execute_tool_call(tc, library, tool_executors)
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

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        library: SkillLibrary,
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> str:
        """Execute a tool call.

        Dispatch order:
          1. If the tool name is a registered skill, delegate to the skill.
          2. If a dataset-supplied executor exists for the tool, call it.
          3. Otherwise return a stub string (preserves legacy behavior).
        """
        name = tool_call["name"]
        skill = library.get(name)
        if skill:
            library.record_usage(skill.skill_id)
            result = skill.execute({"arguments": self._parse_arguments(tool_call["arguments"])})
            return str(result.output) if hasattr(result, "output") else str(result)

        executors = tool_executors or {}
        if name in executors:
            try:
                args = self._parse_arguments(tool_call["arguments"])
                return str(executors[name](args))
            except Exception as exc:
                logger.warning("Tool executor %s raised: %s", name, exc)
                return f"[Tool {name} error: {exc}]"

        return f"[Environment would execute: {name}({tool_call['arguments']})]"

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"input": arguments}
            except json.JSONDecodeError:
                return {"input": arguments}
        return {"input": str(arguments)}

    @staticmethod
    def _normalize_env_tools(env_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize dataset tool specs into OpenAI-compatible function tools."""
        normalized: list[dict[str, Any]] = []
        for tool in env_tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and "function" in tool:
                normalized.append(tool)
                continue
            if "name" not in tool:
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {
                                    "input": {
                                        "type": "string",
                                        "description": "Tool input payload",
                                    }
                                },
                                "required": ["input"],
                            },
                        ),
                    },
                }
            )
        return normalized

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a capable agent. Use the available tools to solve the task. "
            "Call tools as needed. When you have the final answer, respond directly."
        )
