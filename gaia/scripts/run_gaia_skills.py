import os
import re
import subprocess
import sys

# Get the project root (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from camel.messages import BaseMessage

from utils.gaia import GAIABenchmark
from utils.llm_engine import LLMEngineError, create_engine
from utils.test_openai import run_openai


# GPT judge prompt for semantic answer comparison
GPT_JUDGE_SYSTEM_PROMPT = """You are a strict answer evaluator. Your task is to determine if the model's answer is semantically equivalent to the ground truth answer.

Rules:
- The model answer may contain extra explanation or context, but the core answer must match the ground truth.
- Minor differences in formatting, punctuation, or phrasing are acceptable if the meaning is the same.
- Numbers must be equivalent (e.g., "1000" and "1,000" are the same).
- For names, slight variations are acceptable if they clearly refer to the same entity.
- If the model answer contains the correct answer along with additional (correct) context, mark it as correct.
- If the model answer is wrong, incomplete, or contradicts the ground truth, mark it as incorrect.

You must respond with ONLY one word: "correct" or "incorrect". No explanation, no punctuation, no other text."""


def _gpt_judge(question: str, model_answer: str, ground_truth: str, gpt_model: str) -> dict:
    user_prompt = f"""Question: {question}

    Model Answer: {model_answer}

    Ground Truth: {ground_truth}

    Is the model answer correct? output only one word: "correct" or "incorrect"."""

    raw_response = run_openai(GPT_JUDGE_SYSTEM_PROMPT, gpt_model, user_prompt)

    raw_lower = raw_response.strip().lower()
    if "correct" in raw_lower and "incorrect" not in raw_lower:
        verdict = "correct"
        score = 1
    elif "incorrect" in raw_lower:
        verdict = "incorrect"
        score = 0
    else:
        verdict = "unknown"
        score = 0

    return {
        "verdict": verdict,
        "raw": raw_response,
        "score": score,
        "model": gpt_model,
    }


class GPTEvalGAIABenchmark(GAIABenchmark):
    """
    GAIABenchmark subclass that uses GPT (via run_openai) for scoring
    instead of exact-match comparison.
    """

    def __init__(
        self,
        data_dir: str,
        save_to: str,
        gpt_eval_model: str = "gpt-4o-mini",
        **kwargs,
    ):
        super().__init__(data_dir, save_to, **kwargs)
        self.gpt_eval_model = gpt_eval_model

    def _process_result(
        self,
        agent,
        task: Dict[str, Any],
        result: Any,
        file_obj: Any,
    ) -> None:
        model_answer = self.get_final_answer(result.msgs[0].content)
        final_answer = task["Final answer"]
        question = task["Question"]

        gpt_result = _gpt_judge(question, model_answer, final_answer, self.gpt_eval_model)
        score = gpt_result["score"]

        tool_calls = result.info.get("tool_calls", [])

        result_data = {
            "task_id": task["task_id"],
            "question": task["Question"],
            "level": task["Level"],
            "model_answer": model_answer,
            "ground_truth": final_answer,
            "tool_calls": [tool.model_dump() for tool in tool_calls] if tool_calls else [],
            "error": None,
            "score": int(score),
            "history": agent.memory.get_context(),
            "gpt_eval": {
                "model": gpt_result["model"],
                "verdict": gpt_result["verdict"],
                "raw": gpt_result["raw"],
            },
        }
        self._results.append(result_data)
        file_obj.write(json.dumps(result_data, indent=2, ensure_ascii=False) + "\n")
        file_obj.flush()


class AgentResult:
    """Wrapper for agent results to match GAIA benchmark interface."""

    def __init__(self, content: str, info: Optional[Dict] = None):
        self.msgs = [BaseMessage.make_assistant_message(role_name="Assistant", content=content)]
        self.info = info or {}


class DirectInferenceAgent:
    """
    Minimal GAIA agent that does direct chat completion (no tools).

    Provider modes:
    - hf: local Transformers model
    - openai: OpenAI Chat Completions via `utils.llm_engine`
    - grok: xAI/Grok Chat Completions via `utils.llm_engine`
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        self.provider = (provider or "").strip().lower()
        self.model_id = str(model_id)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self._system_prompt_msg = {"role": "system", "content": system_prompt}

        self._engine = None
        self._tokenizer = None
        self._model = None

        if self.provider in ("openai", "grok", "xai"):
            # Normalize xai -> grok for llm_engine
            p = "grok" if self.provider == "xai" else self.provider
            self._engine = create_engine(p, model=self.model_id)
        elif self.provider == "hf":
            self._init_hf_model(self.model_id)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Expected 'hf', 'openai', or 'grok'.")

        self.reset()

    def _init_hf_model(self, model_id: str) -> None:
        # Lazy imports so OpenAI/Grok runs don't require torch/transformers installed.
        import transformers  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        print(f"Loading HF model {model_id}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except ValueError as e:
            msg = str(e)
            if "model type `qwen3`" in msg or "KeyError: 'qwen3'" in msg:
                raise RuntimeError(
                    "Failed to load a Qwen3 model because your current "
                    f"Transformers build does not recognize `model_type=qwen3`.\n"
                    f"- transformers=={transformers.__version__}\n\n"
                    "Fix options:\n"
                    "- Upgrade: `pip install -U --pre transformers accelerate`\n"
                    "- Or use a Qwen2.5 model id (e.g. "
                    "`Qwen/Qwen2.5-7B-Instruct`) instead of Qwen3.\n"
                ) from e
            raise

    def step(self, input_message: BaseMessage) -> AgentResult:
        self.history.append({"role": "user", "content": input_message.content})

        if self.provider == "hf":
            # Local Transformers inference
            import torch  # type: ignore

            assert self._tokenizer is not None and self._model is not None

            text = self._tokenizer.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = (response or "").strip()
        else:
            # OpenAI/Grok inference via llm_engine
            assert self._engine is not None
            try:
                response = self._engine.chat(
                    self.history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                ).strip()
            except LLMEngineError as e:
                response = f"ERROR: {e}"

        self.history.append({"role": "assistant", "content": response})
        return AgentResult(
            content=response,
            info={
                "provider": self.provider,
                "model": self.model_id,
                "tool_calls": [],
            },
        )

    def reset(self):
        self.history = [self._system_prompt_msg]

    @property
    def memory(self):
        return self

    def get_context(self):
        return self.history


SKILLS_ROOT = os.path.join(PROJECT_ROOT, "skills")
INTERMEDIATE_DIR = os.path.join(SCRIPT_DIR, "intermediate")
INTERMEDIATE_FILE = os.path.join(INTERMEDIATE_DIR, "intermediate.py")
AVAILABLE_SKILLS = {
    "search": os.path.join(SKILLS_ROOT, "search", "SKILL.md"),
}


class SkillLoopAgent:
    """
    GAIA agent with an iterative skill-orchestration loop.

    Each iteration:
    - choose one action token: "search" or "final"
    - for "search", generate Python code using the skill guide, run it via
      scripts/intermediate/intermediate.py, and feed runtime output back
    - for "final", return a response that includes "FINAL ANSWER: ..."
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_skill_iters: int = 6,
        python_timeout: int = 30,
        max_exec_output_chars: int = 12000,
    ):
        self.provider = (provider or "").strip().lower()
        self.model_id = str(model_id)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_skill_iters = int(max_skill_iters)
        self.python_timeout = int(python_timeout)
        self.max_exec_output_chars = int(max_exec_output_chars)
        self._system_prompt = system_prompt

        self._engine = None
        self._tokenizer = None
        self._model = None

        if self.provider in ("openai", "grok", "xai"):
            p = "grok" if self.provider == "xai" else self.provider
            self._engine = create_engine(p, model=self.model_id)
        elif self.provider == "hf":
            self._init_hf_model(self.model_id)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Expected 'hf', 'openai', or 'grok'.")

        self._validate_skills()
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        self.reset()

    def _init_hf_model(self, model_id: str) -> None:
        # Lazy imports so OpenAI/Grok runs don't require torch/transformers installed.
        import transformers  # type: ignore
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        print(f"Loading HF model {model_id}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except ValueError as e:
            msg = str(e)
            if "model type `qwen3`" in msg or "KeyError: 'qwen3'" in msg:
                raise RuntimeError(
                    "Failed to load a Qwen3 model because your current "
                    f"Transformers build does not recognize `model_type=qwen3`.\n"
                    f"- transformers=={transformers.__version__}\n\n"
                    "Fix options:\n"
                    "- Upgrade: `pip install -U --pre transformers accelerate`\n"
                    "- Or use a Qwen2.5 model id (e.g. "
                    "`Qwen/Qwen2.5-7B-Instruct`) instead of Qwen3.\n"
                ) from e
            raise

    def _validate_skills(self) -> None:
        missing = [name for name, path in AVAILABLE_SKILLS.items() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing skill files for: {', '.join(missing)}. "
                f"Expected under {SKILLS_ROOT}"
            )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        if self.provider == "hf":
            import torch  # type: ignore

            assert self._tokenizer is not None and self._model is not None
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return (response or "").strip()

        assert self._engine is not None
        try:
            return self._engine.chat(
                messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
            ).strip()
        except LLMEngineError as e:
            return f"ERROR: {e}"

    def _record_turn(self, messages: List[Dict[str, str]], response: str) -> None:
        # Store each interaction pair so GAIA output includes full loop context.
        self.history.extend(messages)
        self.history.append({"role": "assistant", "content": response})

    def _extract_action(self, raw: str) -> str:
        text = (raw or "").strip().lower()
        m = re.match(r"([a-z]+)", text)
        if not m:
            return ""
        token = m.group(1)
        if token in ("search", "final"):
            return token
        return ""

    def _extract_python_code(self, raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""

        # Prefer fenced blocks if present.
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            candidates = [b.strip() for b in blocks if b.strip()]
            if candidates:
                return max(candidates, key=len)

        # Remove single-line fences if the model still wrapped entire content.
        text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        return text.strip()

    def _load_skill_text(self, skill_name: str) -> str:
        path = AVAILABLE_SKILLS.get(skill_name)
        if not path:
            raise ValueError(f"Unknown skill selected: {skill_name!r}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_exec_output_chars:
            return text
        return text[: self.max_exec_output_chars] + "\n...[truncated]..."

    def _run_intermediate(self, code: str) -> str:
        # Validate syntax before writing/executing
        try:
            compile(code, INTERMEDIATE_FILE, "exec")
        except SyntaxError as e:
            return (
                f"status: error\n"
                f"exit_code: 1\n"
                f"stdout:\n[empty]\n"
                f"stderr:\nSyntaxError: {e.msg} (line {e.lineno})\n"
                f"hint: The generated code has a syntax error — likely truncated. "
                f"Please regenerate complete, valid Python code."
            )
        with open(INTERMEDIATE_FILE, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, INTERMEDIATE_FILE],
                capture_output=True,
                text=True,
                timeout=self.python_timeout,
                cwd=PROJECT_ROOT,
            )
            status = "success" if result.returncode == 0 else "error"
            stdout = self._truncate(result.stdout or "")
            stderr = self._truncate(result.stderr or "")
            return (
                f"status: {status}\n"
                f"exit_code: {result.returncode}\n"
                f"stdout:\n{stdout if stdout else '[empty]'}\n"
                f"stderr:\n{stderr if stderr else '[empty]'}"
            )
        except subprocess.TimeoutExpired as e:
            stdout = self._truncate((e.stdout or "").strip()) if e.stdout else "[empty]"
            stderr = self._truncate((e.stderr or "").strip()) if e.stderr else "[empty]"
            return (
                "status: timeout\n"
                f"exit_code: timeout_after_{self.python_timeout}s\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        except Exception as e:
            return (
                "status: runner_error\n"
                "exit_code: runner_error\n"
                "stdout:\n[empty]\n"
                f"stderr:\n{e}"
            )

    def _select_action(self, question: str, run_log: List[str], iteration: int) -> str:
        history_text = "\n\n".join(run_log[-3:]) if run_log else "No prior skill runs."
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a GAIA task controller. Decide the next action.\n"
                    "Allowed actions: search, final.\n"
                    "Output exactly ONE lowercase word: search or final.\n"
                    "Do not output anything else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"GAIA Question:\n{question}\n\n"
                    f"Iteration: {iteration}/{self.max_skill_iters}\n"
                    f"Recent execution feedback:\n{history_text}\n\n"
                    "Reply with only one word."
                ),
            },
        ]
        raw = self._chat(messages)
        self._record_turn(messages, raw)
        action = self._extract_action(raw)
        if action:
            return action

        repair_messages = [
            {
                "role": "system",
                "content": "Output exactly one word: search or final.",
            },
            {
                "role": "user",
                "content": f"Your prior reply was invalid:\n{raw}\n\nOutput exactly one word now.",
            },
        ]
        repaired = self._chat(repair_messages)
        self._record_turn(repair_messages, repaired)
        repaired_action = self._extract_action(repaired)
        if repaired_action:
            return repaired_action
        return "final"

    def _generate_search_code(self, question: str, run_log: List[str], skill_text: str) -> str:
        history_text = "\n\n".join(run_log[-3:]) if run_log else "No prior skill runs."
        messages = [
            {
                "role": "system",
                "content": (
                    "You write Python code to solve GAIA questions via a skill guide.\n"
                    "Return only runnable Python source code.\n"
                    "No markdown, no backticks, no explanation.\n"
                    "Use print() for every output that should be visible."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    "Selected skill: search\n\n"
                    f"Skill guide (SKILL.md):\n{skill_text}\n\n"
                    f"Execution feedback from previous iterations:\n{history_text}\n\n"
                    "Now return Python code only."
                ),
            },
        ]
        raw_code = self._chat(messages)
        self._record_turn(messages, raw_code)
        return self._extract_python_code(raw_code)

    def _generate_final(self, question: str, run_log: List[str]) -> str:
        history_text = "\n\n".join(run_log) if run_log else "No tool outputs were collected."
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self._system_prompt}\n"
                    "You must output a final response ending with: FINAL ANSWER: [answer]"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Evidence from skill executions:\n{history_text}\n\n"
                    "Provide the answer now."
                ),
            },
        ]
        response = self._chat(messages)
        self._record_turn(messages, response)
        if re.search(r"final\s*answer\s*[:\-]", response, flags=re.IGNORECASE):
            return response
        return f"{response.strip()}\n\nFINAL ANSWER: {response.strip()}"

    def step(self, input_message: BaseMessage) -> AgentResult:
        question = input_message.content
        run_log: List[str] = []
        skill_iterations = 0

        for i in range(1, self.max_skill_iters + 1):
            action = self._select_action(question=question, run_log=run_log, iteration=i)
            if action == "final":
                final_response = self._generate_final(question=question, run_log=run_log)
                return AgentResult(
                    content=final_response,
                    info={
                        "provider": self.provider,
                        "model": self.model_id,
                        "tool_calls": [],
                        "skill_iterations": skill_iterations,
                    },
                )

            # Current workflow has one concrete skill: search.
            skill_iterations += 1
            skill_text = self._load_skill_text("search")
            code = self._generate_search_code(question=question, run_log=run_log, skill_text=skill_text)
            if not code.strip():
                exec_feedback = (
                    "action: search\n"
                    "status: invalid_code\n"
                    "exit_code: invalid_code\n"
                    "stdout:\n[empty]\n"
                    "stderr:\nNo executable Python code was produced."
                )
            else:
                exec_result = self._run_intermediate(code)
                exec_feedback = f"action: search\n{exec_result}"

            run_log.append(exec_feedback)

        # Force finalization when iteration budget is exhausted.
        forced_final = self._generate_final(question=question, run_log=run_log)
        return AgentResult(
            content=forced_final,
            info={
                "provider": self.provider,
                "model": self.model_id,
                "tool_calls": [],
                "skill_iterations": skill_iterations,
                "forced_finalization": True,
            },
        )

    def reset(self):
        self.history = []

    @property
    def memory(self):
        return self

    def get_context(self):
        return self.history


DEFAULT_MODEL_NAME = os.getenv("GAIA_MODEL", "Alibaba-NLP/WebSailor-3B")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with direct model inference (no tools).")

    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument("--hf", action="store_true", help="Use local HuggingFace/Transformers model (default).")
    provider_group.add_argument("--openai", action="store_true", help="Use OpenAI via API (requires OPENAI_API_KEY).")
    provider_group.add_argument("--grok", action="store_true", help="Use Grok/xAI via API (requires XAI_API_KEY).")

    parser.add_argument(
        "--model_name",
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name/id. Examples: 'Alibaba-NLP/WebSailor-3B', 'gpt-4o-mini', 'grok-2-latest'.",
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run only the first N tasks after filtering by split/level (e.g. --subset 1).",
    )

    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", "--top-p", type=float, default=0.95)
    parser.add_argument(
        "--max_skill_iters",
        "--max-skill-iters",
        type=int,
        default=6,
        help="Maximum number of skill loop iterations per GAIA task.",
    )
    parser.add_argument(
        "--python_timeout",
        "--python-timeout",
        type=int,
        default=30,
        help="Timeout (seconds) for scripts/intermediate/intermediate.py execution.",
    )
    parser.add_argument(
        "--max_exec_output_chars",
        "--max-exec-output-chars",
        type=int,
        default=12000,
        help="Maximum captured characters for stdout/stderr per intermediate run.",
    )

    parser.add_argument(
        "--gpt_eval",
        action="store_true",
        help="Use GPT (OpenAI) to evaluate answers as correct/incorrect instead of exact-match scoring.",
    )
    parser.add_argument(
        "--gpt_eval_model",
        "--gpt-eval-model",
        default="gpt-4o-mini",
        help="OpenAI model name for GPT-based evaluation (default: gpt-4o-mini).",
    )

    return parser.parse_args()


def _resolve_provider(args: argparse.Namespace) -> str:
    if args.openai:
        return "openai"
    if args.grok:
        return "grok"
    if args.hf:
        return "hf"
    env_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if env_provider in ("openai", "grok", "xai"):
        return "grok" if env_provider == "xai" else env_provider
    return "hf"


def _sanitize_for_filename(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    value = value.replace(os.sep, "_").replace("/", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "unknown"


SYSTEM_PROMPT = (
    "You are an intelligent assistant helping with the GAIA benchmark. "
    "For each question, reasoning steps are helpful, but you MUST end your response "
    "with the final answer in the format: FINAL ANSWER: [answer]"
)


def main():
    args = _parse_args()
    provider = _resolve_provider(args)
    model_name = str(args.model_name)
    subset = args.subset
    gpt_eval = bool(args.gpt_eval)
    gpt_eval_model = str(args.gpt_eval_model)
    max_skill_iters = int(args.max_skill_iters)
    python_timeout = int(args.python_timeout)
    max_exec_output_chars = int(args.max_exec_output_chars)

    if subset is not None and subset <= 0:
        raise ValueError("--subset must be a positive integer.")
    if max_skill_iters <= 0:
        raise ValueError("--max_skill_iters must be a positive integer.")
    if python_timeout <= 0:
        raise ValueError("--python_timeout must be a positive integer.")
    if max_exec_output_chars <= 0:
        raise ValueError("--max_exec_output_chars must be a positive integer.")

    if gpt_eval and not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise ValueError("--gpt_eval requires OPENAI_API_KEY to be set (e.g. via .env).")

    # Output path derived from provider + model (+ subset if specified), stored under outputs/
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    subset_suffix = f"_subset{subset}" if subset is not None else ""
    output_file = os.path.join(
        output_dir,
        f"gaia_{_sanitize_for_filename(provider)}_{_sanitize_for_filename(model_name)}_skills{subset_suffix}.jsonl",
    )

    print("=" * 60)
    print("GAIA Benchmark (Direct Model Inference, No Tools)")
    print("=" * 60)

    gaia_data_dir = os.path.join(PROJECT_ROOT, "gaia_data")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Data directory: {gaia_data_dir}")
    print(f"Output file: {output_file}")
    if subset is not None:
        print(f"Subset: {subset}")
    print(f"Skill loop max iterations: {max_skill_iters}")
    print(f"Intermediate python timeout: {python_timeout}s")
    if gpt_eval:
        print(f"GPT Eval: enabled (model: {gpt_eval_model})")
    print("=" * 60)

    # Initialize benchmark (use GPT-eval subclass if --gpt_eval is set)
    if gpt_eval:
        benchmark = GPTEvalGAIABenchmark(
            data_dir=gaia_data_dir,
            save_to=output_file,
            gpt_eval_model=gpt_eval_model,
        )
    else:
        benchmark = GAIABenchmark(data_dir=gaia_data_dir, save_to=output_file)

    print("\nLoading GAIA dataset...")
    benchmark.load()

    print(f"\nInitializing skill-loop agent ({provider}) with model: {model_name}")
    agent = SkillLoopAgent(
        provider=provider,
        model_id=model_name,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_skill_iters=max_skill_iters,
        python_timeout=python_timeout,
        max_exec_output_chars=max_exec_output_chars,
    )

    print("\nRunning GAIA Benchmark...")
    results = benchmark.run(
        agent=agent,
        on="valid",
        level="all",
        randomize=False,
        subset=subset,
    )

    # Print scores by level
    per_level = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})
    for r in results.get("results", []):
        lvl = r.get("level", "unknown")
        per_level[lvl]["total"] += 1
        per_level[lvl]["correct"] += int(r.get("score", 0) or 0)
        if r.get("error"):
            per_level[lvl]["errors"] += 1

    print(f"\nOverall Score: {results['correct']} / {results['total']}")
    for lvl in sorted([k for k in per_level.keys() if isinstance(k, int)]):
        c = per_level[lvl]["correct"]
        t = per_level[lvl]["total"]
        e = per_level[lvl]["errors"]
        acc = (100.0 * c / t) if t else 0.0
        print(f"Level {lvl}: {c} / {t} ({acc:.1f}%) | errors: {e}")
    if "unknown" in per_level:
        c = per_level["unknown"]["correct"]
        t = per_level["unknown"]["total"]
        e = per_level["unknown"]["errors"]
        acc = (100.0 * c / t) if t else 0.0
        print(f"Level unknown: {c} / {t} ({acc:.1f}%) | errors: {e}")


if __name__ == "__main__":
    main()
