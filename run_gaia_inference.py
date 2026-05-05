import os
import re
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
from typing import Any, Dict, Optional

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


def _extract_intermediate_output(content: str) -> Optional[str]:
    """Return model output before the FINAL ANSWER marker, if present."""
    text = (content or "").strip()
    if not text:
        return None
    m = re.search(r"final\s*answer\s*[:\-]?\s*", text, flags=re.IGNORECASE)
    if not m:
        return text
    intermediate = text[: m.start()].strip()
    return intermediate or None


class MemoryTracker:
    """Persist per-question memory records to a JSON array on disk."""

    def __init__(self, path: str):
        self.path = path
        self.records = []
        self._flush()

    def add(
        self,
        *,
        question_id: str,
        question: str,
        intermediate_output: Optional[str],
        final_answer: Optional[str],
    ) -> None:
        self.records.append(
            {
                "question_id": question_id,
                "question": question,
                "intermediate_output": intermediate_output,
                "final_answer": final_answer,
            }
        )
        self._flush()

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)


class MemoryTrackingGAIABenchmark(GAIABenchmark):
    def __init__(
        self,
        data_dir: str,
        save_to: str,
        memory_tracker: MemoryTracker,
        **kwargs,
    ):
        super().__init__(data_dir, save_to, **kwargs)
        self.memory_tracker = memory_tracker

    def _process_result(
        self,
        agent,
        task: Dict[str, Any],
        result: Any,
        file_obj: Any,
    ) -> None:
        super()._process_result(agent, task, result, file_obj)
        raw_content = result.msgs[0].content if result and result.msgs else ""
        self.memory_tracker.add(
            question_id=str(task["task_id"]),
            question=task["Question"],
            intermediate_output=_extract_intermediate_output(raw_content),
            final_answer=self.get_final_answer(raw_content),
        )

    def _handle_error(self, task: Dict[str, Any], error: Exception, file_obj: Any) -> None:
        super()._handle_error(task, error, file_obj)
        self.memory_tracker.add(
            question_id=str(task["task_id"]),
            question=task["Question"],
            intermediate_output=None,
            final_answer=None,
        )


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
        memory_tracker: Optional[MemoryTracker] = None,
        **kwargs,
    ):
        super().__init__(data_dir, save_to, **kwargs)
        self.gpt_eval_model = gpt_eval_model
        self.memory_tracker = memory_tracker

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

        if self.memory_tracker is not None:
            self.memory_tracker.add(
                question_id=str(task["task_id"]),
                question=task["Question"],
                intermediate_output=_extract_intermediate_output(result.msgs[0].content),
                final_answer=model_answer,
            )

    def _handle_error(self, task: Dict[str, Any], error: Exception, file_obj: Any) -> None:
        super()._handle_error(task, error, file_obj)
        if self.memory_tracker is not None:
            self.memory_tracker.add(
                question_id=str(task["task_id"]),
                question=task["Question"],
                intermediate_output=None,
                final_answer=None,
            )


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
                raise RuntimeError(
                    f"LLM provider error (provider={self.provider}, model={self.model_id}): {e}"
                ) from e

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

    if subset is not None and subset <= 0:
        raise ValueError("--subset must be a positive integer.")

    if gpt_eval and not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise ValueError("--gpt_eval requires OPENAI_API_KEY to be set (e.g. via .env).")

    # Output path derived from provider + model (+ subset if specified), stored under outputs/
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    subset_suffix = f"_subset{subset}" if subset is not None else ""
    output_file = os.path.join(
        output_dir,
        f"gaia_{_sanitize_for_filename(provider)}_{_sanitize_for_filename(model_name)}_inference{subset_suffix}.jsonl",
    )
    memory_file = os.path.join(output_dir, "memories.json")
    memory_tracker = MemoryTracker(memory_file)

    print("=" * 60)
    print("GAIA Benchmark (Direct Model Inference, No Tools)")
    print("=" * 60)

    gaia_data_dir = os.path.join(PROJECT_ROOT, "gaia_data")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Data directory: {gaia_data_dir}")
    print(f"Output file: {output_file}")
    print(f"Memory file: {memory_file}")
    if subset is not None:
        print(f"Subset: {subset}")
    if gpt_eval:
        print(f"GPT Eval: enabled (model: {gpt_eval_model})")
    print("=" * 60)

    # Initialize benchmark (use GPT-eval subclass if --gpt_eval is set)
    if gpt_eval:
        benchmark = GPTEvalGAIABenchmark(
            data_dir=gaia_data_dir,
            save_to=output_file,
            gpt_eval_model=gpt_eval_model,
            memory_tracker=memory_tracker,
        )
    else:
        benchmark = MemoryTrackingGAIABenchmark(
            data_dir=gaia_data_dir,
            save_to=output_file,
            memory_tracker=memory_tracker,
        )

    print("\nLoading GAIA dataset...")
    benchmark.load()

    print(f"\nInitializing direct inference agent ({provider}) with model: {model_name}")
    agent = DirectInferenceAgent(
        provider=provider,
        model_id=model_name,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
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
