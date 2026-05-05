#!/usr/bin/env python3
"""Run the SkillEval framework on GAIA.

Mirrors the data + judging conventions of ``run_gaia_inference.py`` so the
SkillEval scores are directly comparable to the no-tools baseline. Writes one
self-contained JSON object per question to ``logs/<run>.jsonl``.

Example:
    python experiments/run_gaia_skilleval.py \\
        --provider openai --model_name gpt-4o-mini \\
        --visibility FL --retrieval NR --evolution FR --skill_origin SD \\
        --subset 5 --gpt_eval_model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Project root (parent of experiments/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (parallel to run_gaia_inference.py)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry, model_registry
from skilleval.core.types import (
    EpisodeTrace,
    RuntimePolicy,
    SkillOrigin,
    SkillRetrieval,
    SkillVisibility,
    TaskInstance,
    UpdateStrategy,
)
from skilleval.evaluation.axis_mapping import resolve_initial_library, resolve_retrieval_policy
from skilleval.evaluation.evaluator import Evaluator
from skilleval.skills.library import SkillLibrary
from skilleval.agents.executor import SkillAgent
from skilleval.utils.logging import setup_logging
from experiments.common import MODEL_CONFIG_MAP, warn_compatibility_issues

import skilleval.datasets  # noqa: F401  (registers GAIA)
import skilleval.models  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Copied from run_gaia_inference.py so the SkillEval logs use identical
# system-prompt + judging logic. Kept inline to avoid coupling/imports.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an intelligent assistant helping with the GAIA benchmark. "
    "For each question, reasoning steps are helpful, but you MUST end your response "
    "with the final answer in the format: FINAL ANSWER: [answer]"
)

GPT_JUDGE_SYSTEM_PROMPT = """You are a strict answer evaluator. Your task is to determine if the model's answer is semantically equivalent to the ground truth answer.

Rules:
- The model answer may contain extra explanation or context, but the core answer must match the ground truth.
- Minor differences in formatting, punctuation, or phrasing are acceptable if the meaning is the same.
- Numbers must be equivalent (e.g., "1000" and "1,000" are the same).
- For names, slight variations are acceptable if they clearly refer to the same entity.
- If the model answer contains the correct answer along with additional (correct) context, mark it as correct.
- If the model answer is wrong, incomplete, or contradicts the ground truth, mark it as incorrect.

You must respond with ONLY one word: "correct" or "incorrect". No explanation, no punctuation, no other text."""


def _extract_intermediate_output(content: str) -> str | None:
    text = (content or "").strip()
    if not text:
        return None
    m = re.search(r"final\s*answer\s*[:\-]?\s*", text, flags=re.IGNORECASE)
    if not m:
        return text
    return text[: m.start()].strip() or None


def _extract_final_answer(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    m = re.search(r"final\s*answer\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _gpt_judge(question: str, model_answer: str, ground_truth: str, gpt_model: str) -> dict:
    """Score a single (model_answer, ground_truth) pair via OpenAI."""
    user_prompt = (
        f"Question: {question}\n\n"
        f"    Model Answer: {model_answer}\n\n"
        f"    Ground Truth: {ground_truth}\n\n"
        f"    Is the model answer correct? output only one word: \"correct\" or \"incorrect\"."
    )
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": GPT_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "unknown", "raw": f"<judge error: {exc}>", "score": 0, "model": gpt_model}

    raw_lower = raw.lower()
    if "correct" in raw_lower and "incorrect" not in raw_lower:
        return {"verdict": "correct", "raw": raw, "score": 1, "model": gpt_model}
    if "incorrect" in raw_lower:
        return {"verdict": "incorrect", "raw": raw, "score": 0, "model": gpt_model}
    return {"verdict": "unknown", "raw": raw, "score": 0, "model": gpt_model}


# ---------------------------------------------------------------------------
# Per-question JSONL logger
# ---------------------------------------------------------------------------


class GaiaJsonlLogger:
    """Writes one self-contained JSON object per task to ``logs/*.jsonl``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at start of run so reruns don't double-append.
        self._fp = open(self.path, "w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def _sanitize(value: str) -> str:
    value = (value or "").strip() or "unknown"
    value = value.replace(os.sep, "_").replace("/", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("._-") or "unknown"


def _build_record(
    *,
    task: TaskInstance,
    trace: EpisodeTrace,
    library_snapshot: dict[str, Any],
    visible_ids: list[str],
    skills_used: list[str],
    axes: dict[str, str],
    raw_response: str,
    final_answer: str,
    gold: str,
    judge: dict[str, Any],
    eval_result: dict[str, float],
    tools_available: list[dict[str, str]],
    elapsed_ms: float,
    error: str | None,
) -> dict[str, Any]:
    metadata = task.metadata or {}
    attached = []
    for key in ("file_name", "file_path", "files"):
        v = metadata.get(key)
        if v:
            attached = v if isinstance(v, list) else [v]
            break
    return {
        "question_id": task.task_id,
        "level": metadata.get("level"),
        "input": {
            "question": task.instruction,
            "system_prompt": SYSTEM_PROMPT,
            "attached_files": attached,
            "tools_available": tools_available,
        },
        "skills": {
            "library_size": library_snapshot.get("size"),
            "all_skill_ids": library_snapshot.get("skill_ids", []),
            "visible_skill_ids": visible_ids,
            "skills_used_in_trace": skills_used,
            "skill_origin": axes["skill_origin"],
            "visibility": axes["visibility"],
            "retrieval": axes["retrieval"],
            "evolution": axes["evolution"],
        },
        "output": {
            "raw_response": raw_response,
            "intermediate_output": _extract_intermediate_output(raw_response),
            "final_answer": final_answer,
            "ground_truth": gold,
        },
        "evaluation": {
            "exact_match": float(eval_result.get("exact_match", 0.0)),
            "success": float(eval_result.get("success", 0.0)),
            "gpt_judge": judge,
        },
        "trace": {
            "num_steps": trace.num_steps,
            "num_errors": trace.num_errors,
            "total_tokens": trace.total_tokens,
            "total_cost": trace.total_cost,
            "elapsed_ms": elapsed_ms,
            "entries": [e.to_dict() for e in trace.entries],
        },
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkillEval framework on GAIA with per-question JSONL logs.")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "google", "local", "hf_transformers"])
    parser.add_argument("--model_name", "--model-name", default="gpt-4o-mini",
                        help="Model id (must match a configs/models/*.yaml or be a registered alias).")
    parser.add_argument("--model_config", "--model-config", default=None,
                        help="Path to model config YAML. Defaults based on model_name.")
    parser.add_argument("--dataset_config", "--dataset-config",
                        default="configs/datasets/gaia.yaml")

    parser.add_argument("--visibility", default="FL", choices=["NL", "LB", "FL"])
    parser.add_argument("--retrieval", default="NR", choices=["NR", "RR", "PR"])
    parser.add_argument("--evolution", default="FR", choices=["FR", "BU"])
    parser.add_argument("--skill_origin", "--skill-origin", default="SD", choices=["SD", "TD"])

    parser.add_argument("--subset", type=int, default=None,
                        help="Run only the first N test tasks.")
    parser.add_argument("--max_steps", "--max-steps", type=int, default=20)
    parser.add_argument("--max_episodes", "--max-episodes", type=int, default=200)
    parser.add_argument("--probe_tasks", "--probe-tasks", type=int, default=None,
                        help="Cap TD probe rollouts to N train tasks (default: 20%% of train pool).")

    parser.add_argument("--gpt_eval_model", "--gpt-eval-model", default="gpt-4o-mini")
    parser.add_argument("--no_tools", "--no-tools", action="store_true",
                        help="Disable env tools (run direct inference with skills as prompt text).")
    parser.add_argument("--logs_dir", "--logs-dir", default="logs")
    parser.add_argument("--log_level", "--log-level", default="INFO")
    return parser.parse_args()


def _resolve_model_config(provider: str, model_name: str, override: str | None) -> str:
    if override:
        return override
    if model_name in MODEL_CONFIG_MAP:
        return MODEL_CONFIG_MAP[model_name][1]
    candidate = Path(f"configs/models/{model_name.replace('-', '_').replace('.', '_')}.yaml")
    if candidate.exists():
        return str(candidate)
    if provider == "openai":
        return "configs/models/gpt4o_mini.yaml"
    raise FileNotFoundError(
        f"No model config found for provider={provider}, model={model_name}. "
        f"Pass --model_config explicitly."
    )


def _augment_task_with_system_prompt(task: TaskInstance) -> TaskInstance:
    """Prepend the GAIA system-prompt directive to the task instruction.

    The framework's protocols inject their own system prompt, so we attach
    the FINAL ANSWER directive directly to the user instruction to keep
    parity with run_gaia_inference.py's contract.
    """
    if "FINAL ANSWER" in (task.instruction or ""):
        return task
    task.instruction = f"{SYSTEM_PROMPT}\n\nQuestion:\n{task.instruction}"
    return task


def _run_cell(
    *,
    args: argparse.Namespace,
    dataset: Any,
    test_tasks: list[TaskInstance],
    model: Any,
    full_library: SkillLibrary,
    library_snapshot: dict[str, Any],
    visibility: SkillVisibility,
    retrieval: SkillRetrieval,
    evolution: UpdateStrategy,
    skill_origin: SkillOrigin,
    logs_dir: Path,
    subset_suffix: str,
) -> dict[str, Any]:
    """Run one (visibility, retrieval, evolution, skill_origin) cell."""
    axes = {
        "skill_origin": skill_origin.value,
        "visibility": visibility.value,
        "retrieval": retrieval.value,
        "evolution": evolution.value,
    }
    run_name = (
        f"gaia_{_sanitize(args.provider)}_{_sanitize(args.model_name)}"
        f"_{visibility.value}_{retrieval.value}_{evolution.value}"
        f"_{skill_origin.value}{subset_suffix}"
    )
    jsonl_path = logs_dir / f"{run_name}.jsonl"
    summary_path = logs_dir / f"{run_name}_summary.json"
    logger.info("Logs: %s", jsonl_path)
    jsonl_logger = GaiaJsonlLogger(jsonl_path)

    policy_code = resolve_retrieval_policy(retrieval)
    env_tools = [] if args.no_tools else dataset.get_tools()
    tool_executors = {} if args.no_tools else dataset.get_tool_executors()
    tools_available = [
        {"name": t.get("name"), "description": t.get("description", "")}
        for t in env_tools
    ]

    per_level: dict[Any, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "judge_correct": 0, "errors": 0}
    )
    total_correct_judge = 0
    total_correct_em = 0
    total_tokens = 0
    total_cost = 0.0

    cell_library = (
        SkillLibrary() if visibility == SkillVisibility.NO_LIBRARY else full_library
    )
    cell_snapshot = (
        {"size": 0, "skill_ids": []}
        if visibility == SkillVisibility.NO_LIBRARY
        else library_snapshot
    )

    for idx, task in enumerate(test_tasks, start=1):
        if idx > args.max_episodes:
            break

        visible_library = resolve_initial_library(visibility, cell_library, task)
        visible_ids = [s.skill_id for s in visible_library.list_skills()]

        agent = SkillAgent(
            model=model,
            library=visible_library,
            protocol="tool_using" if env_tools else "in_context",
            policy=policy_code,
            env_tools=env_tools,
            full_library=cell_library,
            tool_executors=tool_executors,
            config=Config(),
        )

        t0 = time.perf_counter()
        error: str | None = None
        try:
            trace = agent.solve(task, max_steps=args.max_steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Task %s crashed: %s", task.task_id, exc)
            trace = EpisodeTrace(
                task_id=task.task_id,
                model_id=model.model_id,
                entries=[],
                success=False,
                final_answer=None,
                metadata={"error": str(exc)},
            )
            error = str(exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_response = trace.final_answer or ""
        final_answer = _extract_final_answer(raw_response)
        gold = str(task.gold_answer or "")
        skills_used = [e.skill_used for e in trace.entries if e.skill_used]

        try:
            eval_result = dataset.evaluate_prediction(task, final_answer)
        except Exception as exc:  # noqa: BLE001
            eval_result = {"success": 0.0, "error": str(exc)}
        if "success" not in eval_result:
            eval_result["success"] = float(eval_result.get("exact_match", 0.0))

        judge = _gpt_judge(task.instruction, final_answer, gold, args.gpt_eval_model)

        record = _build_record(
            task=task,
            trace=trace,
            library_snapshot=cell_snapshot,
            visible_ids=visible_ids,
            skills_used=skills_used,
            axes=axes,
            raw_response=raw_response,
            final_answer=final_answer,
            gold=gold,
            judge=judge,
            eval_result=eval_result,
            tools_available=tools_available,
            elapsed_ms=elapsed_ms,
            error=error,
        )
        jsonl_logger.write(record)

        lvl = (task.metadata or {}).get("level", "unknown")
        per_level[lvl]["total"] += 1
        per_level[lvl]["correct"] += int(eval_result.get("exact_match", 0.0) >= 0.5)
        per_level[lvl]["judge_correct"] += int(judge.get("score", 0))
        if error:
            per_level[lvl]["errors"] += 1
        total_correct_em += int(eval_result.get("exact_match", 0.0) >= 0.5)
        total_correct_judge += int(judge.get("score", 0))
        total_tokens += int(trace.total_tokens or 0)
        total_cost += float(trace.total_cost or 0.0)

        logger.info(
            "[%s][%d/%d] %s — judge=%s em=%s steps=%d",
            run_name, idx, len(test_tasks), task.task_id, judge.get("verdict"),
            int(eval_result.get("exact_match", 0.0) >= 0.5), trace.num_steps,
        )

    jsonl_logger.close()

    n = sum(v["total"] for v in per_level.values())
    summary: dict[str, Any] = {
        "run_name": run_name,
        "provider": args.provider,
        "model": args.model_name,
        "axes": axes,
        "judge_model": args.gpt_eval_model,
        "n_tasks": n,
        "judge_accuracy": (total_correct_judge / n) if n else 0.0,
        "exact_match_accuracy": (total_correct_em / n) if n else 0.0,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "library_size": cell_snapshot["size"],
        "skill_ids": cell_snapshot["skill_ids"],
        "per_level": {
            str(k): {
                "total": v["total"],
                "correct_em": v["correct"],
                "correct_judge": v["judge_correct"],
                "errors": v["errors"],
                "judge_accuracy": (v["judge_correct"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in per_level.items()
        },
        "log_file": str(jsonl_path),
        "summary_file": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("=" * 60)
    print(f"Run: {run_name}")
    print(
        f"Tasks: {n} | Judge accuracy: {summary['judge_accuracy']:.1%} "
        f"| EM: {summary['exact_match_accuracy']:.1%}"
    )
    for lvl in sorted(per_level.keys(), key=lambda x: (isinstance(x, str), x)):
        v = per_level[lvl]
        acc = (v["judge_correct"] / v["total"]) if v["total"] else 0.0
        print(
            f"  Level {lvl}: judge {v['judge_correct']}/{v['total']} ({acc:.1%}) "
            f"| em {v['correct']} | err {v['errors']}"
        )
    print(f"Log: {jsonl_path}")
    print(f"Summary: {summary_path}")
    return summary


def _build_library_for_origin(
    evaluator: Evaluator,
    dataset: Any,
    model: Any,
    origin: SkillOrigin,
) -> tuple[SkillLibrary, dict[str, Any]]:
    try:
        lib = evaluator.create_library_once(dataset, model, origin)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Skill library construction failed for origin=%s (%s); using empty.",
            origin.value, exc,
        )
        lib = SkillLibrary()
    snap = {
        "size": lib.size,
        "skill_ids": [s.skill_id for s in lib.list_skills()],
    }
    logger.info("Library [origin=%s]: %d skills", origin.value, lib.size)
    return lib, snap


def main() -> None:
    args = _parse_args()
    os.chdir(PROJECT_ROOT)
    setup_logging(args.log_level)
    warn_compatibility_issues()

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY must be set (used for GPT judge and likely the agent)."
        )

    # ---- Build model + dataset ------------------------------------------------
    model_cfg_path = _resolve_model_config(args.provider, args.model_name, args.model_config)
    model_cfg = Config.from_yaml(model_cfg_path)
    if args.model_name:
        model_cfg.set("model.model_id", args.model_name)
        model_cfg.set("model.name", args.model_name)
    model = model_registry.build(args.provider, model_cfg)
    logger.info("Model: %s (provider=%s)", model.model_id, args.provider)

    ds_cfg = Config.from_yaml(args.dataset_config)
    if args.subset is not None:
        ds_cfg.set("dataset.max_samples", int(args.subset))
    dataset = dataset_registry.build("gaia", ds_cfg)
    test_tasks = dataset.test_tasks() if hasattr(dataset, "test_tasks") else dataset.tasks()
    if args.subset is not None:
        test_tasks = test_tasks[: args.subset]
    test_tasks = [_augment_task_with_system_prompt(t) for t in test_tasks]
    logger.info("Dataset: %s — %d tasks", dataset.name, len(test_tasks))

    eval_cfg = Config()
    if args.probe_tasks is not None:
        eval_cfg.set("experiment.skill_creation.probe_tasks", int(args.probe_tasks))
    evaluator = Evaluator(eval_cfg)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    subset_suffix = f"_subset{args.subset}" if args.subset is not None else ""

    # ---- Single-cell mode -----------------------------------------------------
    visibility = SkillVisibility(args.visibility)
    retrieval = SkillRetrieval(args.retrieval)
    evolution = UpdateStrategy(args.evolution)
    skill_origin = SkillOrigin(args.skill_origin)

    if visibility == SkillVisibility.NO_LIBRARY:
        full_library = SkillLibrary()
        library_snapshot = {"size": 0, "skill_ids": []}
    else:
        full_library, library_snapshot = _build_library_for_origin(
            evaluator, dataset, model, skill_origin,
        )

    _run_cell(
        args=args,
        dataset=dataset,
        test_tasks=test_tasks,
        model=model,
        full_library=full_library,
        library_snapshot=library_snapshot,
        visibility=visibility,
        retrieval=retrieval,
        evolution=evolution,
        skill_origin=skill_origin,
        logs_dir=logs_dir,
        subset_suffix=subset_suffix,
    )


if __name__ == "__main__":
    main()
