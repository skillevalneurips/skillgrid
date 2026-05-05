"""Generate recommendation reasoning traces for skill authoring.

This script samples train tasks from Reddit-V2 and ReDial, asks a local
Qwen3-4B-Instruct-2507 model to produce explicit recommendation rationale and
final recommendation JSON, and writes one trace file per source.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skilleval.core.config import Config
from skilleval.core.types import TaskInstance
from skilleval.datasets.conversational_rec import (
    ConversationalRecDataset,
    parse_recommendation_prediction,
)

logger = logging.getLogger("generate_reasoning_traces")

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = Path("datasets/conversational_rec/outputs/reasoning_traces")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    config_path: Path
    output_name: str


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_specs = selected_sources(args)
    source_tasks = {
        spec.name: load_sampled_tasks(
            spec=spec,
            samples=args.samples_per_source,
            candidate_pool_size=args.candidate_pool_size,
            seed=args.seed,
            redial_auto_download=args.redial_auto_download,
        )
        for spec in source_specs
    }

    if args.dry_run:
        for source, tasks in source_tasks.items():
            logger.info("Dry run loaded %d %s tasks", len(tasks), source)
            if tasks:
                print(render_prompt(tasks[0])[: args.preview_chars])
        return 0

    tokenizer, model = load_model(args)

    for spec in source_specs:
        tasks = source_tasks[spec.name]
        output_path = output_dir / spec.output_name
        generate_source_traces(
            source=spec.name,
            tasks=tasks,
            output_path=output_path,
            tokenizer=tokenizer,
            model=model,
            args=args,
        )

    logger.info("Trace generation complete. Outputs written under %s", output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate visible reasoning traces for conversational recommendation "
            "skill authoring."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--reddit-config",
        default="datasets/conversational_rec/config.yaml",
        type=Path,
    )
    parser.add_argument(
        "--redial-config",
        default="datasets/conversational_rec/config_redial.yaml",
        type=Path,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["redditv2", "redial"],
        default=["redditv2", "redial"],
    )
    parser.add_argument("--samples-per-source", type=int, default=100)
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=1000,
        help=(
            "Load this many train tasks from each source before seeded sampling. "
            "Use 100 to take exactly the first 100 adapter tasks."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Passed to transformers AutoModelForCausalLM.from_pretrained.",
    )
    parser.add_argument(
        "--redial-auto-download",
        action="store_true",
        help="Allow adapter to download missing ReDial CSVs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing trace files instead of resuming from them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/sample tasks and print one prompt preview without loading the model.",
    )
    parser.add_argument("--preview-chars", type=int, default=2000)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def selected_sources(args: argparse.Namespace) -> list[SourceSpec]:
    specs = []
    for source in args.sources:
        if source == "redditv2":
            specs.append(
                SourceSpec(
                    name="redditv2",
                    config_path=args.reddit_config,
                    output_name="redditv2_qwen3_4b_instruct_2507_traces.json",
                )
            )
        elif source == "redial":
            specs.append(
                SourceSpec(
                    name="redial",
                    config_path=args.redial_config,
                    output_name="redial_qwen3_4b_instruct_2507_traces.json",
                )
            )
    return specs


def load_sampled_tasks(
    *,
    spec: SourceSpec,
    samples: int,
    candidate_pool_size: int,
    seed: int,
    redial_auto_download: bool,
) -> list[TaskInstance]:
    cfg = Config.from_yaml(spec.config_path)
    cfg.set("dataset.sources", [spec.name])
    cfg.set("dataset.max_train_samples", max(samples, candidate_pool_size))
    cfg.set("dataset.max_test_samples", 1)
    cfg.set("dataset.max_samples", 1)
    if spec.name == "redial" and redial_auto_download:
        cfg.set("dataset.redial.auto_download", True)

    dataset = ConversationalRecDataset(cfg)
    tasks = list(dataset.train_tasks())
    if not tasks:
        raise RuntimeError(f"No train tasks loaded for {spec.name}")

    rng = random.Random(seed)
    if len(tasks) > samples:
        tasks = rng.sample(tasks, samples)

    logger.info(
        "Loaded %d sampled %s train tasks from %s",
        len(tasks),
        spec.name,
        spec.config_path,
    )
    return tasks


def load_model(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Trace generation requires torch and transformers in the active env."
        ) from exc

    logger.info("Loading tokenizer for %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    logger.info("Loading model %s", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Model loaded; CUDA available=%s", torch.cuda.is_available())
    return tokenizer, model


def generate_source_traces(
    *,
    source: str,
    tasks: list[TaskInstance],
    output_path: Path,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> None:
    payload = load_existing_payload(output_path) if not args.overwrite else None
    if payload is None:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "source": source,
            "model_id": args.model_id,
            "samples_requested": args.samples_per_source,
            "seed": args.seed,
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            },
            "traces": [],
        }

    existing_ids = {trace["task_id"] for trace in payload.get("traces", [])}
    remaining = [task for task in tasks if task.task_id not in existing_ids]
    logger.info(
        "%s: %d existing traces, %d remaining",
        source,
        len(existing_ids),
        len(remaining),
    )

    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        logger.info(
            "%s: generating %d-%d/%d batch_size=%d",
            source,
            batch_start + 1,
            batch_start + len(batch),
            len(remaining),
            len(batch),
        )
        prompts = [render_prompt(task) for task in batch]
        raw_responses = generate_batch_responses(prompts, tokenizer, model, args)
        for task, prompt, raw_response in zip(batch, prompts, raw_responses):
            parsed_response = parse_trace_response(raw_response)
            parsed_final_answer = extract_final_answer(parsed_response)
            payload["traces"].append(
                {
                    "task_id": task.task_id,
                    "source": source,
                    "split": task.metadata.get("split", "train"),
                    "instruction": task.instruction,
                    "gold_answer": task.gold_answer,
                    "metadata": task.metadata,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_response": parsed_response,
                    "parsed_final_answer": parsed_final_answer,
                    "parsed_recommendations": parse_recommendation_prediction(
                        parsed_final_answer or raw_response
                    ),
                }
            )
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, payload)

    write_json(output_path, payload)
    logger.info("%s: wrote %d traces to %s", source, len(payload["traces"]), output_path)


def render_prompt(task: TaskInstance) -> str:
    source = task.metadata.get("source", "unknown")
    context_titles = task.metadata.get("context_titles") or []
    gold_count = len(task.gold_answer) if isinstance(task.gold_answer, list) else 1

    return f"""You are creating reusable traces for a movie recommendation benchmark.

Task source: {source}
Gold recommendation count in the dataset: {gold_count}
Conversation:
{task.instruction}

Known movies already mentioned in the conversation:
{json.dumps(context_titles, ensure_ascii=False)}

Think step by step about the user's preferences and the conversation context, then produce final recommendations.

Return exactly one JSON object and no Markdown fences. Use this schema:
{{
  "reasoning_steps": [
    "Step 1: infer the user's movie preferences from the dialogue.",
    "Step 2: identify movies already mentioned and avoid repeating them unless the user explicitly asks.",
    "Step 3: rank candidate recommendations from best to worst."
  ],
  "final_answer": {{
    "recommendations": [
      {{"title": "Movie Title", "imdb_id": "tt1234567"}}
    ]
  }}
}}

Rules:
- Include up to 10 recommendations.
- Use null for imdb_id when you do not know the IMDb id.
- The final_answer.recommendations list must be ranked from strongest to weakest.
- Keep reasoning_steps concise but specific enough for a later skill writer to learn from.
"""


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> str:
    return generate_batch_responses([prompt], tokenizer, model, args)[0]


def generate_batch_responses(
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> list[str]:
    import torch

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = old_padding_side
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **generation_kwargs,
        )
    prompt_len = model_inputs.input_ids.shape[1]
    return [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        for row in generated_ids
    ]


def parse_trace_response(raw_response: str) -> dict[str, Any] | None:
    text = raw_response.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_final_answer(parsed_response: dict[str, Any] | None) -> Any:
    if not isinstance(parsed_response, dict):
        return None
    final_answer = parsed_response.get("final_answer")
    if final_answer:
        return final_answer
    if "recommendations" in parsed_response:
        return {"recommendations": parsed_response["recommendations"]}
    return None


def load_existing_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "traces" not in payload:
        raise ValueError(f"Existing trace file has unexpected shape: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
