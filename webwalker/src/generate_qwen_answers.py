import argparse
import json
import os
from typing import Dict, List, Set

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question as accurately and concisely as possible."


def _read_visited_questions(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    visited: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                visited.add(json.loads(line)["question"])
            except Exception:
                # Ignore malformed lines to keep resume robust.
                continue
    return visited


def _build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {SYSTEM_PROMPT}\nUser: {question}\nAssistant:"


def _generate_openai_batch(
    client: OpenAI,
    model: str,
    questions: List[str],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    preds: List[str] = []
    for q in questions:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        preds.append(resp.choices[0].message.content.strip())
    return preds


@torch.inference_mode()
def _generate_batch(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode full sequences, then strip the prompt portion.
    preds: List[str] = []
    for i in range(outputs.shape[0]):
        generated_ids = outputs[i][inputs["input_ids"][i].shape[0] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        preds.append(text)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate WebWalkerQA predictions (jsonl of {question, pred}) "
            "using a local websailor-7b checkpoint."
        )
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API instead of a local model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (only used with --openai).",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help=(
            "GPU index to use (without changing CUDA_VISIBLE_DEVICES). "
            "The script will explicitly place the whole model on cuda:<gpu_id> to avoid multi-GPU sharding."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/deepfreeze/yav13/models",
        help=(
            "Path to the local model directory. This should point to the actual HF model folder "
            "(e.g., /deepfreeze/yav13/models/Qwen2.5-7B-Instruct)."
        ),
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="main",
        choices=["main", "silver"],
        help="WebWalkerQA split to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="websailor_7b_instruct_webwalkerqa_preds.jsonl",
        help="Where to write predictions jsonl (appends if exists).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (increase carefully; depends on GPU memory).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate per question.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (only used when temperature > 0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. 'auto' uses the model default.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help=(
            "Transformers device_map. If set to 'cpu', the model is kept on CPU. "
            "Otherwise, the script forces a single GPU via an explicit device_map to cuda:<gpu_id>."
        ),
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=100,
        help="Only process the first N questions (after filtering already-visited). Use 0 for all.",
    )

    args = parser.parse_args()

    ds = load_dataset("callanwu/WebWalkerQA", split=args.dataset_split)

    visited = _read_visited_questions(args.output_path)
    questions = [q for q in ds["question"] if q not in visited]
    if args.subset > 0:
        questions = questions[: args.subset]

    if not questions:
        print(f"No remaining questions to process. Output already contains all items: {args.output_path}")
        return

    # Ensure output exists for append mode.
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if not os.path.exists(args.output_path):
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write("")

    if args.openai:
        client = OpenAI()
        print(f"Using OpenAI model: {args.model_name}")

        pbar = tqdm(total=len(questions), desc="Generating predictions (OpenAI)")
        batch_size = max(1, args.batch_size)
        idx = 0
        while idx < len(questions):
            batch_qs = questions[idx : idx + batch_size]
            preds = _generate_openai_batch(
                client=client,
                model=args.model_name,
                questions=batch_qs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            with open(args.output_path, "a", encoding="utf-8") as f:
                for q, pred in zip(batch_qs, preds):
                    row: Dict[str, str] = {"question": q, "pred": pred}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            idx += len(batch_qs)
            pbar.update(len(batch_qs))

        pbar.close()
    else:
        if args.dtype == "auto":
            torch_dtype = None
        elif args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        device_map = None
        if str(args.device_map).lower() == "cpu":
            device_map = None
        elif torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
            device_map = {"": f"cuda:{args.gpu_id}"}

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()

        batch_size = max(1, args.batch_size)
        pbar = tqdm(total=len(questions), desc="Generating predictions")
        idx = 0
        while idx < len(questions):
            batch_qs = questions[idx : idx + batch_size]
            prompts = [_build_prompt(tokenizer, q) for q in batch_qs]
            preds = _generate_batch(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            with open(args.output_path, "a", encoding="utf-8") as f:
                for q, pred in zip(batch_qs, preds):
                    row: Dict[str, str] = {"question": q, "pred": pred}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            idx += len(batch_qs)
            pbar.update(len(batch_qs))

        pbar.close()

    print(f"Done. Wrote predictions to: {args.output_path}")


if __name__ == "__main__":
    main()


