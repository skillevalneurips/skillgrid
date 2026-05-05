---
name: evaluation
description: Use this skill whenever the agent needs to evaluate, score, or compare answers against reference answers. This includes using LLM-based evaluators (GPT-4, LangChain CoT QA), computing accuracy metrics across difficulty levels and task types (single-source vs multi-source), generating evaluation reports, processing prediction files (JSONL format), and benchmarking agent performance on the WebWalkerQA dataset. If the task involves evaluating answer quality or computing benchmark scores, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# Evaluation Guide

## Overview

This guide covers evaluating agent answers against reference answers in the WebWalker benchmark. The WebWalkerQA dataset contains 680 human-verified questions across 4 real-world scenarios, categorized by hop type (single-source/multi-source) and difficulty (easy/medium/hard). Evaluation uses GPT-4 based chain-of-thought QA evaluation via LangChain.

## Quick Start

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("cot_qa")

result = evaluator.evaluate_strings(
    prediction="The deadline is March 21, 2025.",
    input="When is the submission deadline for ACL 2025 Industry Track?",
    reference="The paper submission deadline for the ACL 2025 Industry Track is March 21, 2025."
)
print(f"Score: {result['score']}")  # 1 or 0
```

## Dependencies

```bash
pip install langchain langchain-community langchain-core datasets openai
```

## Loading the Dataset

### Load WebWalkerQA from Hugging Face
```python
from datasets import load_dataset

ds = load_dataset("callanwu/WebWalkerQA", split="main")

# Build a question -> (answer, info) lookup
info_dict = {}
for question, answer, info in zip(ds["question"], ds["answer"], ds["info"]):
    info_dict[question] = {"answer": answer, "info": info}

print(f"Total questions: {len(info_dict)}")
```

### Dataset Structure
```python
# Each entry contains:
# {
#     "Question": "When is the submission deadline for ACL 2025 Industry Track?",
#     "Answer": "The deadline is March 21, 2025.",
#     "Root_Url": "https://2025.aclweb.org/",
#     "Info": {
#         "Hop": "multi-source" or "single-source",
#         "Domain": "Conference",
#         "Language": "English",
#         "Difficulty_Level": "Easy" / "Medium" / "Hard",
#         "Source_Website": [...],
#         "Golden_Path": [...]
#     }
# }
```

## LLM-Based Evaluation

### Single Answer Evaluation
```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("cot_qa")

def evaluate_single(question, prediction, reference):
    """Evaluate a single prediction against reference."""
    result = evaluator.evaluate_strings(
        prediction=prediction,
        input=question,
        reference=reference
    )
    return result["score"]  # 1 (correct) or 0 (incorrect)
```

### Evaluate with Retry
```python
import time
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("cot_qa")

def evaluate_with_retry(question, prediction, reference, max_retries=10):
    """Evaluate with exponential backoff on API failures."""
    for attempt in range(max_retries):
        try:
            result = evaluator.evaluate_strings(
                prediction=prediction,
                input=question,
                reference=reference
            )
            return result["score"]
        except Exception as e:
            print(f"Evaluation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))
            else:
                raise e
```

### Batch Evaluation with Threading
```python
import json
import concurrent.futures
from tqdm import tqdm
from langchain.evaluation import load_evaluator

def batch_evaluate(predictions_path, output_path, info_dict):
    """Evaluate all predictions in a JSONL file."""
    evaluator = load_evaluator("cot_qa")

    # Load predictions
    data_list = []
    with open(predictions_path, "r") as f:
        for line in f:
            data = json.loads(line)
            ref = info_dict.get(data["question"], {}).get("answer")
            if ref:
                data["answer"] = ref
                data_list.append(data)

    # Skip already evaluated
    visited = set()
    try:
        with open(output_path, "r") as f:
            for line in f:
                visited.add(json.loads(line)["question"])
    except FileNotFoundError:
        pass
    data_list = [d for d in data_list if d["question"] not in visited]

    def eval_one(data):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return evaluator.evaluate_strings(
                    prediction=data["pred"],
                    input=data["question"],
                    reference=data["answer"]
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 * (2 ** attempt))
                else:
                    raise e

    total_score = 0
    count = 0

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_data = {executor.submit(eval_one, d): d for d in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    result = future.result(timeout=4)
                    data = future_to_data[future]
                    data["score"] = result["score"]
                    total_score += data["score"]
                    count += 1

                    with open(output_path, "a") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {e}")

    return total_score / count if count > 0 else 0
```

## Computing Metrics

### Compute Accuracy by Category
```python
import json

def compute_metrics(results_path, info_dict):
    """Compute accuracy metrics broken down by type and difficulty."""
    categories = {
        "single_source_easy": [],
        "single_source_medium": [],
        "single_source_hard": [],
        "multi_source_easy": [],
        "multi_source_medium": [],
        "multi_source_hard": [],
        "overall": []
    }

    with open(results_path, "r") as f:
        for line in f:
            item = json.loads(line)
            score = item.get("score")
            if score is None:
                continue

            categories["overall"].append(score)

            info = item.get("info", {})
            if not info and item["question"] in info_dict:
                info = info_dict[item["question"]].get("info", {})

            q_type = info.get("type", info.get("Hop", ""))
            difficulty = info.get("difficulty_level", info.get("Difficulty_Level", ""))

            key = f"{q_type}_{difficulty}".lower().replace("-", "_")
            if key in categories:
                categories[key].append(score)

    # Compute averages
    metrics = {}
    for key, scores in categories.items():
        metrics[key] = sum(scores) / len(scores) if scores else None

    return metrics
```

### Generate Report
```python
import json

def generate_report(results_path, info_dict, report_path=None):
    """Generate a full evaluation report."""
    metrics = compute_metrics(results_path, info_dict)

    if report_path is None:
        report_path = results_path.replace(".jsonl", "_report.json")

    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    # Print summary
    print("=" * 50)
    print("WebWalker Evaluation Report")
    print("=" * 50)
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key:30s}: {value:.3f}")
        else:
            print(f"  {key:30s}: N/A")
    print("=" * 50)

    return metrics
```

## Prediction File Format

### JSONL Prediction Format
```python
# Each line in the predictions file should be:
# {"question": "...", "pred": "..."}

import json

def write_predictions(predictions, output_path):
    """Write predictions to JSONL file."""
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps({
                "question": pred["question"],
                "pred": pred["prediction"]
            }, ensure_ascii=False) + "\n")
```

### Read and Filter Predictions
```python
import json

def load_predictions(filepath):
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions

def filter_unevaluated(predictions, results_path):
    """Filter out already-evaluated predictions."""
    evaluated = set()
    try:
        with open(results_path, "r") as f:
            for line in f:
                evaluated.add(json.loads(line)["question"])
    except FileNotFoundError:
        pass
    return [p for p in predictions if p["question"] not in evaluated]
```

## RAG Comparison

### Evaluate Multiple Models
```python
def compare_models(model_results):
    """Compare evaluation metrics across multiple models."""
    print(f"{'Model':<30} {'Overall':>10} {'SS-Easy':>10} {'MS-Easy':>10}")
    print("-" * 60)
    for model_name, metrics in model_results.items():
        overall = metrics.get("overall", 0) or 0
        ss_easy = metrics.get("single_source_easy", 0) or 0
        ms_easy = metrics.get("multi_source_easy", 0) or 0
        print(f"{model_name:<30} {overall:>10.3f} {ss_easy:>10.3f} {ms_easy:>10.3f}")
```

## Common Tasks

### End-to-End Evaluation Pipeline
```python
import json
from datasets import load_dataset
from langchain.evaluation import load_evaluator

def full_evaluation_pipeline(predictions_path, output_path):
    """Complete evaluation from predictions to report."""
    # Load dataset
    ds = load_dataset("callanwu/WebWalkerQA", split="main")
    info_dict = {}
    for q, a, info in zip(ds["question"], ds["answer"], ds["info"]):
        info_dict[q] = {"answer": a, "info": info}

    # Run evaluation
    accuracy = batch_evaluate(predictions_path, output_path, info_dict)
    print(f"Overall accuracy: {accuracy:.3f}")

    # Generate report
    metrics = generate_report(output_path, info_dict)
    return metrics
```

### Quick Accuracy Check
```python
import json

def quick_accuracy(results_path):
    """Quick accuracy calculation without category breakdown."""
    scores = []
    with open(results_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if "score" in item:
                scores.append(item["score"])
    if scores:
        print(f"Accuracy: {sum(scores)/len(scores):.3f} ({sum(scores)}/{len(scores)})")
    return sum(scores) / len(scores) if scores else 0
```

### CLI Usage
```python
# Run evaluation from command line:
# python evaluate.py --input_path predictions.jsonl --output_path results.jsonl

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Predictions JSONL path")
    parser.add_argument("--output_path", type=str, help="Results output path")
    args = parser.parse_args()

    full_evaluation_pipeline(args.input_path, args.output_path)
```

## Quick Reference

| Task | Method | Example |
|------|--------|---------|
| Load dataset | datasets | `load_dataset("callanwu/WebWalkerQA")` |
| Evaluate one | LangChain | `evaluator.evaluate_strings(pred, input, ref)` |
| Batch evaluate | ThreadPoolExecutor | Parallel evaluation with retries |
| Compute metrics | Category grouping | Group scores by type + difficulty |
| Generate report | JSON output | Save metrics to `_report.json` |
| Write predictions | JSONL | `{"question": "...", "pred": "..."}` per line |
| Quick accuracy | Simple average | `sum(scores) / len(scores)` |
