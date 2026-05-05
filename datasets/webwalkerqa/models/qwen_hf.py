"""Qwen instruct model from HuggingFace for WebWalkerQA evaluation."""

from __future__ import annotations

import os

from skilleval.models.hf_transformers_model import create_hf_transformers_model

MODEL_NAME = "qwen-hf"
MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", MODEL_ID)


def create_model():
    return create_hf_transformers_model(
        model_id=MODEL_ID,
        model_path=MODEL_PATH,
        dtype=os.getenv("QWEN_DTYPE", "auto"),
        max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "2048")),
        temperature=float(os.getenv("QWEN_TEMPERATURE", "0.0")),
        top_p=float(os.getenv("QWEN_TOP_P", "1.0")),
        device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
        gpu_id=int(os.getenv("QWEN_GPU_ID", "0")),
        single_gpu=os.getenv("QWEN_SINGLE_GPU", "1").lower() not in {"0", "false", "no"},
    )
