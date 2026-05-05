"""Qwen3-4B from HuggingFace for GAIA evaluation."""

from __future__ import annotations

import os

from skilleval.models.hf_transformers_model import create_hf_transformers_model

MODEL_NAME = "qwen3-4b"
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_PATH = os.getenv("QWEN3_4B_MODEL_PATH", MODEL_ID)


def create_model():
    return create_hf_transformers_model(
        model_id=MODEL_ID,
        model_path=MODEL_PATH,
        dtype=os.getenv("QWEN3_4B_DTYPE", "bfloat16"),
        max_tokens=int(os.getenv("QWEN3_4B_MAX_TOKENS", "2048")),
        temperature=float(os.getenv("QWEN3_4B_TEMPERATURE", "0.0")),
        top_p=float(os.getenv("QWEN3_4B_TOP_P", "1.0")),
        device_map=os.getenv("QWEN3_4B_DEVICE_MAP", "auto"),
        gpu_id=int(os.getenv("QWEN3_4B_GPU_ID", "0")),
        single_gpu=os.getenv("QWEN3_4B_SINGLE_GPU", "1").lower()
        not in {"0", "false", "no"},
    )
