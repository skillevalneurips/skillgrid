"""Qwen2.5-7B-Instruct via vLLM for fast inference."""

from skilleval.models.vllm_model import create_vllm_model

MODEL_NAME = "qwen25-7b-instruct-vllm"


def create_model():
    """Create Qwen2.5-7B-Instruct model using vLLM backend."""
    return create_vllm_model(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        max_tokens=2048,
        temperature=0.0,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=16384,
    )
