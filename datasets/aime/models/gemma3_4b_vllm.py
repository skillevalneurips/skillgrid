"""Gemma-3-4B-IT via vLLM for fast inference."""

from skilleval.models.vllm_model import create_vllm_model

MODEL_NAME = "gemma3-4b-vllm"


def create_model():
    """Create Gemma-3-4B-IT model using vLLM backend."""
    return create_vllm_model(
        model_id="google/gemma-3-4b-it",
        dtype="bfloat16",
        max_tokens=2048,
        temperature=0.0,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=8192,  # Limit context length
    )
