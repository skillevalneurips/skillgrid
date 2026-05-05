"""Qwen3-8B via vLLM for fast inference (non-thinking mode)."""

from skilleval.models.vllm_model import create_vllm_model

MODEL_NAME = "qwen3-8b-vllm"


def create_model():
    """Create Qwen3-8B model using vLLM backend in non-thinking mode.
    
    Uses enable_thinking=False to disable thinking tags.
    """
    return create_vllm_model(
        model_id="Qwen/Qwen3-8B",
        dtype="bfloat16",
        max_tokens=2048,
        temperature=0.0,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
