"""Qwen3-4B via vLLM for fast inference (non-thinking mode)."""

from skilleval.models.vllm_model import create_vllm_model

MODEL_NAME = "qwen3-4b-vllm"


def create_model():
    """Create Qwen3-4B model using vLLM backend in non-thinking mode.
    
    Non-thinking mode is handled by the chat template (no /think tags).
    
    Adjust parameters as needed:
    - gpu_memory_utilization: Set to 0.5 for lower memory usage
    - tensor_parallel_size: Increase for multi-GPU
    - max_tokens: Increase for longer outputs
    """
    return create_vllm_model(
        model_id="Qwen/Qwen3-4B",
        dtype="bfloat16",
        max_tokens=2048,
        temperature=0.0,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
