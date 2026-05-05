"""Llama-3.2-3B-Instruct via vLLM for fast inference."""

from skilleval.models.vllm_model import create_vllm_model

MODEL_NAME = "llama32-3b-vllm"


def create_model():
    """Create Llama-3.2-3B-Instruct model using vLLM backend.
    
    Llama doesn't have thinking mode, so it should follow ReAct format directly.
    """
    return create_vllm_model(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        dtype="bfloat16",
        max_tokens=2048,
        temperature=0.0,
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
