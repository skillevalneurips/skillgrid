"""Qwen3-4B-Instruct-2507 via vLLM for fast inference."""

from skilleval.models.vllm_model import create_vllm_model

MODEL_NAME = "qwen3-4b-instruct-vllm"


def create_model():
    """Create Qwen3-4B-Instruct-2507 model using vLLM backend.
    
    This is an instruction-tuned version that should follow ReAct format better.
    max_model_len is set to 16384 to fit in GPU memory (model default is 262144).
    """
    return create_vllm_model(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        dtype="bfloat16",
        max_tokens=2048,
        temperature=0.0,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=16384,  # Limit context to fit in GPU memory
    )
