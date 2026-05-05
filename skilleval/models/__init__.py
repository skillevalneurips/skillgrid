from skilleval.models.base import BaseModel, ModelResponse
from skilleval.models.openai_model import OpenAIModel
from skilleval.models.anthropic_model import AnthropicModel
from skilleval.models.google_model import GoogleModel
from skilleval.models.local_model import LocalModel

try:
    from skilleval.models.hf_transformers_model import (
        HFTransformersModel,
        create_hf_transformers_model,
    )
except Exception:  # optional dependency
    HFTransformersModel = None
    create_hf_transformers_model = None

try:
    from skilleval.models.vllm_model import VLLMModel
except Exception:  # optional dependency
    VLLMModel = None

__all__ = [
    "BaseModel",
    "ModelResponse",
    "OpenAIModel",
    "AnthropicModel",
    "GoogleModel",
    "LocalModel",
    "HFTransformersModel",
    "create_hf_transformers_model",
    "VLLMModel",
]
