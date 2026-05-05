"""GPT-5-mini via OpenAI API for WebWalker evaluation (low reasoning, no temperature)."""

from skilleval.core.config import Config
from skilleval.models.openai_model import OpenAIModel


MODEL_NAME = "gpt-5-mini"


def create_model() -> OpenAIModel:
    config = Config({
        "model": {
            "provider": "openai",
            "model_id": "gpt-5-mini",
            "max_tokens": 4096,
            "skip_temperature": True,
            "reasoning_effort": "low",
        }
    })
    return OpenAIModel(config)
