"""GPT-4o-mini via OpenAI API for WebWalker evaluation."""

from skilleval.core.config import Config
from skilleval.models.openai_model import OpenAIModel


MODEL_NAME = "gpt-4o-mini"


def create_model() -> OpenAIModel:
    config = Config({
        "model": {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 4096,
        }
    })
    return OpenAIModel(config)
