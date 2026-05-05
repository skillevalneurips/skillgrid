"""GPT-4o via OpenAI API for GAIA evaluation."""

from skilleval.core.config import Config
from skilleval.models.openai_model import OpenAIModel


MODEL_NAME = "gpt-4o"


def create_model() -> OpenAIModel:
    config = Config({
        "model": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 4096,
        }
    })
    return OpenAIModel(config)
