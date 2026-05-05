"""GPT-5o-mini with high reasoning via OpenAI API."""

from skilleval.core.config import Config
from skilleval.models.openai_model import OpenAIModel


MODEL_NAME = "gpt-5-mini"


def create_model() -> OpenAIModel:
    config = Config({
        "model": {
            "provider": "openai",
            "model_id": "gpt-5-mini",
            "reasoning_effort": "high",
            "max_tokens": 4096,
        }
    })
    return OpenAIModel(config)
