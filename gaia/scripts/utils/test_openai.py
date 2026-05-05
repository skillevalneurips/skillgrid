import os

from .llm_engine import LLMEngineError, create_engine


def run_openai(system_prompt: str, model: str, user_prompt: str) -> str:
    """
    Minimal OpenAI runner.

    This is intentionally dependency-light so it can be imported and used by
    evaluation code without requiring the `openai` Python package.

    Reads credentials from environment variables:
    - OPENAI_API_KEY (required)
    - OPENAI_BASE_URL (optional)
    """
    try:
        # create_engine reads OPENAI_API_KEY via os.getenv internally
        engine = create_engine("openai", model=model)
        output_text = engine.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=256,
        ).strip()
    except (LLMEngineError, Exception) as e:
        output_text = f"ERROR: {e}"

    return output_text

if __name__ == "__main__":
    # Optional: load .env for local testing
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    capital = run_openai("You are a helpful assistant","gpt-4o-mini", "Capital of France?")
    print(capital)