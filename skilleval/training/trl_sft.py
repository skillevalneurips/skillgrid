"""Optional TRL SFTTrainer helper with compatibility guards."""

from __future__ import annotations

from typing import Any

from skilleval.utils.compat import check_transformers_trl_compat


def build_sft_trainer(
    model_name_or_path: str,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    output_dir: str = "./outputs/sft",
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 100,
    **kwargs: Any,
) -> Any:
    """Build a TRL ``SFTTrainer`` safely.

    This function intentionally performs imports lazily so the rest of the
    codebase can run without transformers/trl installed.
    """
    check_transformers_trl_compat(strict=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "TRL integration requires: transformers, trl, accelerate. "
            "Install with: pip install -r constraints-transformers-trl.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    config = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        **kwargs,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
    )
