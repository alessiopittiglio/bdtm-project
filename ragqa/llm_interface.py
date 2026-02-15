import logging
import re
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger(__name__)

CHANNEL_FINAL_PATTERN = re.compile(r"<\|channel\|>final<\|message\|>\s*")


def load_model(model_path: Path, config: dict | None = None) -> Llama | None:
    """Load a GGUF Llama model."""
    try:
        return Llama(
            model_path=str(model_path),
            **(config or {}),
        )
    except Exception:
        logger.exception("Failed to load GGUF model from %s", model_path)
        return None


def clean_output(text: str | None) -> str | None:
    if not text:
        return text

    match = CHANNEL_FINAL_PATTERN.search(text)
    cleaned = text[match.end() :] if match else text

    return cleaned.strip()


def generate_response(
    llm: Llama,
    messages: list[dict],
    config: dict | None = None,
    return_raw: bool = False,
) -> str | tuple[str, str] | None:
    """
    Generate a chat response from a loaded GGUF model.
    """
    try:
        response = llm.create_chat_completion(
            messages=messages,
            **(config or {}),
        )

        raw_text = response["choices"][0]["message"]["content"]
        cleaned_text = clean_output(raw_text)

        if return_raw:
            return raw_text.strip(), cleaned_text

        return cleaned_text

    except (KeyError, IndexError, TypeError):
        logger.exception("Unexpected response format from LLM.")
    except Exception:
        logger.exception("Failed to generate response.")

    return None
