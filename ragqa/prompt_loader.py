import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prompt(path: str | Path) -> str | None:
    """
    Load a prompt template from a file.
    """
    file_path = Path(path)

    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Prompt template file not found: %s", file_path)
    except OSError:
        logger.exception("Failed to read prompt template: %s", file_path)

    return None
