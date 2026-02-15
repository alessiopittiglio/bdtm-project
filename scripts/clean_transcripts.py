import logging
import re
import shutil
from pathlib import Path

from ragqa import config

logger = logging.getLogger(__name__)

SHORT_TOKEN_RE = re.compile(
    rf"\b([a-z]{{1,{config.MAX_SHORT_TOKEN_LENGTH}}})\b"
    rf"(?:\s+\1\b){{{config.MIN_SHORT_TOKEN_REPETITIONS - 1},}}",
    flags=re.IGNORECASE,
)

REPEATED_PHRASE_RE = re.compile(r"(?is)(\b[^.!?]+?[.!?])(?:\s+\1)+")
REPEATED_SYMBOL_RE = re.compile(r"([^\w\s]{2,})(?:\s+\1){2,}")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and line breaks."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def remove_repeated_short_tokens(text: str) -> str:
    """Remove repeated short tokens (e.g., 'a a a a')."""
    return SHORT_TOKEN_RE.sub("", text)


def collapse_repeated_phrases(text: str) -> str:
    """Collapse repeated sentences into one occurrence."""
    return REPEATED_PHRASE_RE.sub(r"\1", text)


def remove_repeated_symbols(text: str) -> str:
    """Remove repeated symbol sequences (e.g., '!!! !!! !!!')."""
    return REPEATED_SYMBOL_RE.sub("", text)


def clean_text(text: str) -> str:
    """Run full text cleaning pipeline."""
    text = remove_repeated_short_tokens(text)
    text = collapse_repeated_phrases(text)
    text = remove_repeated_symbols(text)
    text = normalize_whitespace(text)
    return text


def process_text_file(src: Path, dst: Path):
    """Clean and save a text file."""
    logger.debug("Cleaning file: %s", src)

    cleaned = clean_text(src.read_text(encoding="utf-8"))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")


def copy_file(src: Path, dst: Path):
    """Copy file while preserving metadata."""
    logger.debug("Copying file: %s", src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def process_file(file_path: Path):
    """Process a single file based on extension."""
    relative_path = file_path.relative_to(config.RAW_DIR)
    output_path = config.CLEANED_DIR / relative_path

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        process_text_file(file_path, output_path)

    elif suffix == ".json":
        copy_file(file_path, output_path)

    else:
        logger.debug("Skipping unsupported file: %s", relative_path)


def iter_files(root: Path):
    """Return all files under a directory."""
    return [p for p in root.rglob("*") if p.is_file()]


def main():
    """Run the cleaning pipeline."""
    if not config.RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {config.RAW_DIR}")

    logging.basicConfig(level=logging.INFO)

    logger.info("Processing files from: %s", config.RAW_DIR)
    logger.info("Output directory: %s", config.CLEANED_DIR)

    for file_path in iter_files(config.RAW_DIR):
        process_file(file_path)

    print("Cleaning completed successfully.")


if __name__ == "__main__":
    main()
