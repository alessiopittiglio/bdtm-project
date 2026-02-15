import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_transcripts(data_dir: str) -> list[dict]:
    """Load all `.txt` files recursively from a directory."""
    root = Path(data_dir)

    return [
        {
            "text": path.read_text(encoding="utf-8"),
            "source": path.name,
        }
        for path in root.rglob("*.txt")
    ]


def load_transcripts_and_metadata(data_dir: str) -> list[dict]:
    """Load transcripts and corresponding metadata from a structured directory."""
    root = Path(data_dir)
    documents = []

    for course_dir in root.iterdir():
        if not course_dir.is_dir():
            continue

        course_name = course_dir.name

        module_dirs = [
            d
            for d in course_dir.iterdir()
            if d.is_dir() and d.name.startswith("module")
        ]
        target_dirs = module_dirs or [course_dir]

        for target_dir in target_dirs:
            transcripts_dir = target_dir / "transcripts_txt"
            metadata_dir = target_dir / "transcripts_metadata"

            for txt_path in tqdm(
                transcripts_dir.glob("*.txt"),
                desc=f"Lectures: {course_name}",
                leave=False,
            ):
                metadata_path = metadata_dir / f"{txt_path.stem}.json"

                text = txt_path.read_text(encoding="utf-8")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

                documents.append(
                    {
                        "text": text,
                        "source_file_txt": txt_path.stem,
                        "course_name": course_name,
                        "relative_path": str(txt_path.relative_to(root)),
                        "metadata": metadata,
                    }
                )

    logger.info("Loaded %d transcripts.", len(documents))
    return documents


def chunk_text(
    text: str,
    embedding_model: Any,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Split text into overlapping token chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk size.")

    tokenizer = embedding_model.tokenizer

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    if not input_ids:
        return []

    stride = chunk_size - overlap
    chunks = []

    for start_token in range(0, len(input_ids), stride):
        end_token = min(start_token + chunk_size, len(input_ids))

        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]

        chunk_text = text[start_char:end_char]

        chunks.append(
            {
                "text": chunk_text,
                "start_token": start_token,
                "end_token": end_token,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        if end_token >= len(input_ids):
            break

    return chunks
