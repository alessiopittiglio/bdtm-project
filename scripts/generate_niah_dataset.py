import argparse
import json
import logging
import random
from pathlib import Path

import torch

from ragqa import config
from ragqa.data_processing import load_transcripts_and_metadata

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NEEDLE_TEXT = "The secret code for this lecture is ALPHA-7392-BETA"
QUESTION_TEXT = "What is the secret code for this lecture?"
ANSWER_TEXT = NEEDLE_TEXT.split("is ")[-1]

INPUT_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("data/niah")

POSITION_MAP = {
    "start": 0.10,
    "middle": 0.50,
    "end": 0.90,
}


def find_insertion_boundary(text: str, index: int):
    """
    Find a natural insertion point after the given index.
    """
    for i in range(index, len(text)):
        if text[i] in ".!?":
            return i + 1

    for i in range(index, len(text)):
        if text[i].isspace():
            return i

    return index


def insert_needle(text: str, needle: str, position: float):
    """
    Insert needle into text at a relative position.
    """
    if not text:
        return text, None

    position = max(0.0, min(1.0, position))
    raw_index = int(len(text) * position)
    insert_index = find_insertion_boundary(text, raw_index)

    while insert_index < len(text) and text[insert_index].isspace():
        insert_index += 1

    prefix = text[:insert_index].rstrip()
    suffix = text[insert_index:].lstrip()

    modified = f"{prefix}\n\n{needle}\n\n{suffix}"

    start_char = len(prefix) + 2
    gold_span = {
        "start_char": start_char,
        "end_char": start_char + len(needle),
    }

    return modified, gold_span


def resolve_position(mode):
    """Resolve insertion mode into a relative position."""
    if mode == "random":
        return random.uniform(0.05, 0.95)

    if mode not in POSITION_MAP:
        raise ValueError(f"Invalid position mode: {mode}")

    return POSITION_MAP[mode]


def write_text_file(base_dir, relative_path, content):
    target_path = base_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def copy_metadata(base_dir, lecture):
    metadata = lecture["metadata"]
    relative_path = Path(lecture["relative_path"])

    parts = list(relative_path.parts)

    if "transcripts_txt" not in parts:
        raise ValueError(f"'transcripts_txt' not found in path: {relative_path}")

    idx = parts.index("transcripts_txt")
    parts[idx] = "transcripts_metadata"

    metadata_path = Path(*parts).with_suffix(".json")
    write_json(base_dir / metadata_path, metadata)


def generate_dataset(sizes, position_mode: str):
    """Main dataset generation loop."""
    logger.info("\n%s", "=" * 60)
    logger.info("Needle-in-a-Haystack Generation")
    logger.info("%s\n", "=" * 60)

    lectures = load_transcripts_and_metadata(INPUT_DIR)
    random.shuffle(lectures)

    # ordered_sources = [lec["source_file_txt"] for lec in lectures]

    target_index = 0
    target_lecture = lectures[target_index]

    position = resolve_position(position_mode)

    logger.info("Target lecture: %s", target_lecture["source_file_txt"])
    logger.info("Seed: %d", SEED)

    modified_text, gold_span = insert_needle(
        target_lecture["text"],
        NEEDLE_TEXT,
        position,
    )

    lectures[target_index]["text"] = modified_text

    for size in sizes:
        if size > len(lectures):
            raise ValueError(f"Requested size {size} exceeds corpus size.")

        dataset_dir = OUTPUT_DIR / f"dataset_{size}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        selected = lectures[:size]

        for lecture in selected:
            write_text_file(dataset_dir, lecture["relative_path"], lecture["text"])
            copy_metadata(dataset_dir, lecture)

        logger.info("Created dataset_%d", size)

    queries = [
        {
            "question": QUESTION_TEXT,
            "answer": ANSWER_TEXT,
        }
    ]

    write_json(OUTPUT_DIR / "queries.json", queries)

    metadata = {
        "needle": NEEDLE_TEXT,
        "target_lecture": target_lecture["source_file_txt"],
        "seed": SEED,
        # "needle_position": position_mode,
        # "resolved_position": position,
        # "gold_span": gold_span,
        # "ordered_lectures": ordered_sources,
    }

    write_json(OUTPUT_DIR / "metadata.json", metadata)

    logger.info("\nAll datasets created successfully.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate incremental Needle-in-a-Haystack test dataset."
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="Dataset sizes (e.g. --sizes 1 5 10 20 50)",
    )

    parser.add_argument(
        "--position",
        type=str,
        default="middle",
        choices=["start", "middle", "end", "random"],
        help="Needle position.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generate_dataset(
        sizes=args.sizes,
        position_mode=args.position,
    )


if __name__ == "__main__":
    main()
