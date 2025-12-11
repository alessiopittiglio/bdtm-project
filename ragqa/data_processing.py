import logging
import json
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_transcripts(data_dir: str) -> list[dict]:
    """
    Load all .txt files from the given directory.

    Args:
        data_dir (str): The directory containing the .txt files.

    Returns:
        list[dict]: A list of dictionaries, each containing 'text' and 'source' (the
            filename).
    """
    documents = []
    data_path = Path(data_dir)

    for file_path in data_path.rglob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(
                {
                    "text": text,
                    "source": file_path.name,
                }
            )
    return documents


def load_transcripts_and_metadata(data_dir: str) -> list[dict]:
    """
    Load transcripts and metadata from the specified directory.

    Args:
        data_dir (str): The directory containing the .txt files and metadata files.

    Returns:
        list[dict]: A list of dictionaries, each containing 'text', 'source_file_txt',
            'course_name', and 'metadata'.
    """
    all_documents_data = []
    data_path = Path(data_dir)

    for course_dir in data_path.iterdir():
        if not course_dir.is_dir():
            continue

        course_name = course_dir.name
        module_dirs = [
            d
            for d in course_dir.iterdir()
            if d.is_dir() and d.name.startswith("module")
        ]

        target_dirs = module_dirs if module_dirs else [course_dir]

        for target_dir in target_dirs:
            transcript_path = target_dir / "transcripts_txt"
            metadata_path = target_dir / "transcripts_metadata"

            for txt_file_path in tqdm(
                transcript_path.glob("*.txt"),
                desc=f"Lectures: {course_name}",
                leave=False,
            ):
                base_filename = txt_file_path.stem
                meta_file_path = metadata_path / f"{base_filename}.json"

                with open(meta_file_path, "r", encoding="utf-8") as f_meta:
                    metadata_content = json.load(f_meta)

                with open(txt_file_path, "r", encoding="utf-8") as f_txt:
                    text_content = f_txt.read()

                all_documents_data.append(
                    {
                        "text": text_content,
                        "source_file_txt": base_filename,
                        "course_name": course_name,
                        "metadata": metadata_content,
                    }
                )

    logger.info(f"Loaded {len(all_documents_data)} transcripts.")
    return all_documents_data


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split the text into chunks of specified size with overlap.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The number of overlapping characters between consecutive
            chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size.")

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index].strip()
        if chunk and len(chunk) > 50:
            chunks.append(chunk)
        start_index += chunk_size - chunk_overlap
    return chunks
