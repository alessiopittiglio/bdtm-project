import argparse
import gc
import json
import hashlib
import logging
import re
from pathlib import Path

import torch
from rapidfuzz import fuzz

from ragqa import config
from ragqa.data_processing import chunk_text, load_transcripts_and_metadata
from ragqa.embedding_utils import get_embedding_model
from ragqa.llm_interface import generate_response, load_model
from ragqa.prompt_loader import load_prompt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)


def find_substring_offsets(text: str, substring: str, threshold: int = 85):
    """Return (start, end) offsets of substring inside text using fuzzy match."""
    if not substring:
        return None

    start = text.find(substring)
    if start != -1:
        return start, start + len(substring)

    window_size = len(substring)
    best_score = 0
    best_span = None

    for i in range(0, len(text) - window_size + 1):
        window = text[i : i + window_size]
        score = fuzz.partial_ratio(window, substring)

        if score > best_score:
            best_score = score
            best_span = (i, i + window_size)

    return best_span if best_score >= threshold else None


def extract_json_block(text: str) -> dict:
    """Extract the first valid JSON object found in text."""
    if not text:
        return None

    # Attempt 1: direct parsing
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt 2: extract first valid JSON object
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except Exception:
            continue

    return None


def build_chunk_selection_prompt(chunks) -> str:
    """Build prompt for selecting relevant chunks."""
    template = load_prompt(config.SELECTION_PROMPT_PATH)
    if not template:
        raise RuntimeError("Failed to load selection prompt template.")

    formatted_chunks = [
        f'[Chunk {i}]\n"""\n{chunk}\n"""' for i, chunk in enumerate(chunks)
    ]

    return template.format(chunks_block="\n\n".join(formatted_chunks))


def build_generation_prompt(context: str) -> str:
    """Build prompt for MCQ generation."""
    template = load_prompt(config.GENERATION_PROMPT_PATH)
    if not template:
        raise RuntimeError("Failed to load generation prompt template.")

    return template.format(context=context)


def parse_selected_chunks(output: str):
    """Parse selected chunk IDs from LLM output."""
    data = extract_json_block(output)
    if not data:
        return None

    ids = data.get("selected_chunks")

    if isinstance(ids, list) and len(ids) == 3 and all(isinstance(i, int) for i in ids):
        return ids

    return None


def is_valid_mcq(question, options, correct_answer):
    """Validate MCQ structure."""
    return (
        isinstance(question, str)
        and question.strip()
        and isinstance(options, dict)
        and set(options.keys()) == {"A", "B", "C", "D"}
        and all(isinstance(v, str) and v.strip() for v in options.values())
        and isinstance(correct_answer, str)
        and correct_answer in {"A", "B", "C", "D"}
    )


def extract_course_info(lecture):
    """Extract only the course information needed for MCQ source metadata."""
    details = lecture.get("metadata", {}).get("course_details", {})

    module = details.get("module")
    if module == "N/A_NO_MODULE":
        module = None

    return {
        "name": lecture.get("course_name"),
        "lecture_num": details.get("lecture_num"),
        "module": module,
        "lecture_date": details.get("lecture_date"),
    }


def parse_mcq(output: str):
    """Parse MCQ from LLM output."""
    data = extract_json_block(output)
    if not data:
        logger.error("Invalid JSON in LLM output.")
        return None

    question = data.get("question")
    options = data.get("options")
    correct_answer = data.get("correct_answer")
    evidence = data.get("evidence")

    if not is_valid_mcq(question, options, correct_answer):
        return None

    return {
        "question": question.strip(),
        "options": {k: v.strip() for k, v in options.items()},
        "correct_answer": correct_answer.strip(),
        "evidence": evidence,
    }


def select_chunks(llm, chunk_texts):
    if len(chunk_texts) == 1:
        return [0]

    prompt = build_chunk_selection_prompt(chunk_texts)

    response = generate_response(
        llm,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        config={"temperature": 0.2},
    )

    return parse_selected_chunks(response)


def run_generation(test_mode: bool = False, num_test_lectures: int = 1):
    """Main MCQ dataset generation pipeline."""
    logger.info("\n%s", "=" * 60)
    logger.info("MCQA Dataset Generation (%s MODE)", "TEST" if test_mode else "FULL")
    logger.info("%s\n", "=" * 60)

    llm = load_model(
        model_path=config.GENERATION_MODEL_PATH,
        config=config.LLM_MODEL_CONFIG,
    )

    if not llm:
        logger.error("Failed to load LLM.")
        return

    embedding_model = get_embedding_model(
        model_name=config.EMBEDDING_MODEL_NAME,
        device=device,
    )

    lectures = load_transcripts_and_metadata(config.CLEANED_DIR)

    if test_mode:
        lectures = lectures[:num_test_lectures]
        logger.warning("Running in TEST MODE (%d lecture(s))", num_test_lectures)

    generated_mcqs = []

    for lecture in lectures:
        mcqs = process_lecture(lecture, llm, embedding_model, test_mode)
        generated_mcqs.extend(mcqs)

    save_results(generated_mcqs)
    cleanup(llm)


def process_lecture(lecture, llm, embedding_model, test_mode):
    lecture_text = lecture["text"]
    course_info = extract_course_info(lecture)
    doc_id = lecture["relative_path"]

    logger.info(
        "\n%s\nLecture %s - %s\n%s",
        "-" * 60,
        course_info["lecture_num"],
        lecture["course_name"],
        "-" * 60,
    )

    chunks = chunk_text(
        lecture_text,
        embedding_model,
        config.GENERATION_CHUNK_SIZE,
        config.GENERATION_CHUNK_OVERLAP,
    )

    selected_ids = select_chunks(llm, [c["text"] for c in chunks])
    if not selected_ids:
        logger.warning("Chunk selection failed.")
        return []

    logger.info("Selected chunk IDs: %s", selected_ids)

    mcqs = []
    for chunk_id in selected_ids:
        mcq = generate_mcq(
            chunks[chunk_id],
            doc_id,
            llm,
            course_info,
            test_mode,
        )
        if mcq:
            mcqs.append(mcq)

    return mcqs


def generate_mcq(chunk, doc_id, llm, course_info, test_mode):
    context = chunk["text"]

    prompt = build_generation_prompt(context)

    response = generate_response(
        llm,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        config=config.LLM_GENERATION_CONFIG,
    )

    if not response:
        return None

    mcq = parse_mcq(response)
    if not mcq:
        logger.warning("Failed to parse MCQ.")
        return None

    base_string = f"{doc_id}_{chunk['start_char']}_{mcq['question']}"
    mcq["id"] = hashlib.md5(base_string.encode()).hexdigest()

    offsets = find_substring_offsets(context, mcq.get("evidence"))
    if not offsets:
        logger.warning("Evidence not found in chunk.")
        return None

    local_start, local_end = offsets

    mcq["evidence"] = context[local_start:local_end]
    mcq["source_info"] = {
        "doc_id": doc_id,
        "gold_span": {
            "start_char": chunk["start_char"] + local_start,
            "end_char": chunk["start_char"] + local_end,
        },
        "course": course_info,
    }

    if test_mode:
        logger.debug("Raw LLM output:\n%s", response)
        logger.info("Parsed MCQ:")
        print(json.dumps(mcq, indent=2, ensure_ascii=False))
        print("----------------------------------------\n")

    return mcq


def save_results(mcqs):
    logger.info("Generated %d MCQs.", len(mcqs))
    if not mcqs:
        return

    output_path = Path(config.MCQA_GENERATED_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config.MCQA_GENERATED_JSON, "w", encoding="utf-8") as f:
        json.dump(mcqs, f, ensure_ascii=False, indent=2)

    logger.info("Saved dataset to %s", config.MCQA_GENERATED_JSON)


def cleanup(llm):
    del llm
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an MCQ dataset from lecture transcripts."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited lectures.",
    )
    parser.add_argument(
        "--num-lectures",
        type=int,
        default=1,
        help="Number of lectures to process in test mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_generation(
        test_mode=args.test,
        num_test_lectures=args.num_lectures,
    )


if __name__ == "__main__":
    main()
