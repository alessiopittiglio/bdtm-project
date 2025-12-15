import argparse
import gc
import json
import logging
import random
import re

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ragqa import config
from ragqa.data_processing import chunk_text, load_transcripts_and_metadata
from ragqa.embedding_utils import get_embedding_model, generate_embeddings
from ragqa.llm_interface import generate_response, load_llm
from ragqa.prompt_loader import load_prompt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def select_central_chunks(
    chunks,
    embeddings,
    num_clusters=3,
    random_state=42,
):

    if len(chunks) <= num_clusters:
        return list(enumerate(chunks))

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    selected = []

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]

        similarities = cosine_similarity(
            cluster_embeddings, centers[cluster_id].reshape(1, -1)
        ).squeeze()

        best_index = cluster_indices[np.argmax(similarities)]
        selected.append((best_index, chunks[best_index]))

    selected.sort(key=lambda x: x[0])
    return selected


def build_prompt(text_chunk):
    prompt_template = load_prompt(config.GENERATION_PROMPT_PATH)

    if not prompt_template:
        raise RuntimeError("Failed to load generation prompt template.")

    return prompt_template.format(context=text_chunk)


def _extract_json_from_output(output):
    # Attempt 1: fenced ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
    if match:
        return match.group(1)

    # Attempt 2: first '{' to last '}'
    start, end = output.find("{"), output.rfind("}")
    if start != -1 and end != -1 and end > start:
        return output[start : end + 1]

    return None


def find_quote_offsets(text, quote):
    start = text.find(quote)
    if start == -1:
        return None
    return {
        "quote": quote,
        "start_char": start,
        "end_char": start + len(quote),
    }


def parse_llm_output(output, source_info, chunk):
    if not output:
        return None

    json_str = _extract_json_from_output(output.strip())

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in LLM output.")
        return None

    evidence = []
    for quote in data.get("evidences", [])[:3]:
        offsets = find_quote_offsets(chunk, quote.strip())
        if offsets:
            evidence.append(offsets)

    question = data.get("question")
    correct_answer = data.get("correct_answer")
    distractors = data.get("distractors")

    if not _is_valid_mcq(question, correct_answer, distractors):
        return None

    question = question.strip()
    correct_answer = correct_answer.strip()
    distractors = [d.strip() for d in distractors]

    choices = [correct_answer] + distractors
    random.shuffle(choices)

    parsed_mcq_item = {
        "question": question,
        "choices": choices,
        "correct_index": choices.index(correct_answer),
        "source_info": source_info,
        "evidence": evidence,
    }

    return parsed_mcq_item


def _is_valid_mcq(question, correct_answer, distractors):
    if not isinstance(question, str) or not question.strip():
        return False
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        return False
    if (
        not isinstance(distractors, list)
        or len(distractors) != 3
        or not all(isinstance(d, str) and d.strip() for d in distractors)
    ):
        return False
    return True


def run_generation(test_mode=False, num_test_lectures=1):
    logger.info(
        "Starting MCQA Dataset Generation (%s MODE)",
        "TEST" if test_mode else "FULL",
    )

    logger.info("Loading LLM...")
    llm = load_llm(
        model_path=config.GENERATION_MODEL_PATH,
        model_config=config.LLM_MODEL_CONFIG,
    )

    if not llm:
        logger.error("Failed to load LLM. Aborting.")
        return

    lectures = load_transcripts_and_metadata(config.DATA_DIR)
    if test_mode:
        lectures = lectures[:num_test_lectures]
        logger.warning("Running in TEST MODE (%d lecture(s))", num_test_lectures)

    embedding_model = get_embedding_model(config.EMBEDDING_MODEL_NAME)
    generated_mcqs = []
    total_chunks = 0

    for lecture in lectures:
        lecture_text = lecture["text"]
        source_file = lecture["source_file_txt"]
        metadata = lecture["metadata"]
        course_details = metadata.get("course_details", {})

        module = metadata["course_details"].get("module")
        formatted_course_name = lecture["course_name"].replace("_", " ").title()
        formatted_module_name = (
            module.replace("_", " ").title() if module != "N/A_NO_MODULE" else ""
        )

        logger.info(
            f"Lecture {metadata['course_details']['lecture_num']} - "
            f"{formatted_course_name} {formatted_module_name})"
        )

        doc_id = f"{lecture['course_name']}_{source_file}"

        chunks = chunk_text(lecture_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        embeddings = generate_embeddings(
            texts=chunks,
            embedding_model=embedding_model,
            batch_size=32,
        )

        sampled_chunks = select_central_chunks(
            chunks=chunks,
            embeddings=embeddings,
            num_clusters=3,
        )

        for idx, (chunk_idx, chunk) in enumerate(sampled_chunks, start=1):
            logger.info("Chunk %d/%d", idx, len(sampled_chunks))
            total_chunks += 1
            text_chunk = chunk["text"]

            prompt = build_prompt(text_chunk)

            if test_mode:
                logger.info("Text chunk:\n%s", text_chunk)

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ]

            response = generate_response(
                llm,
                messages=messages,
                gen_config=config.LLM_GENERATION_CONFIG,
            )

            if not response:
                continue

            source_info = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_c{chunk['char_start']}_{chunk['char_end']}",
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "course_name": lecture["course_name"],
                "lecture_filename": source_file,
                "instructor_name": course_details.get("instructor", "N/A"),
                "lecture_date": course_details.get("lecture_date", "N/A"),
            }

            mcq = parse_llm_output(response, source_info, text_chunk)
            if not mcq:
                continue

            mcq["id"] = f"{source_info['chunk_id']}_mcq"

            if test_mode:
                logger.debug("Raw LLM output:\n%s", response)
                logger.info("Parsed MCQ:")
                print(json.dumps(mcq, indent=2, ensure_ascii=False))
                print("----------------------------------------\n")

            generated_mcqs.append(mcq)

    _save_results(generated_mcqs, total_chunks)
    _cleanup(llm)


def _save_results(mcqs, total_chunks):
    logger.info("Processed %d chunks.", total_chunks)
    logger.info("Generated %d MCQs.", len(mcqs))

    if not mcqs:
        return

    logger.info("Saving dataset to %s", config.MCQA_GENERATED_JSON)
    with open(config.MCQA_GENERATED_JSON, "w", encoding="utf-8") as f:
        json.dump(mcqs, f, ensure_ascii=False, indent=2)


def _cleanup(llm):
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
        help="Run in test mode with limited lectures and verbose output.",
    )
    parser.add_argument(
        "--num_lectures",
        type=int,
        default=1,
        help="Number of lectures to process in test mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(
        test_mode=args.test,
        num_test_lectures=args.num_lectures,
    )
