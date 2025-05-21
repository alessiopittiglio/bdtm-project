import argparse
import gc
import json
import logging
import os
import random
import re
import torch

from ragqa import config
from ragqa.data_processing import load_transcripts_and_metadata, chunk_text
from ragqa.llm_interface import load_llm, generate_response
from ragqa.prompt_loader import load_prompt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def format_prompt(text_chunk):
    prompt_template = load_prompt(config.PROMPT_TEMPLATE_PATH)
    if not prompt_template:
        pass # Handle the case where the prompt template is not loaded
    return prompt_template.format(context=text_chunk)

def parse_output(output, source_file):
    if not output:
        return None
    
    output_str = output.strip()

    # Attempt 1: Look for JSON delimited by ```json ... ```
    json_block_match = re.search(
        r"```json\s*(\{.*?\})\s*```", output_str, re.DOTALL
    )
    if json_block_match:
        json_str = json_block_match.group(1)
    else:
        # Attempt 2: Look the first '{' and the last '}'
        first_brace = output_str.find('{')
        last_brace = output_str.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = output_str[first_brace : last_brace + 1]
        else:
            logger.error("Failed to find JSON in LLM output.")
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from LLM output.")
        return None
    
    question = data.get("question")
    correct_answer = data.get("correct_answer")
    distractors = data.get("distractors")

    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        return None
    if (
        not isinstance(distractors, list) 
        or len(distractors) != 3 
        or not all(isinstance(d, str) and d.strip() for d in distractors)
    ):
        return None
    
    question = question.strip()
    correct_answer = correct_answer.strip()
    distractors = [d.strip() for d in distractors]

    choices = [correct_answer] + distractors
    random.shuffle(choices)
    correct_index = choices.index(correct_answer)

    parsed_mcq_item = {
        "question": question,
        "choices": choices,
        "correct_index": correct_index,
        "source_info": source_file
    }

    return parsed_mcq_item
    
def main(test_mode=False, num_test_lectures=1):
    logger.info(
        "Starting MCQA Dataset Generation "
        f"({'TEST MODE' if test_mode else 'FULL MODE'})"
    )

    logger.info("Loading LLM for MCQ generation...")
    llm = load_llm(
        model_path=config.LLM_MODEL_PATH,
        model_config=config.LLM_MODEL_CONFIG,
    )
    if not llm:
        logger.error("Unable to load the generator LLM. Exiting.")
        return
    
    lectures = load_transcripts_and_metadata(config.DATA_DIR)

    generated_mcqs = []
    total_chunks_processed = 0

    lectures_to_process = lectures
    if test_mode:
        logger.warning(
            "EXECUTING IN TEST MODE. "
            f"Processing {num_test_lectures} lecture(s).\n"
        )
        lectures_to_process = lectures[:num_test_lectures]
        
    for lecture in lectures_to_process:
        lecture_text = lecture['text']
        source_file = lecture['source_file_txt']
        course_name = lecture['course_name']
        metadata = lecture['metadata']

        module = metadata['course_details'].get('module')
        formatted_course_name = course_name.replace('_', ' ').title()
        formatted_module_name = (
            module.replace('_', ' ').title()
            if module != "N/A_NO_MODULE" else ""
        )

        logger.info(
            f"Lecture {metadata['course_details']['lecture_num']} - "
            f"{formatted_course_name} {formatted_module_name})"
        )

        chunks = chunk_text(
            lecture_text,
            config.CHUNK_SIZE, 
            config.CHUNK_OVERLAP
        )
        
        num_to_sample = config.NUM_CHUNKS_TO_SAMPLE

        if len(chunks) > num_to_sample:
            step = len(chunks) // num_to_sample
            sampled_chunks = [
                (chunks[i], i) for i in range(0, len(chunks), step)
            ][:num_to_sample]
        else:
            sampled_chunks = list(enumerate(chunks))

        for i, (text_chunk, chunk_index) in enumerate(sampled_chunks, start=1):
            logger.info(f"Chunk {i}/{len(sampled_chunks)}")
            
            prompt = format_prompt(text_chunk)

            llm_messages = [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": "<think>\n" # Forcing LLM to think
                },
            ]

            llm_response = generate_response(
                llm,
                messages=llm_messages,
                gen_config=config.LLM_GENERATION_CONFIG,
            )
            total_chunks_processed += 1


            if llm_response:
                course_details = metadata.get("course_details", {})
                source_info = {
                    "course_name": course_name,
                    "lecture_filename": source_file,
                    "chunk_index": chunk_index,
                    "instructor_name": course_details.get("instructor", "N/A"),
                    "lecture_date": course_details.get("lecture_date", "N/A"),
                }

                mcq_item = parse_output(
                    llm_response,
                    source_info
                )

                if test_mode and mcq_item:
                    logger.debug(f"Raw LLM Output:\n{llm_response}")
                    logger.info("Parsed MCQ:")
                    print(json.dumps(mcq_item, indent=2, ensure_ascii=False))
                    print("----------------------------------------\n")

                if mcq_item:
                    mcq_id = (
                        f"{os.path.splitext(source_file)[0]}"
                        f"_c{chunk_index}_mcq{len(generated_mcqs)}"
                    )
                    mcq_item['id'] = mcq_id
                    mcq_item['source_details'] = mcq_item.pop('source_info')
                    generated_mcqs.append(mcq_item)
    
    logger.info(f"MCQA generation completed")
    logger.info(f"Processed {total_chunks_processed} sampled chunks.")
    logger.info(f"Generated {len(generated_mcqs)} potential MCQs.")
    
    if generated_mcqs:
        logger.info(f"Saving generated dataset to: {config.MCQA_GENERATED_JSON}")
        with open(config.MCQA_GENERATED_JSON, 'w', encoding='utf-8') as f:
            json.dump(generated_mcqs, f, ensure_ascii=False, indent=2)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a MCQ dataset from transcripts."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode, generating only a few example MCQs and printing them.",
    )
    parser.add_argument(
        "--num_lectures",
        type=int,
        default=1,
        help="Number of lectures to process in test mode (default: 1).",
    )
    args = parser.parse_args()
    main(test_mode=args.test, num_test_lectures=args.num_lectures)
