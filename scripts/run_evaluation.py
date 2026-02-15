import argparse
import gc
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from ragqa import config
from ragqa.embedding_utils import encode_texts, get_embedding_model
from ragqa.evaluation_utils import calculate_accuracy, parse_llm_choice, save_results
from ragqa.llm_interface import generate_response, load_model
from ragqa.prompt_loader import load_prompt
from ragqa.rag_core import create_client, get_or_create_collection, query_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)


def load_yaml(path: Path):
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: Path):
    """Load experiment and its sub-configurations."""
    exp_cfg = load_yaml(path)

    retrieval_cfg = load_yaml(exp_cfg["retrieval_config"])
    generation_cfg = load_yaml(exp_cfg["generation_config"])

    return {
        "experiment": exp_cfg,
        "retrieval": retrieval_cfg,
        "generation": generation_cfg,
        "mode": exp_cfg.get("mode", "rag"),
    }


def load_mcq_dataset(path: str):
    """Load and validate MCQ dataset."""
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    required_fields = {"question", "options", "correct_answer"}
    if not dataset or not all(required_fields <= item.keys() for item in dataset):
        raise ValueError("Invalid MCQ dataset format")

    logger.info("Loaded %d MCQs", len(dataset))
    return dataset


def build_prompt(question, options, contexts):
    """Create an MCQ prompt with optional RAG context."""

    context_block = ""
    if contexts:
        joined = "\n".join(
            f"CONTEXT {i + 1}\n{chunk}" for i, chunk in enumerate(contexts)
        )
        context_block = f"{joined}\nEND CONTEXT\n\n"

    choices_block = "\n".join(f"{k}. {options[k]}" for k in ["A", "B", "C", "D"])

    template = load_prompt(config.ANSWER_FORMAT_PROMPT)

    return template.format(
        context_block=context_block,
        question=question,
        choices_block=choices_block,
    )


def normalize(text: str) -> str:
    """Basic text normalization."""
    return " ".join(text.lower().split())


# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------
def retrieve_chunks(
    question,
    embedding_model,
    collection,
    top_k,
):
    """Retrieve relevant chunks for a query."""
    try:
        embedding = encode_texts([question], embedding_model, normalize=True)[0]

        results = query_collection(
            collection,
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
        )

        if not results:
            return [], [], []

        return results["documents"][0], results.get("metadatas", [[]])[0]

    except Exception:
        logger.exception("Retrieval failed.")
        return [], []


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------
def compute_metrics(
    contexts,
    evidence,
    k,
):
    """Compute Recall@K, MRR@K, and NDCG@K."""
    evidence_norm = normalize(evidence)

    relevant_positions = [
        i for i, chunk in enumerate(contexts[:k]) if evidence_norm in normalize(chunk)
    ]

    # Recall@k
    recall = float(bool(relevant_positions))

    # MRR@k
    mrr = 1.0 / (relevant_positions[0] + 1) if relevant_positions else 0.0

    # NDCG@k
    ndcg = (
        1.0 / torch.log2(torch.tensor(relevant_positions[0] + 2)).item()
        if relevant_positions
        else 0.0
    )

    return {"recall@k": recall, "mrr@k": mrr, "ndcg@k": ndcg}


def run_llm_only(mcq, llm, generation_cfg):
    """LLM without retrieval."""

    prompt = build_prompt(
        mcq["question"],
        mcq["options"],
        contexts=[],
    )

    response = generate_response(
        llm=llm,
        messages=[
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": prompt},
        ],
        config=generation_cfg["generation_config"],
    )

    if response is None:
        return "", {}

    return parse_llm_choice(response), {}


# ------------------------------------------------------------------
# RAG PIPELINE
# ------------------------------------------------------------------
def run_rag(
    mcq,
    embedding_model,
    collection,
    llm,
    retrieval_cfg,
    generation_cfg,
):
    """Run full RAG pipeline for a single question."""

    contexts, _ = retrieve_chunks(
        mcq["question"],
        embedding_model,
        collection,
        retrieval_cfg["top_k"],
    )

    metrics = compute_metrics(
        contexts,
        mcq["evidence"],
        retrieval_cfg["top_k"],
    )

    prompt = build_prompt(mcq["question"], mcq["options"], contexts)

    response = generate_response(
        llm=llm,
        messages=[
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": prompt},
        ],
        config=generation_cfg["generation_config"],
    )

    if response is None:
        return "", metrics

    # DEBUG
    # print("LLM Response:", response)
    return parse_llm_choice(response), metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG MCQA experiments")

    parser.add_argument(
        "--exp",
        nargs="+",
        default=None,
        help="Experiment names (default: all)",
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs/experiments",
        help="Directory with experiment configs",
    )

    return parser.parse_args()


def get_experiment_paths(config_dir, selected):
    exp_dir = Path(config_dir)

    if not selected:
        return sorted(exp_dir.glob("*.yaml"))

    paths = []
    for name in selected:
        path = exp_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {name}")
        paths.append(path)

    return paths


def cleanup():
    """Free GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    logger.info("Starting RAG MCQA evaluation.")

    dataset = load_mcq_dataset(config.MCQA_CURATED_JSON)

    chroma_client = create_client(config.VECTOR_STORE_DIR)
    if not chroma_client:
        logger.error("Failed to initialize Chroma client.")
        return

    experiment_paths = get_experiment_paths(args.config_dir, args.exp)
    logger.info("Experiments to run: %s", [p.stem for p in experiment_paths])

    results = []

    for exp_path in experiment_paths:
        cfg = load_experiment_config(exp_path)

        exp = cfg["experiment"]
        retrieval_cfg = cfg["retrieval"]
        generation_cfg = cfg["generation"]
        mode = cfg["mode"]

        logger.info("Running experiment: %s", exp["experiment_name"])

        embedding_model = get_embedding_model(retrieval_cfg["embedding_model"], device)
        llm = load_model(
            generation_cfg["model_path"], config=generation_cfg["model_config"]
        )

        collection = get_or_create_collection(chroma_client, exp["collection_name"])

        start = time.time()

        for mcq in tqdm(dataset, desc=exp["experiment_name"]):
            if mode == "rag":
                pred, scores = run_rag(
                    mcq,
                    embedding_model,
                    collection,
                    llm,
                    retrieval_cfg,
                    generation_cfg,
                )
            else:
                pred, scores = run_llm_only(mcq, llm, generation_cfg)

            correct = mcq["correct_answer"]

            results.append(
                {
                    "experiment": exp["experiment_name"],
                    # "question_id": mcq.get("id"),
                    "correct": correct,
                    "predicted": pred,
                    "is_correct": pred == correct,
                    **scores,
                }
            )

        logger.info("Finished %s in %.2fs", exp["experiment_name"], time.time() - start)

    results_df = pd.DataFrame(results)
    summary_df = calculate_accuracy(results_df)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_results(
        results_df,
        summary_df,
        config.OUTPUT_DIR,
        f"rag_mcqa_eval_{timestamp}",
    )

    cleanup()


if __name__ == "__main__":
    main()
