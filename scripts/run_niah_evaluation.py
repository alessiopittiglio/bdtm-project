import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ragqa import config
from ragqa.embedding_utils import encode_texts, get_embedding_model
from ragqa.llm_interface import generate_response, load_model
from ragqa.rag_core import create_client, get_or_create_collection, query_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

QUESTION = "The secret code for this lecture"
ANSWER = "ALPHA-7392-BETA"


def load_yaml(path: Path):
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: Path):
    """Load experiment and retrieval configurations."""
    exp_cfg = load_yaml(path)

    retrieval_cfg = load_yaml(exp_cfg["retrieval_config"])
    generation_cfg = load_yaml(exp_cfg["generation_config"])

    return {
        "experiment": exp_cfg,
        "retrieval": retrieval_cfg,
        "generation": generation_cfg,
        "mode": exp_cfg.get("mode", "rag"),
    }


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def compute_exact_match(prediction: str, gold: str):
    return float(normalize(prediction) == normalize(gold))


def compute_recall_at_k(contexts, gold_answer, k):
    gold_norm = normalize(gold_answer)
    return float(any(gold_norm in normalize(ctx) for ctx in contexts[:k]))


def bm25_retrieve(question, documents, top_k=10):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(question.split())
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [documents[i] for i in top_indices]


def dense_retrieve(question, embedding_model, collection, top_k=10):
    embedding = encode_texts([question], embedding_model, normalize=True)[0]

    results = query_collection(
        collection,
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        include=["documents"],
    )

    if not results:
        return []

    return results["documents"][0]


def hybrid_retrieve(
    question,
    embedding_model,
    collection,
    cross_encoder,
    top_k_dense=10,
    top_k_bm25=10,
    final_top_k=10,
):
    dense_docs = dense_retrieve(question, embedding_model, collection, top_k_dense)

    all_docs = collection.get(include=["documents"])["documents"]
    bm25_docs = bm25_retrieve(question, all_docs, top_k_bm25)

    merged_docs = list(dict.fromkeys(dense_docs + bm25_docs))

    if not merged_docs:
        return []

    pairs = [(question, doc) for doc in merged_docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(merged_docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, _ in ranked[:final_top_k]]


def load_all_txt(folder_path: str):
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    texts = [
        txt_file.read_text(encoding="utf-8")
        for txt_file in sorted(folder_path.rglob("*.txt"))
    ]

    logger.info("Loaded %d txt files", len(texts))
    return "\n\n".join(texts)


def run_llm(exp_cfg, generation_cfg, llm):
    txt_folder = config.NIAH_DIR / exp_cfg["dataset"]
    context = load_all_txt(txt_folder)

    prompt = f"Context:\n{context}\n\nQuestion: {QUESTION}\nAnswer:"

    response = generate_response(
        llm=llm,
        messages=[
            {"role": "system", "content": generation_cfg.get("system_prompt", "")},
            {"role": "user", "content": prompt},
        ],
        config=generation_cfg["generation_config"],
    )

    if response is None:
        return "", {"exact_match": 0.0}

    prediction = response.strip()
    exact = compute_exact_match(prediction, ANSWER)

    return prediction, {"exact_match": exact}


def run_rag(
    exp_cfg,
    retrieval_cfg,
    generation_cfg,
    llm,
    embedding_model,
    collection,
    cross_encoder,
):
    top_k = retrieval_cfg["top_k"]

    contexts = hybrid_retrieve(
        QUESTION,
        embedding_model,
        collection,
        cross_encoder,
        top_k_dense=10,
        top_k_bm25=10,
        final_top_k=top_k,
    )

    recall = compute_recall_at_k(contexts, ANSWER, top_k)
    context_block = "\n\n".join(contexts)

    prompt = f"Context:\n{context_block}\n\nQuestion: {QUESTION}\nAnswer:"

    response = generate_response(
        llm=llm,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        config=generation_cfg["generation_config"],
    )

    if response is None:
        return "", {"recall@k": recall, "exact_match": 0.0}

    prediction = response.strip()
    exact = compute_exact_match(prediction, ANSWER)

    return prediction, {
        "recall@k": recall,
        "exact_match": exact,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Needle in a Haystack tests")

    parser.add_argument(
        "--exp",
        nargs="+",
        default=None,
        help="Experiment names",
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs/niah",
    )

    parser.add_argument(
        "--vector_dir",
        type=str,
        default=config.VECTOR_STORE_DIR,
        help="Vector store directory.",
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

    logger.info("Starting Needle-in-a-Haystack evaluation")

    chroma_client = create_client(args.vector_dir)

    experiment_paths = get_experiment_paths(args.config_dir, args.exp)
    logger.info("Experiments: %s", [p.stem for p in experiment_paths])

    results = []

    for exp_path in experiment_paths:
        cfg = load_experiment_config(exp_path)

        exp = cfg["experiment"]
        retrieval_cfg = cfg["retrieval"]
        generation_cfg = cfg["generation"]
        mode = cfg["mode"]

        logger.info("Running experiment: %s", exp["experiment_name"])

        llm = load_model(
            generation_cfg["model_path"],
            config=generation_cfg["model_config"],
        )

        embedding_model = None
        collection = None

        embedding_model = get_embedding_model(
            retrieval_cfg["embedding_model"],
            device,
        )

        cross_encoder = CrossEncoder(
            retrieval_cfg["cross_encoder_model"],
            device=device,
        )

        collection = get_or_create_collection(
            chroma_client,
            exp["collection_name"],
        )

        start = time.time()

        if mode == "rag":
            pred, scores = run_rag(
                exp,
                retrieval_cfg,
                generation_cfg,
                llm,
                embedding_model,
                collection,
                cross_encoder,
            )
        else:
            pred, scores = run_llm(
                exp,
                generation_cfg,
                llm,
            )

        elapsed = time.time() - start

        results.append(
            {
                "experiment": exp["experiment_name"],
                "prediction": pred,
                "time_seconds": round(elapsed, 4),
                **scores,
            }
        )

        logger.info("Finished %s in %.2fs", exp["experiment_name"], time.time() - start)

    results_df = pd.DataFrame(results)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = Path(config.OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_path / f"niah_results_{timestamp}.csv", index=False, encoding="utf-8-sig"
    )
    cleanup()


if __name__ == "__main__":
    main()
