import argparse
import hashlib
import logging

import torch
import yaml

from ragqa import config
from ragqa.data_processing import chunk_text, load_transcripts_and_metadata
from ragqa.embedding_utils import encode_texts, get_embedding_model
from ragqa.rag_core import (
    add_documents,
    create_client,
    get_or_create_collection,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def load_yaml(path):
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_chunk_id(relative_path: str, start_token: int, end_token: int) -> str:
    base = f"{relative_path}_{start_token}_{end_token}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def prepare_chunks(documents, embedding_model, chunk_size, chunk_overlap):
    texts = []
    metadatas = []
    ids = []

    for doc in documents:
        text = doc["text"]
        source_file = doc["source_file_txt"]
        relative_path = doc["relative_path"]

        chunks = chunk_text(
            text, embedding_model, chunk_size=chunk_size, overlap=chunk_overlap
        )

        for chunk in chunks:
            chunk_id = generate_chunk_id(
                relative_path=relative_path,
                start_token=chunk["start_token"],
                end_token=chunk["end_token"],
            )

            texts.append(chunk["text"])
            metadatas.append(
                {
                    "source": source_file,
                    "start_token": chunk["start_token"],
                    "end_token": chunk["end_token"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                }
            )
            ids.append(chunk_id)

    return texts, metadatas, ids


def build_vector_index(
    collection, embedding_model, documents, chunk_size, chunk_overlap
):
    texts, metadatas, ids = prepare_chunks(
        documents, embedding_model, chunk_size, chunk_overlap
    )

    if not texts:
        logger.warning("No chunks found. Skipping vector indexing.")
        return

    embeddings = encode_texts(
        texts=texts,
        model=embedding_model,
        normalize=True,
    )

    add_documents(
        collection=collection,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build a vector index for RAG.")

    parser.add_argument(
        "--input-dir",
        type=str,
        default=config.CLEANED_DIR,
        help="Input transcript directory.",
    )

    parser.add_argument(
        "--vector-dir",
        type=str,
        default=config.VECTOR_STORE_DIR,
        help="Vector store directory.",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=config.COLLECTION_NAME,
        help="Collection name.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval/mpnet_256.yaml",
        help="Experiment config file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    embedding_model = get_embedding_model(
        model_name=cfg["embedding_model"],
        device=device,
    )

    client = create_client(args.vector_dir)
    collection = get_or_create_collection(
        client=client,
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    documents = load_transcripts_and_metadata(args.input_dir)

    build_vector_index(
        collection=collection,
        embedding_model=embedding_model,
        documents=documents,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["overlap"],
    )

    if collection.count() > 0:
        query_text = "What is perplexity?"
        logger.info("Running example query: %r", query_text)

        query_embedding = encode_texts(
            texts=[query_text],
            model=embedding_model,
            normalize=True,
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
        )

        print(f"\nExample query results")
        print("--------------------------------")

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            print("No results found.")
            return

        for idx, text in enumerate(results["documents"][0]):
            metadata = metadatas[0][idx]
            print(f"\nResult {idx + 1}")
            print(f"Source: {metadata['source']}")
            print(f"Chunk Text: {text[:300]}...\n")
    else:
        logger.warning("Vector store is empty. Skipping example query.")


if __name__ == "__main__":
    main()
