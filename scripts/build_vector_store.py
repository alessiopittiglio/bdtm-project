import hashlib
import logging

import torch

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


def generate_chunk_id(relative_path: str, start_token: int, end_token: int) -> str:
    base = f"{relative_path}_{start_token}_{end_token}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def prepare_chunks(documents, embedding_model):
    texts = []
    metadatas = []
    ids = []

    for doc in documents:
        source_file = doc["source_file_txt"]
        relative_path = doc["relative_path"]

        chunks = chunk_text(
            doc["text"],
            embedding_model,
            config.RETRIEVAL_CHUNK_SIZE,
            config.RETRIEVAL_CHUNK_OVERLAP,
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
                }
            )
            ids.append(chunk_id)

    return texts, metadatas, ids


def build_vector_index(collection, embedding_model, documents):
    texts, metadatas, ids = prepare_chunks(documents, embedding_model)

    if not texts:
        logger.warning("No chunks found. Skipping vector indexing.")
        return

    embeddings = encode_texts(
        texts=texts,
        embedding_model=embedding_model,
        normalize_embeddings=True,
    )

    add_documents(
        collection=collection,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def main():
    embedding_model = get_embedding_model(
        model_name=config.EMBEDDING_MODEL_NAME,
        device=device,
    )

    client = create_client(config.VECTOR_STORE_DIR)
    collection = get_or_create_collection(
        client=client,
        collection_name=config.COLLECTION_NAME,
        metadata_config={"hnsw:space": "cosine"},
    )

    documents = load_transcripts_and_metadata(config.CLEANED_DIR)

    build_vector_index(
        collection=collection,
        embedding_model=embedding_model,
        documents=documents,
    )

    if collection.count() > 0:
        query_text = "What is perplexity?"
        logger.info("Running example query: %r", query_text)

        query_embedding = encode_texts(
            texts=[query_text],
            embedding_model=embedding_model,
            normalize_embeddings=True,
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
