import logging
import torch

from src import config
from src.data_processing import load_transcripts_and_metadata, chunk_text
from src.embedding_utils import get_embedding_model, generate_embeddings
from src.rag_core import (
    initialize_chroma_client,
    get_chroma_collection,
    add_to_collection
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

def main():
    embedding_model = get_embedding_model(
        model_name=config.EMBEDDING_MODEL_NAME,
        device=device,
    )

    client = initialize_chroma_client(config.VECTOR_STORE_PATH)

    collection = get_chroma_collection(
        client=client,
        collection_name=config.COLLECTION_NAME,
        metadata_config={"hnsw:space": "cosine"},
    )

    all_lessons_data = load_transcripts_and_metadata(config.DATA_PATH)

    all_chunks_texts = []
    all_metadatas = []
    all_ids = []

    for idx, doc in enumerate(all_lessons_data):
        doc_text = doc['text']
        doc_source = doc['source_file_txt']
        chunks = chunk_text(doc_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_source}_{idx}_chunk_{i}"
            all_chunks_texts.append(chunk)
            all_metadatas.append({"source": doc_source})
            all_ids.append(chunk_id)

    if all_chunks_texts:
        embeddings = generate_embeddings(
            texts=all_chunks_texts,
            embedding_model=embedding_model,
            normalize_embeddings=True,
        )

        add_to_collection(
            collection=collection,
            texts=all_chunks_texts,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids,
        )
    else:
        logger.warning("No chunks to add to the vector store. Exiting.")

    count = collection.count()
    if count > 0:
        query_text = "What is perplexity?"
        logger.info(f"Running example query: '{query_text}'")
        query_embedding = generate_embeddings(
            texts=[query_text],
            embedding_model=embedding_model,
            normalize_embeddings=True,
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3, 
        )

        print(f"\n--- Example Query Results ---")

        if results and results.get('documents'):
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nResult {i + 1}")
                print(f"Source: {results['metadatas'][0][i]['source']}")
                print(f"Chunk Text: {doc[:300]}...\n")  # Print first 300 characters
        else:
            print("No results found for the example query.")

if __name__ == "__main__":
    main()
