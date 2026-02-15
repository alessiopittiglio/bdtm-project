import logging
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


logger = logging.getLogger(__name__)


def create_client(path: str | Path):
    """Create a persistent ChromaDB client."""
    try:
        return chromadb.PersistentClient(path=str(path))
    except Exception:
        logger.exception("Failed to initialize ChromaDB client at %s", path)
        return None


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str,
    embedding_model: str = None,
    metadata: dict = None,
):
    """Get or create a ChromaDB collection."""
    if client is None:
        logger.error("ChromaDB client is required.")
        return None

    embedding_function = _build_embedding_function(embedding_model)

    try:
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata=metadata,
        )

        logger.info(
            "Collection '%s' ready. Items: %d",
            name,
            collection.count(),
        )
        return collection

    except Exception:
        logger.exception("Failed to get/create collection '%s'", name)
        return None


def _build_embedding_function(model_name):
    """Create a SentenceTransformer embedding function if requested."""
    if not model_name:
        return None

    try:
        logger.info("Using embedding model: %s", model_name)
        return SentenceTransformerEmbeddingFunction(model_name=model_name)
    except Exception:
        logger.warning(
            "Could not initialize embedding model '%s'. "
            "Falling back to external embeddings.",
            model_name,
            exc_info=True,
        )
        return None


def add_documents(
    collection: Collection,
    documents: list,
    embeddings: list,
    metadatas: list,
    ids: list,
    batch_size: int = 1000,
):
    """Add documents to a ChromaDB collection in batches."""
    if collection is None:
        logger.error("Collection is required.")
        return False

    if not _validate_lengths(documents, embeddings, metadatas, ids):
        logger.error("Input sequences must have the same length.")
        return False

    if not documents:
        logger.info("No documents to add.")
        return True

    try:
        for start in range(0, len(documents), batch_size):
            end = start + batch_size

            collection.add(
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

        logger.info(
            "Added %d documents to '%s'. Total: %d",
            len(ids),
            collection.name,
            collection.count(),
        )
        return True

    except Exception:
        logger.exception("Failed to add documents to '%s'", collection.name)
        return False


def _validate_lengths(*sequences) -> bool:
    """Ensure all sequences have identical length."""
    lengths = {len(seq) for seq in sequences}
    return len(lengths) == 1


def query_collection(
    collection: chromadb.Collection,
    query_embeddings: list,
    n_results: int = 5,
    metadata_filter: dict = None,
    include: list = None,
):
    """Perform a similarity search on a collection."""
    if collection is None:
        logger.error("Collection is required.")
        return None

    if not query_embeddings:
        logger.error("query_embeddings must not be empty.")
        return None

    normalized_embeddings = _normalize_embeddings(query_embeddings)

    try:
        return collection.query(
            query_embeddings=normalized_embeddings,
            n_results=n_results,
            where=metadata_filter,
            include=include or ["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("Query failed for collection '%s'", collection.name)
        return None


def _normalize_embeddings(embeddings):
    """Ensure embeddings are in the format expected by Chroma."""
    # Single embedding (flat list)
    if embeddings and not isinstance(embeddings[0], (list, tuple)):
        return [list(embeddings)]

    # NumPy 1D array
    if hasattr(embeddings, "ndim") and embeddings.ndim == 1:
        return [embeddings.tolist()]

    return [list(e) for e in embeddings]
