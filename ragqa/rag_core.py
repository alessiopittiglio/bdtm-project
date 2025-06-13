import logging
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

def initialize_chroma_client(path=None):
    """
    Initialize and return a PersistentClient of ChromaDB.

    Args:
        path (str): Path where the index is/will be saved.
    Returns:
        chromadb.PersistentClient: ChromaDB client object or None if error.
    """
    path_str = str(path)
    try:
        client = chromadb.PersistentClient(path=path_str)
        return client
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client in {path}: {e}")
        return None

def get_chroma_collection(
        client: chromadb.PersistentClient,
        collection_name: str,
        embedding_model_name: str = None,
        metadata_config: dict = None
    ):
    """
    Obtains an existing collection or creates a new one.

    Args:
        client (chromadb.PersistentClient): ChromaDB client object.
        collection_name (str): Name of the collection.
        embedding_model_name (str, optional): Name of the SentenceTransformer model to 
            use if you want ChromaDB to embed queries.
            If None, it is assumed that query embeddings will be provided externally.
        metadata_config (dict, optional): Metadata dictionary for collection creation 
            (e.g. {"hnsw:space": "cosine"}).
    Returns:
        chromadb.Collection: ChromaDB collection object or None if error.
    """
    if not client:
        logger.error("Error: ChromaDB client not provided to get_or_create_chroma_collection.")
        return None
    try:
        ef = None
        if embedding_model_name:
            try:
                # This requires that sentence-transformers is installed in the ChromaDB 
                # environment or that you have a custom embedding function that 
                # conforms.
                ef = SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model_name
                )
                logger.info(f"Embedding function for Chroma '{embedding_model_name}' set.")
            except Exception as e_ef:
                logger.warning(
                    f"Warning: Unable to create SentenceTransformerEmbeddingFunction "
                    f"for {embedding_model_name}: {e_ef}. The collection will be "
                    "created without a specific embedding function (pre-computed "
                    "vectors expected)."
                )

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata=metadata_config # es. {"hnsw:space": "cosine"}
        )

        logger.info(
            f"Collection '{collection_name}' obtained/created. "
            f"Items: {collection.count()}"
        )
        return collection
    except Exception as e:
        logger.error(
            f"Error during get/create of ChromaDB collection '{collection_name}': {e}"
        )
        return None

def add_to_collection(
        collection: chromadb.Collection,
        texts: list,
        embeddings: list,
        metadatas: list,
        ids: list,
        batch_size: int = 1000
    ):
    """
    Adds documents, embeddings, metadata, and IDs to a ChromaDB collection.
    Handles batch addition if necessary (ChromaDB does this internally for `add`).

    Args:
        collection (chromadb.Collection): ChromaDB collection object.
        texts (list): List of strings.
        embeddings (list): List of vectors (embeddings) for each text.
        metadatas (list): List of metadata dictionaries for each text.
        ids (list): List of unique IDs for each text.
        batch_size (int): Batch size for adding data to the collection.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if not collection:
        logger.error("Error: ChromaDB collection not provided to add_to_collection.")
        return False
    if not len(texts) == len(embeddings) == len(metadatas) == len(ids):
        logger.error(
            "Error: The lists of texts, embeddings, metadatas and IDs must have the ", 
            "same length."
        )
        return False
    if not texts:
        logger.info("No data to add to the collection.")
        return True
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # Add the batch to the collection
            collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        logger.info(
            f"Added {len(ids)} items to collection '{collection.name}'. "
            f"Total now: {collection.count()}"
        )
        return True
    except Exception as e:
        logger.error(
            f"Error adding data to ChromaDB collection '{collection.name}': {e}"
        )
        return False


def retrieve_from_collection(
        collection: chromadb.Collection,
        query_embeddings: list,
        n_results: int = 5,
        metadata_filter: dict = None, # Example: {"course_name": "Deep Learning"}
        include_fields: list = None
    ):
    """
    Does a similarity query on the collection and returns the results.

    Args:
        collection (chromadb.Collection): ChromaDB collection object.
        query_embeddings (list): List of query vectors. ChromaDB expects a list of 
            embeddings.
        n_results (int): Number of results to return per query.
        metadata_filter (dict, optional): Dictionary to filter results on metadata.
        include_fields (list, optional): List of fields to include in the results 
            (e.g. 'documents', 'metadatas', 'distances'). If None, ChromaDB uses its 
            defaults (usually includes everything except embeddings).
    Returns:
        dict: Dictionary of results from the ChromaDB query, or None if error.

            Typical format: {
                'ids': [[id1,..]], 
                'documents': [[doc1,..]],
                'metadatas': [[meta1,..]],
                'distances': [[dist1,..]]
            }

            Note the double brackets: it's a list of results for each query_embedding 
            passed.
            If you pass a single embedding, you'll access results['documents'][0], etc.
    """
    if not collection:
        logger.error("Error: ChromaDB collection not provided to retrieve_from_collection.")
        return None
    if query_embeddings is None:
        logger.error("Error: no query_embedding provided.")
        return None

    if (
        isinstance(query_embeddings, list) 
        and query_embeddings 
        and not isinstance(query_embeddings[0], list)
    ):
        # It's a list of numbers (a single embedding), wrap it
        query_embeddings_for_chroma = [query_embeddings]
    elif hasattr(query_embeddings, 'ndim') and query_embeddings.ndim == 1:
        query_embeddings_for_chroma = [query_embeddings.tolist()]
    else:
        query_embeddings_for_chroma = query_embeddings

    include_params = include_fields if include_fields else [
        'documents', 'metadatas', 'distances'
    ]

    try:
        results = collection.query(
            query_embeddings=query_embeddings_for_chroma,
            n_results=n_results,
            where=metadata_filter,
            include=include_params
        )
        return results
    except Exception as e:
        logger.error(f"Error querying ChromaDB collection '{collection.name}': {e}")
        return None
