import logging
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embedding_model_cache = {}

def get_embedding_model(model_name: str, device: str = None) -> SentenceTransformer:
    """
    Load (or retrieve from cache) a SentenceTransformer model.

    Args:
        model_name (str): The name or path of the SentenceTransformer model.
        device (str, optional): The device to load the model on ('cuda' or 'cpu'). 
            Defaults to None.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cache_key = (model_name, device)
    if cache_key not in _embedding_model_cache:
        model = SentenceTransformer(model_name, device=device)
        _embedding_model_cache[cache_key] = model
    return _embedding_model_cache[cache_key]

def generate_embeddings(
        texts: list[str],
        embedding_model: SentenceTransformer,
        batch_size: int = 32,
        show_progress: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
    """
    Generate embeddings for a list of texts using a SentenceTransformer model.

    Args:
        texts (list[str]): List of texts to generate embeddings for.
        embedding_model (SentenceTransformer): The SentenceTransformer model to use.
        batch_size (int, optional): Batch size for encoding.
        show_progress (bool, optional): Whether to show a progress bar.
        convert_to_numpy (bool, optional): Whether to convert embeddings to a NumPy 
            arrays.
        normalize_embeddings (bool, optional): Whether to normalize the embeddings.
    
    Returns:
        list: A list or NumPy array of embeddings.
    """
    if not texts:
        return []
    
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=convert_to_numpy,
        normalize_embeddings=normalize_embeddings,
    )
    logger.info(f"Generated {len(embeddings)} embeddings.")
    return embeddings
