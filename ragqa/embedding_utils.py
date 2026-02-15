import logging

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}


def get_embedding_model(
    model_name: str, device: str | None = None
) -> SentenceTransformer:
    """Load (and cache) a SentenceTransformer model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (model_name, device)

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, device=device)

    return _MODEL_CACHE[cache_key]


def encode_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = False,
    to_numpy: bool = True,
    normalize: bool = True,
):
    """Encode a list of texts into embeddings."""
    if not texts:
        logger.debug("No texts provided for embedding.")
        return []

    logger.info("Encoding %d texts", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=to_numpy,
        normalize_embeddings=normalize,
    )

    logger.info("Generated %d embeddings.", len(embeddings))
    return embeddings
