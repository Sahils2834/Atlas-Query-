
import os
import logging
from functools import lru_cache

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


def get_model_name():
    if MODEL_NAME.startswith("sentence-transformers/"):
        return MODEL_NAME

    return f"sentence-transformers/{MODEL_NAME}"


@lru_cache(maxsize=1)
def get_embedder():
    """
    Load FastEmbed only once per Python process.
    Shared by ingestion and retrieval.
    """
    model_name = get_model_name()

    logger.info(
        "Loading shared FastEmbed model: %s",
        model_name
    )

    return TextEmbedding(
        model_name=model_name,
        threads=1
    )
