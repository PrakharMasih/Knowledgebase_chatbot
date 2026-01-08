"""
Embedding function configuration
"""
from chromadb.utils import embedding_functions
import logging


logger = logging.getLogger(__name__)


def get_embedding_function():
    """
    Get the embedding function based on configuration

    Returns:
        Embedding function for ChromaDB
    """
    try:
        model_name = "all-MiniLM-L6-v2"  # Good for general text
        # model_name = "all-mpnet-base-v2" #- Higher quality but slower

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device="cpu",
            normalize_embeddings=True
        )

    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {str(e)}")
        raise