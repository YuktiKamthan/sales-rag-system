"""
Embeddings Module
-----------------
Converts text descriptions into vector embeddings for semantic search.

This is the "translation layer" that converts human language into 
numbers that computers can compare and search.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
from .telemetry import telemetry, metrics
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Creates embeddings from text using sentence transformers.
    
    Embeddings are like "fingerprints" for text - similar texts get
    similar fingerprints, which lets us find related information quickly.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
                       Default is fast and efficient for most tasks
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initializing embedding model: {model_name}")
        
    @telemetry.trace_operation("load_embedding_model")
    def load_model(self):
        """
        Load the sentence transformer model into memory.
        
        This downloads the model if needed (first time only).
        """
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(self.model_name)
            duration = time.time() - start_time
            
            logger.info(f"Model loaded in {duration:.2f} seconds")
            metrics.record('embedding_model_load', duration, success=True)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            metrics.record('embedding_model_load', time.time() - start_time, success=False)
            raise
    
    @telemetry.trace_operation("create_embeddings")
    def create_embeddings(self, texts: Union[str, List[str]], 
                         batch_size: int = 32,
                         show_progress: bool = True) -> np.ndarray:
        """
        Convert text(s) into vector embeddings.
        
        Args:
            texts: Single text string or list of texts
            batch_size: How many texts to process at once (larger = faster but more memory)
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if self.model is None:
            self.load_model()
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        start_time = time.time()
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        
        try:
            # Create embeddings in batches for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            duration = time.time() - start_time
            logger.info(f"Created {len(embeddings)} embeddings in {duration:.2f} seconds")
            logger.info(f"Embedding dimension: {embeddings.shape[1]}")
            
            metrics.record('embedding_creation', duration, success=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            metrics.record('embedding_creation', time.time() - start_time, success=False)
            raise
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        if self.model is None:
            self.load_model()
        
        # Get dimension by encoding a dummy text
        dummy_embedding = self.model.encode(["test"])
        return dummy_embedding.shape[1]
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Similarity ranges from -1 to 1:
        - 1 = identical
        - 0 = unrelated
        - -1 = opposite
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score
        """
        # Normalize vectors
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute dot product (cosine similarity)
        similarity = np.dot(norm1, norm2)
        
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query.
        
        Args:
            query_embedding: The query embedding (1D array)
            candidate_embeddings: Pool of embeddings to search (2D array)
            top_k: How many similar items to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for idx, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append((idx, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Helper function for easy usage
def create_embeddings_from_texts(texts: List[str], 
                                 model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Quick function to create embeddings from texts.
    
    Args:
        texts: List of text strings
        model_name: Embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    generator = EmbeddingGenerator(model_name)
    return generator.create_embeddings(texts)


if __name__ == "__main__":
    # Test the embedding generator
    print("Testing Embedding Generator...")
    
    test_texts = [
        "Sales of product A in January 2024 for $1000",
        "Product B sold in February 2024 for $500",
        "Product A revenue in January was high"
    ]
    
    generator = EmbeddingGenerator()
    embeddings = generator.create_embeddings(test_texts, show_progress=True)
    
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = generator.compute_similarity(embeddings[0], embeddings[2])
    print(f"\nSimilarity between text 1 and 3: {sim:.4f}")
    print("(They should be similar since both mention Product A in January)")
