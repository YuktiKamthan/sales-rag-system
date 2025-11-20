"""
Vector Store Module
-------------------
Manages the ChromaDB vector database for storing and searching embeddings.

This is like a specialized search engine for finding similar sales records
based on meaning, not just keywords.
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from .telemetry import telemetry, metrics
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesVectorStore:
    """
    Vector database for storing and searching sales data embeddings.
    
    Think of this as a smart library where you can ask questions
    and it finds the most relevant sales records.
    """
    
    def __init__(self, 
                 collection_name: str = "sales_data",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for this collection (like a database table)
            persist_directory: Where to save the database on disk
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        logger.info(f"Initializing vector store: {collection_name}")
    
    @telemetry.trace_operation("initialize_vector_store")
    def initialize(self):
        """
        Set up the ChromaDB client and collection.
        """
        start_time = time.time()
        
        try:
            # Create persistent client (data saved to disk)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Sales transaction embeddings"}
            )
            
            duration = time.time() - start_time
            logger.info(f"Vector store initialized in {duration:.2f} seconds")
            logger.info(f"Collection contains {self.collection.count()} records")
            
            metrics.record('vector_store_init', duration, success=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            metrics.record('vector_store_init', time.time() - start_time, success=False)
            raise
    
    @telemetry.trace_operation("add_documents")
    def add_documents(self, 
                     documents: List[str],
                     embeddings: np.ndarray,
                     metadatas: List[Dict],
                     ids: Optional[List[str]] = None,
                     batch_size: int = 100):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of text descriptions
            embeddings: Corresponding embeddings (numpy array)
            metadatas: List of metadata dicts (e.g., {SKU, Date, Amount})
            ids: Optional list of unique IDs (will generate if not provided)
            batch_size: How many to add at once
        """
        if self.collection is None:
            self.initialize()
        
        start_time = time.time()
        num_docs = len(documents)
        
        logger.info(f"Adding {num_docs} documents to vector store...")
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(num_docs)]
            
            # Convert embeddings to list format for ChromaDB
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
            # Add in batches for efficiency
            for i in range(0, num_docs, batch_size):
                end_idx = min(i + batch_size, num_docs)
                
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings_list[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                self.collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(num_docs-1)//batch_size + 1}")
            
            duration = time.time() - start_time
            logger.info(f"Successfully added {num_docs} documents in {duration:.2f} seconds")
            logger.info(f"Total documents in collection: {self.collection.count()}")
            
            metrics.record('add_documents', duration, success=True)
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            metrics.record('add_documents', time.time() - start_time, success=False)
            raise
    
    @telemetry.trace_operation("search_similar")
    def search_similar(self, 
                      query_embedding: np.ndarray,
                      top_k: int = 5,
                      filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: The query vector
            top_k: How many similar results to return
            filter_metadata: Optional filters (e.g., {"Year": 2024})
            
        Returns:
            Dictionary with documents, distances, and metadatas
        """
        if self.collection is None:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert to list if numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Search the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            duration = time.time() - start_time
            logger.info(f"Search completed in {duration:.2f} seconds, found {len(results['documents'][0])} results")
            
            metrics.record('vector_search', duration, success=True)
            
            # Format results nicely
            formatted_results = {
                'documents': results['documents'][0],
                'distances': results['distances'][0],
                'metadatas': results['metadatas'][0],
                'ids': results['ids'][0]
            }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            metrics.record('vector_search', time.time() - start_time, success=False)
            raise
    
    def search_by_text(self, 
                      query_text: str,
                      embedding_generator,
                      top_k: int = 5,
                      filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search using natural language text (not embeddings).
        
        This is the user-friendly version - you just type a question!
        
        Args:
            query_text: Natural language query (e.g., "Show me sales from January 2024")
            embedding_generator: EmbeddingGenerator instance to convert text
            top_k: How many results to return
            filter_metadata: Optional filters
            
        Returns:
            Dictionary with search results
        """
        # Convert query text to embedding
        query_embedding = embedding_generator.create_embeddings(query_text, show_progress=False)
        
        # Search using the embedding
        return self.search_similar(query_embedding[0], top_k, filter_metadata)
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        if self.collection is None:
            self.initialize()
        
        stats = {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count(),
            'persist_directory': self.persist_directory
        }
        
        return stats
    
    def clear_collection(self):
        """
        Delete all documents from the collection.
        Use with caution!
        """
        if self.collection is None:
            self.initialize()
        
        logger.warning("Clearing all documents from collection...")
        
        # Get all IDs and delete them
        all_data = self.collection.get()
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
            logger.info(f"Deleted {len(all_data['ids'])} documents")
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export all documents and metadata to a pandas DataFrame.
        
        Returns:
            DataFrame with all collection data
        """
        if self.collection is None:
            self.initialize()
        
        # Get all data
        all_data = self.collection.get(include=['documents', 'metadatas'])
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data['metadatas'])
        df['document'] = all_data['documents']
        df['id'] = all_data['ids']
        
        return df


if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store...")
    
    # Create a test store
    store = SalesVectorStore(collection_name="test_sales")
    store.initialize()
    
    # Create some test data
    test_docs = [
        "Sale of Product A in January 2024 for $1000",
        "Product B sold in February 2024 for $500"
    ]
    
    test_embeddings = np.random.rand(2, 384)  # Dummy embeddings
    
    test_metadata = [
        {"SKU": "A", "Amount": 1000, "Month": "January"},
        {"SKU": "B", "Amount": 500, "Month": "February"}
    ]
    
    # Add documents
    store.add_documents(test_docs, test_embeddings, test_metadata)
    
    # Get stats
    stats = store.get_collection_stats()
    print(f"\nCollection stats: {stats}")
    
    # Clean up
    store.clear_collection()
    print("\nTest complete!")
