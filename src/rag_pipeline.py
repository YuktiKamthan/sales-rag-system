"""
RAG Pipeline Module
-------------------
Main pipeline that orchestrates the entire RAG system.

This connects:
1. Data Loading
2. Embeddings Creation
3. Vector Storage
4. LLM Generation
"""

import google.generativeai as genai
from typing import List, Dict, Optional, Any
import logging
from .data_loader import SalesDataLoader
from .embeddings import EmbeddingGenerator
from .vector_store import SalesVectorStore
from .telemetry import telemetry, metrics
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesRAGPipeline:
    """
    Complete RAG pipeline for sales data analysis.
    
    This is the "brain" that combines:
    - Your sales data
    - Vector search
    - AI language model
    """
    
    def __init__(self, 
                 data_path: str,
                 gemini_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "sales_data"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_path: Path to sales data Excel file
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            embedding_model: Sentence transformer model name
            collection_name: Name for vector database collection
        """
        self.data_path = data_path
        self.collection_name = collection_name
        
        # Initialize components
        self.data_loader = SalesDataLoader(data_path)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store = SalesVectorStore(collection_name)
        
        # Setup Gemini
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            
            # Try to find an available model
            try:
                # List available models
                available_models = genai.list_models()
                generation_models = [m for m in available_models 
                                   if 'generateContent' in m.supported_generation_methods]
                
                if generation_models:
                    # Use the first available generation model
                    model_name = generation_models[0].name
                    logger.info(f"Using model: {model_name}")
                    self.llm = genai.GenerativeModel(model_name)
                else:
                    logger.warning("No generation models available")
                    self.llm = None
            except Exception as e:
                logger.error(f"Error setting up Gemini: {e}")
                # Fallback to trying gemini-pro without prefix
                try:
                    self.llm = genai.GenerativeModel('gemini-pro')
                except:
                    self.llm = None
        else:
            logger.warning("No Gemini API key provided. LLM features will be limited.")
            self.llm = None
        
        self.df = None
        self.is_initialized = False
        
        logger.info("RAG Pipeline created")
    
    @telemetry.trace_operation("initialize_pipeline")
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the entire pipeline.
        
        This does:
        1. Load and process sales data
        2. Create embeddings
        3. Store in vector database
        
        Args:
            force_rebuild: If True, rebuild everything from scratch
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and process data
            logger.info("Step 1: Loading and processing data...")
            self.df, stats = self.data_loader.prepare_for_rag()
            logger.info(f"Data loaded: {stats}")
            
            # Step 2: Initialize vector store
            logger.info("Step 2: Initializing vector store...")
            self.vector_store.initialize()
            
            # Check if we need to rebuild
            existing_count = self.vector_store.collection.count()
            
            if existing_count > 0 and not force_rebuild:
                logger.info(f"Vector store already has {existing_count} documents. Skipping embedding creation.")
                logger.info("Use force_rebuild=True to rebuild from scratch.")
            else:
                # Step 3: Create embeddings
                logger.info("Step 3: Creating embeddings...")
                if force_rebuild and existing_count > 0:
                    self.vector_store.clear_collection()
                
                texts = self.df['text_description'].tolist()
                embeddings = self.embedding_generator.create_embeddings(
                    texts, 
                    batch_size=32,
                    show_progress=True
                )
                
                # Step 4: Prepare metadata
                logger.info("Step 4: Preparing metadata...")
                metadatas = []
                for _, row in self.df.iterrows():
                    metadata = {
                        'SKU': str(row['SKU']),
                        'Date': row['Date'].strftime('%Y-%m-%d'),
                        'Year': int(row['Year']),
                        'Month': int(row['Month']),
                        'Quarter': int(row['Quarter']),
                        'Amount': float(row['Amount']),
                        'Qty': int(row['Qty']),
                        'Customer': str(row['Name']),
                        'Season': str(row['Season'])
                    }
                    metadatas.append(metadata)
                
                # Step 5: Add to vector store
                logger.info("Step 5: Adding to vector store...")
                self.vector_store.add_documents(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            
            self.is_initialized = True
            duration = time.time() - start_time
            
            logger.info(f"✅ Pipeline initialized successfully in {duration:.2f} seconds!")
            metrics.record('pipeline_initialization', duration, success=True)
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            metrics.record('pipeline_initialization', time.time() - start_time, success=False)
            raise
    
    @telemetry.trace_operation("query_pipeline")
    def query(self, 
             question: str,
             top_k: int = 5,
             use_llm: bool = True,
             temperature: float = 0.3) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a natural language question.
        
        Args:
            question: Your question (e.g., "What were the top products in 2024?")
            top_k: How many relevant records to retrieve
            use_llm: Whether to use LLM for answer generation
            temperature: LLM creativity (0=focused, 1=creative)
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        if not self.is_initialized:
            logger.warning("Pipeline not initialized. Initializing now...")
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Step 1: Search for relevant documents
            logger.info(f"Searching for: {question}")
            
            results = self.vector_store.search_by_text(
                query_text=question,
                embedding_generator=self.embedding_generator,
                top_k=top_k
            )
            
            # Step 2: Format context from results
            context_parts = []
            for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
                context_parts.append(f"Record {i+1}:\n{doc}\n")
            
            context = "\n".join(context_parts)
            
            # Step 3: Generate answer with LLM (if available and requested)
            answer = None
            if use_llm and self.llm:
                answer = self._generate_llm_answer(question, context, temperature)
            else:
                answer = "LLM not available. Here are the relevant records:\n\n" + context
            
            duration = time.time() - start_time
            metrics.record('query_execution', duration, success=True)
            
            return {
                'question': question,
                'answer': answer,
                'context': context,
                'retrieved_records': len(results['documents']),
                'metadata': results['metadatas'],
                'execution_time': duration
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            metrics.record('query_execution', time.time() - start_time, success=False)
            raise
    
    @telemetry.trace_operation("llm_generation")
    def _generate_llm_answer(self, question: str, context: str, temperature: float) -> str:
        """
        Generate answer using Gemini LLM.
        
        Args:
            question: User's question
            context: Retrieved context from vector store
            temperature: Generation temperature
            
        Returns:
            Generated answer
        """
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = f"""You are a sales data analyst. Answer the question based on the provided sales data context.

Question: {question}

Context from sales records:
{context}

Instructions:
- Provide a clear, concise answer based on the data
- Include specific numbers and dates when relevant
- If the data doesn't fully answer the question, say so
- Be professional and analytical

Answer:"""
            
            # Generate response
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000
                )
            )
            
            answer = response.text
            
            duration = time.time() - start_time
            metrics.record('llm_generation', duration, success=True)
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            metrics.record('llm_generation', time.time() - start_time, success=False)
            # Return data without showing the error to user
            return f"Based on the sales records:\n\n{context}"
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the pipeline.
        
        Returns:
            Dictionary with various stats
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        data_stats = self.data_loader.get_aggregated_stats()
        vector_stats = self.vector_store.get_collection_stats()
        perf_stats = metrics.get_summary()
        
        return {
            'data': data_stats,
            'vector_store': vector_stats,
            'performance': perf_stats
        }


if __name__ == "__main__":
    # Test the pipeline
    print("Testing RAG Pipeline...")
    
    pipeline = SalesRAGPipeline(
        data_path="data/sales_data.xlsx",
        collection_name="test_pipeline"
    )
    
    # Initialize
    pipeline.initialize()
    
    # Test query
    result = pipeline.query(
        "What were the sales in 2024?",
        top_k=3,
        use_llm=False  # Set to True if you have Gemini API key
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nExecution time: {result['execution_time']:.2f} seconds")