#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import re
from sklearn.metrics.pairwise import cosine_similarity

class RAGRetriever:
    """Retrieval system using sentence transformers for semantic search."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the retriever with embeddings."""
        if data_dir is None:
            data_dir = Path("data/embeddings/faiss")
        
        self.data_dir = data_dir
        self.embedding_model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: Optional[List[Dict]] = None
        
        self._load_resources()
    
    def _load_resources(self):
        """Load all necessary resources."""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load chunks metadata
            metadata_path = self.data_dir / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load embeddings
            embeddings_path = self.data_dir / "embeddings.npy"
            self.embeddings = np.load(embeddings_path)
            
            print(f"Loaded retriever with {len(self.chunks) if self.chunks else 0} chunks")
            
        except Exception as e:
            print(f"Error loading retrieval resources: {e}")
            print("Please ensure data files exist in the data/embeddings/faiss directory.")
            raise
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        query = query.lower()
        
        # Expand common dialysis abbreviations
        expansions = {
            'hd': 'hemodialysis',
            'pd': 'peritoneal dialysis',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'dcc': 'dialysis care center',
        }
        
        for abbr, expansion in expansions.items():
            query = re.sub(r'\b' + abbr + r'\b', expansion, query)
        
        return query
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.embedding_model or self.embeddings is None or not self.chunks:
            raise ValueError("Retriever not properly initialized")
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([processed_query])
        
        # Calculate similarities using cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < similarity_threshold:
                continue
                
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = float(similarity)
            results.append(chunk)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        if not self.chunks:
            return {"total_chunks": 0, "total_files": 0}
        
        categories = set(chunk['category'] for chunk in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_files": len(categories)
        }


def test_retriever():
    """Test the retriever functionality."""
    try:
        retriever = RAGRetriever()
        
        test_queries = [
            "What dialysis treatments are available?",
            "How can I contact DCC?",
            "Tell me about the mission and values",
            "What is DCC Cares?",
            "Where are the dialysis centers located?"
        ]
        
        print("=== TESTING RAG RETRIEVER ===\n")
        
        for query in test_queries:
            print(f"Query: {query}")
            results = retriever.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (Score: {result['similarity']:.3f}):")
                print(f"Category: {result['category']}")
                print(f"Content: {result['content'][:200]}...")
            
            print("\n" + "="*80 + "\n")
            
    except Exception as e:
        print(f"Error testing retriever: {e}")


if __name__ == "__main__":
    test_retriever()
