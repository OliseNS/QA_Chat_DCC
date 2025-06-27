#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import string

class RAGRetriever:
    """Enhanced Retrieval system using hybrid semantic and keyword search."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the retriever with embeddings."""
        if data_dir is None:
            data_dir = Path("data/embeddings/faiss")
        
        self.data_dir = data_dir
        self.embedding_model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: Optional[List[Dict]] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None  # Sparse matrix from TF-IDF
        
        self._load_resources()
    
    def _load_resources(self):
        """Load all necessary resources."""
        try:
            # Load embedding model with higher quality model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load chunks metadata
            metadata_path = self.data_dir / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load embeddings
            embeddings_path = self.data_dir / "embeddings.npy"
            self.embeddings = np.load(embeddings_path)
            
            # Initialize TF-IDF for keyword-based search
            self._initialize_tfidf()
            
            print(f"Loaded enhanced retriever with {len(self.chunks) if self.chunks else 0} chunks")
            
        except Exception as e:
            print(f"Error loading retrieval resources: {e}")
            print("Please ensure data files exist in the data/embeddings/faiss directory.")
            raise
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer for keyword search."""
        if not self.chunks:
            return
        
        # Extract texts for TF-IDF
        texts = [chunk['content'] for chunk in self.chunks]
        
        # Create TF-IDF vectorizer with custom preprocessing
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Include alphanumeric tokens
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Expand common dialysis abbreviations and medical terms
        expansions = {
            'hd': 'hemodialysis',
            'pd': 'peritoneal dialysis',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'dcc': 'dialysis care center',
            'dialysis care center': 'dcc',
            'kidney failure': 'renal failure end stage kidney disease',
            'kidney disease': 'renal disease nephrology',
            'blood cleaning': 'hemodialysis filtration',
            'home dialysis': 'peritoneal dialysis home hemodialysis',
            'treatment': 'therapy dialysis care',
            'appointment': 'visit session treatment',
            'cost': 'price insurance coverage',
            'schedule': 'appointment time frequency',
            'location': 'address center facility',
            'staff': 'team doctors nurses technicians',
        }
        
        # Apply expansions with word boundaries
        for abbr, expansion in expansions.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', f"{abbr} {expansion}", text)
        
        return text.strip()
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        return self._preprocess_text(query)
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.15) -> List[Dict]:
        """
        Enhanced hybrid semantic and keyword search with quality control.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (lowered for better recall)
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.embedding_model or self.embeddings is None or not self.chunks:
            raise ValueError("Retriever not properly initialized")
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # 1. SEMANTIC SEARCH using dense embeddings
        semantic_results = self._semantic_search(processed_query, top_k * 3)
        
        # 2. KEYWORD SEARCH using TF-IDF
        keyword_results = self._keyword_search(processed_query, top_k * 3)
        
        # 3. HYBRID FUSION - combine both approaches
        fused_results = self._fuse_results(semantic_results, keyword_results, query)
        
        # 4. Quality filtering and diversity
        final_results = self._apply_quality_filter(fused_results, similarity_threshold, top_k)
        
        return final_results
    
    def _apply_quality_filter(self, results: List[Dict], threshold: float, top_k: int) -> List[Dict]:
        """Apply quality filtering and ensure diversity in results."""
        if not results:
            return []
        
        # First pass: filter by threshold
        filtered_results = [r for r in results if r['similarity'] >= threshold]
        
        # If we don't have enough results, lower the threshold slightly
        if len(filtered_results) < 2 and results:
            lower_threshold = max(0.1, threshold - 0.1)
            filtered_results = [r for r in results if r['similarity'] >= lower_threshold]
        
        # Ensure diversity by avoiding too many results from the same category
        diverse_results = []
        category_counts = {}
        
        for result in filtered_results:
            category = result.get('category', 'unknown')
            count = category_counts.get(category, 0)
            
            # Allow max 2 results per category unless we have very few results
            if count < 2 or len(diverse_results) < 2:
                diverse_results.append(result)
                category_counts[category] = count + 1
                
            if len(diverse_results) >= top_k:
                break
        
        return diverse_results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform dense vector semantic search."""
        if not self.embedding_model or self.embeddings is None or not self.chunks:
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['semantic_score'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform sparse keyword-based search using TF-IDF."""
        if (self.tfidf_vectorizer is None or self.tfidf_matrix is None or 
            not self.chunks):
            return []
        
        # Transform query using fitted TF-IDF vectorizer
        query_tfidf = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = linear_kernel(query_tfidf, self.tfidf_matrix)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero matches
                chunk = self.chunks[idx].copy()
                chunk['keyword_score'] = float(similarities[idx])
                results.append(chunk)
        
        return results
    
    def _fuse_results(self, semantic_results: List[Dict], keyword_results: List[Dict], 
                     original_query: str) -> List[Dict]:
        """Fuse semantic and keyword search results with intelligent weighting."""
        
        # Create a mapping from chunk index to results
        chunk_scores = {}
        
        # Add semantic scores (weight: 0.7 for semantic understanding)
        for result in semantic_results:
            chunk_idx = self._get_chunk_index(result)
            if chunk_idx is not None:
                chunk_scores[chunk_idx] = {
                    'chunk': result,
                    'semantic_score': result.get('semantic_score', 0.0),
                    'keyword_score': 0.0
                }
        
        # Add keyword scores (weight: 0.3 for exact matches)
        for result in keyword_results:
            chunk_idx = self._get_chunk_index(result)
            if chunk_idx is not None:
                if chunk_idx in chunk_scores:
                    chunk_scores[chunk_idx]['keyword_score'] = result.get('keyword_score', 0.0)
                else:
                    chunk_scores[chunk_idx] = {
                        'chunk': result,
                        'semantic_score': 0.0,
                        'keyword_score': result.get('keyword_score', 0.0)
                    }
        
        # Calculate fusion scores with adaptive weighting
        fused_results = []
        for chunk_idx, scores in chunk_scores.items():
            semantic_score = scores['semantic_score']
            keyword_score = scores['keyword_score']
            
            # Adaptive weighting based on query characteristics
            query_len = len(original_query.split())
            if query_len <= 3:  # Short queries - favor keyword matching
                semantic_weight = 0.6
                keyword_weight = 0.4
            else:  # Longer queries - favor semantic understanding
                semantic_weight = 0.8
                keyword_weight = 0.2
            
            # Calculate final fusion score
            fusion_score = (semantic_weight * semantic_score + 
                          keyword_weight * keyword_score)
            
            # Boost score if both methods found the chunk
            if semantic_score > 0 and keyword_score > 0:
                fusion_score *= 1.2  # 20% boost for consensus
            
            # Create final result
            result = scores['chunk'].copy()
            result['similarity'] = fusion_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            
            fused_results.append(result)
        
        # Sort by fusion score
        fused_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return fused_results
    
    def _get_chunk_index(self, chunk: Dict) -> Optional[int]:
        """Get the index of a chunk in the chunks list."""
        try:
            if not self.chunks:
                return None
                
            # Try to find chunk by content matching
            content = chunk.get('content', '')
            for i, original_chunk in enumerate(self.chunks):
                if original_chunk.get('content') == content:
                    return i
            return None
        except:
            return None
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        if not self.chunks:
            return {"total_chunks": 0, "total_files": 0, "hybrid_search": False}
        
        categories = set(chunk['category'] for chunk in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_files": len(categories),
            "hybrid_search": self.tfidf_vectorizer is not None,
            "semantic_model": "all-MiniLM-L6-v2",
            "search_methods": ["semantic_dense", "keyword_sparse", "hybrid_fusion"]
        }


def test_retriever():
    """Test the enhanced retriever functionality."""
    try:
        retriever = RAGRetriever()
        
        test_queries = [
            "What dialysis treatments are available?",
            "How can I contact DCC?", 
            "Tell me about the mission and values",
            "What is DCC Cares?",
            "Where are the dialysis centers located?",
            "home dialysis options",  # Test expansion
            "kidney failure treatment",  # Test medical term expansion
            "cost and insurance",  # Test keyword search
        ]
        
        print("=== TESTING ENHANCED RAG RETRIEVER ===\n")
        print(f"Retriever Stats: {retriever.get_stats()}\n")
        
        for query in test_queries:
            print(f"Query: '{query}'")
            results = retriever.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Overall Score: {result['similarity']:.3f}")
                print(f"  Semantic Score: {result.get('semantic_score', 0):.3f}")
                print(f"  Keyword Score: {result.get('keyword_score', 0):.3f}")
                print(f"  Category: {result['category']}")
                print(f"  Content: {result['content'][:150]}...")
            
            print("\n" + "="*80 + "\n")
            
    except Exception as e:
        print(f"Error testing enhanced retriever: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_retriever()
