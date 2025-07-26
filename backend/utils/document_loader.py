import os
import tempfile
import PyPDF2
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Union
import logging

class DocumentLoader:
    """Document loader for PDF, HTML, and Markdown files."""
    
    def __init__(self):
        """Initialize the document loader."""
        self.logger = logging.getLogger(__name__)
    
    def process_uploaded_file(self, file) -> Dict[str, Any]:
        """
        Process an uploaded file.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Processing result
        """
        try:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                file.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Determine file type and process accordingly
            filename = file.filename.lower()
            if filename.endswith('.pdf'):
                content = self._extract_text_from_pdf(tmp_file_path)
                doc_type = "pdf"
            elif filename.endswith('.html') or filename.endswith('.htm'):
                content = self._extract_text_from_html(tmp_file_path)
                doc_type = "html"
            elif filename.endswith('.md') or filename.endswith('.markdown'):
                content = self._extract_text_from_markdown(tmp_file_path)
                doc_type = "markdown"
            else:
                # Try to read as plain text
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_type = "text"
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return {
                "status": "success",
                "document_type": doc_type,
                "content_length": len(content),
                "message": f"Successfully processed {doc_type} document"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing uploaded file: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process uploaded document"
            }
    
    def load_documents_from_directory(self, directory: str) -> Dict[str, Any]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Directory path to load documents from
            
        Returns:
            Loading result
        """
        try:
            if not os.path.exists(directory):
                return {
                    "status": "error",
                    "error": f"Directory does not exist: {directory}",
                    "documents": []
                }
            
            documents = []
            supported_extensions = ['.pdf', '.html', '.htm', '.md', '.markdown', '.txt']
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Check if file extension is supported
                _, ext = os.path.splitext(filename)
                if ext.lower() not in supported_extensions:
                    continue
                
                try:
                    if ext.lower() == '.pdf':
                        content = self._extract_text_from_pdf(file_path)
                        doc_type = "pdf"
                    elif ext.lower() in ['.html', '.htm']:
                        content = self._extract_text_from_html(file_path)
                        doc_type = "html"
                    elif ext.lower() in ['.md', '.markdown']:
                        content = self._extract_text_from_markdown(file_path)
                        doc_type = "markdown"
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        doc_type = "text"
                    
                    documents.append({
                        "filename": filename,
                        "file_path": file_path,
                        "document_type": doc_type,
                        "content": content,
                        "content_length": len(content)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
                    continue
            
            return {
                "status": "success",
                "documents": documents,
                "count": len(documents),
                "message": f"Successfully loaded {len(documents)} documents"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading documents from directory: {e}")
            return {
                "status": "error",
                "error": str(e),
                "documents": [],
                "message": "Failed to load documents from directory"
            }
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _extract_text_from_html(self, file_path: str) -> str:
        """
        Extract text from an HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from HTML: {e}")
            raise
    
    def _extract_text_from_markdown(self, file_path: str) -> str:
        """
        Extract text from a Markdown file (treat as plain text).
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error extracting text from Markdown: {e}")
            raise
    
    def load_document_from_url(self, url: str) -> Dict[str, Any]:
        """
        Load a document from a URL.
        
        Args:
            url: URL to load document from
            
        Returns:
            Loading result
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type:
                # Save to temporary file and extract text
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name
                
                content = self._extract_text_from_pdf(tmp_file_path)
                os.unlink(tmp_file_path)
                doc_type = "pdf"
            elif 'html' in content_type:
                # Parse HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text()
                
                # Break into lines and remove leading/trailing space
                lines = (line.strip() for line in text.splitlines())
                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # Drop blank lines
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                doc_type = "html"
            else:
                # Treat as plain text
                content = response.text
                doc_type = "text"
            
            return {
                "status": "success",
                "document_type": doc_type,
                "content": content,
                "content_length": len(content),
                "url": url,
                "message": f"Successfully loaded {doc_type} document from URL"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading document from URL: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url,
                "message": "Failed to load document from URL"
            }
    
    def add_document_to_knowledge_base(self, content: str, source_name: str) -> Dict[str, Any]:
        """
        Add a new document to the knowledge base by processing it into chunks and generating embeddings.
        
        Args:
            content: The text content of the document
            source_name: The name/source of the document
            
        Returns:
            Result of the operation
        """
        try:
            # Validate inputs
            if not content or not content.strip():
                return {
                    "status": "error",
                    "error": "No content provided",
                    "message": "Document content is empty"
                }
            
            if not source_name or not source_name.strip():
                source_name = "unnamed_document"
            
            # Import required modules
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            from utils.process_to_chunks import TextChunker
            from utils.generate_embeddings import EmbeddingGenerator
            import json
            import numpy as np
            
            # Create a temporary directory for processing
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save content to a temporary file
                # Sanitize filename
                safe_source_name = "".join(c for c in source_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                if not safe_source_name:
                    safe_source_name = "unnamed_document"
                    
                temp_file_path = os.path.join(temp_dir, f"{safe_source_name}.txt")
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Create directories for chunks and embeddings
                chunks_dir = os.path.join(temp_dir, "chunks")
                embeddings_dir = os.path.join(temp_dir, "embeddings")
                
                # Process text into chunks
                chunker = TextChunker(chunk_size=200, overlap_size=50)
                documents, metadata = chunker.process_files(temp_dir, chunks_dir)
                
                if not documents:
                    return {
                        "status": "error",
                        "error": "No chunks were created from the document",
                        "message": "Failed to process document into chunks"
                    }
                
                # Generate embeddings for chunks
                embedding_generator = EmbeddingGenerator()
                embedding_generator.process_chunks_directory(chunks_dir, embeddings_dir)
                
                # Load the generated embeddings and metadata
                faiss_dir = os.path.join(embeddings_dir, 'faiss')
                metadata_file = os.path.join(faiss_dir, 'metadata.json')
                embeddings_file = os.path.join(faiss_dir, 'embeddings.npy')
                
                if not os.path.exists(metadata_file) or not os.path.exists(embeddings_file):
                    return {
                        "status": "error",
                        "error": "Failed to generate embeddings",
                        "message": "Embedding generation process failed"
                    }
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    new_metadata = json.load(f)
                
                new_embeddings = np.load(embeddings_file)
                
                # Update the main knowledge base
                main_embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'embeddings')
                main_faiss_dir = os.path.join(main_embeddings_dir, 'faiss')
                
                # Load existing metadata and embeddings
                main_metadata_file = os.path.join(main_faiss_dir, 'metadata.json')
                main_embeddings_file = os.path.join(main_faiss_dir, 'embeddings.npy')
                
                if os.path.exists(main_metadata_file) and os.path.exists(main_embeddings_file):
                    with open(main_metadata_file, 'r', encoding='utf-8') as f:
                        existing_metadata = json.load(f)
                    
                    existing_embeddings = np.load(main_embeddings_file)
                    
                    # Combine existing and new data
                    combined_metadata = existing_metadata + new_metadata
                    combined_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
                else:
                    # No existing data, use new data only
                    combined_metadata = new_metadata
                    combined_embeddings = new_embeddings
                
                # Save combined data
                os.makedirs(main_faiss_dir, exist_ok=True)
                
                with open(main_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_metadata, f, ensure_ascii=False, indent=2)
                
                np.save(main_embeddings_file, combined_embeddings)
                
                return {
                    "status": "success",
                    "chunks_created": len(new_metadata),
                    "message": f"Successfully added {len(new_metadata)} chunks to knowledge base"
                }
                
        except Exception as e:
            self.logger.error(f"Error adding document to knowledge base: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to add document to knowledge base"
            }