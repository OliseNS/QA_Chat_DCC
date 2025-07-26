import os
import sys
from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime
import json

# Add the parent directory to the path to import rag_retriever
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag_retriever import RAGRetriever
from backend.agents.healthcare_agent import HealthcareAgent
from backend.utils.document_loader import DocumentLoader
from backend.utils.calculator import CalculatorTool

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize components
retriever = None
agent = None

def initialize_components():
    """Initialize the RAG retriever and agent."""
    global retriever, agent
    try:
        retriever = RAGRetriever()
        agent = HealthcareAgent(retriever)
        current_app.logger.info("RAG retriever and agent initialized successfully")
    except Exception as e:
        current_app.logger.error(f"Error initializing components: {e}")
        retriever = None
        agent = None

# Initialize components on first request
@api_bp.before_app_first_request
def setup():
    initialize_components()

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'rag_retriever': retriever is not None,
            'agent': agent is not None
        }
    }), 200

# Chat endpoint
@api_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with the healthcare agent."""
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("No JSON data provided in chat request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_input = data.get('message')
        if not user_input:
            current_app.logger.warning("No message provided in chat request")
            return jsonify({'error': 'No message provided'}), 400
        
        # Get conversation history if provided
        history = data.get('history', [])
        
        if agent is None:
            current_app.logger.error("Agent not initialized")
            return jsonify({
                'error': 'Agent not initialized',
                'message': 'The healthcare agent is not available. Please check the system logs.'
            }), 500
        
        current_app.logger.info(f"Processing chat request: {user_input}")
        
        # Process the query with the agent
        response = agent.process_query(user_input, history)
        
        current_app.logger.info("Chat request processed successfully")
        
        return jsonify({
            'response': response.get('response', ''),
            'context': response.get('context', []),
            'query': response.get('query', user_input),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Document upload endpoint
@api_bp.route('/upload', methods=['POST'])
def upload_document():
    """Handle document uploads for processing."""
    try:
        if 'file' not in request.files:
            current_app.logger.warning("No file provided in upload request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            current_app.logger.warning("No file selected in upload request")
            return jsonify({'error': 'No file selected'}), 400
        
        current_app.logger.info(f"Processing uploaded file: {file.filename}")
        
        # Process the uploaded document
        loader = DocumentLoader()
        process_result = loader.process_uploaded_file(file)
        
        if process_result.get('status') == 'success':
            # Add document to knowledge base
            content = ""
            try:
                # Save file to temporary location to extract content
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    file.seek(0)  # Reset file pointer to beginning
                    file.save(tmp_file.name)
                    tmp_file_path = tmp_file.name
                
                # Extract content based on file type
                filename = file.filename.lower() if file.filename else ""
                if filename.endswith('.pdf'):
                    content = loader._extract_text_from_pdf(tmp_file_path)
                elif filename.endswith('.html') or filename.endswith('.htm'):
                    content = loader._extract_text_from_html(tmp_file_path)
                elif filename.endswith('.md') or filename.endswith('.markdown'):
                    content = loader._extract_text_from_markdown(tmp_file_path)
                else:
                    with open(tmp_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Clean up temporary file
                import os
                os.unlink(tmp_file_path)
            except Exception as e:
                current_app.logger.error(f"Error extracting content from uploaded file: {e}")
                # Continue with empty content if extraction fails
                
            source_name = file.filename if file.filename else "unnamed_upload"
            add_result = loader.add_document_to_knowledge_base(content, source_name)
            
            if add_result.get('status') == 'success':
                current_app.logger.info("Document uploaded and added to knowledge base successfully")
                return jsonify({
                    'message': 'Document uploaded and added to knowledge base successfully',
                    'result': add_result,
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            else:
                current_app.logger.error(f"Error adding document to knowledge base: {add_result.get('error')}")
                return jsonify({
                    'error': 'Failed to add document to knowledge base',
                    'message': add_result.get('error', 'Unknown error')
                }), 500
        else:
            current_app.logger.error(f"Error processing uploaded document: {process_result.get('error')}")
            return jsonify({
                'error': 'Document processing failed',
                'message': process_result.get('error', 'Unknown error')
            }), 500
        
    except Exception as e:
        current_app.logger.error(f"Error in upload endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Document summarization endpoint
@api_bp.route('/summarize', methods=['POST'])
def summarize_document():
    """Generate a summary of a document or text."""
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("No JSON data provided in summarize request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text')
        doc_id = data.get('document_id')
        
        if not text and not doc_id:
            current_app.logger.warning("No text or document_id provided in summarize request")
            return jsonify({'error': 'No text or document_id provided'}), 400
        
        if agent is None:
            current_app.logger.error("Agent not initialized")
            return jsonify({
                'error': 'Agent not initialized',
                'message': 'The healthcare agent is not available. Please check the system logs.'
            }), 500
        
        current_app.logger.info("Processing summarization request")
        
        # Generate summary using the agent
        summary = agent.summarize_text(text or f"Document with ID: {doc_id}")
        
        current_app.logger.info("Summarization request processed successfully")
        
        return jsonify({
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in summarize endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Calculator endpoint
@api_bp.route('/calculate', methods=['POST'])
def calculate():
    """Perform medical calculations like BMI or dosage."""
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("No JSON data provided in calculate request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        calculation_type = data.get('type')
        parameters = data.get('parameters', {})
        
        if not calculation_type:
            current_app.logger.warning("No calculation type provided in calculate request")
            return jsonify({'error': 'No calculation type provided'}), 400
        
        current_app.logger.info(f"Processing calculation request: {calculation_type}")
        
        calculator = CalculatorTool()
        result = calculator.calculate(calculation_type, parameters)
        
        if 'error' in result:
            current_app.logger.error(f"Calculation error: {result['error']}")
            return jsonify({
                'error': 'Calculation failed',
                'message': result['error']
            }), 400
        
        current_app.logger.info("Calculation request processed successfully")
        
        return jsonify({
            'result': result,
            'calculation_type': calculation_type,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in calculate endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Document search endpoint
@api_bp.route('/search', methods=['POST'])
def search_documents():
    """Search for relevant documents using RAG."""
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("No JSON data provided in search request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        if not query:
            current_app.logger.warning("No query provided in search request")
            return jsonify({'error': 'No query provided'}), 400
        
        top_k = data.get('top_k', 4)
        
        if retriever is None:
            current_app.logger.error("Retriever not initialized")
            return jsonify({
                'error': 'Retriever not initialized',
                'message': 'The document retriever is not available. Please check the system logs.'
            }), 500
        
        current_app.logger.info(f"Processing search request: {query}")
        
        # Search for relevant documents
        results = retriever.search(query, top_k=top_k)
        
        current_app.logger.info(f"Search request processed successfully, found {len(results)} results")
        
        return jsonify({
            'results': results,
            'query': query,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Get system stats
@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        if retriever is None:
            current_app.logger.error("Retriever not initialized")
            return jsonify({
                'error': 'Retriever not initialized',
                'message': 'The document retriever is not available. Please check the system logs.'
            }), 500
        
        current_app.logger.info("Processing stats request")
        
        stats = retriever.get_stats()
        
        current_app.logger.info("Stats request processed successfully")
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500
# Website scraping endpoint
@api_bp.route('/scrape', methods=['POST'])
def scrape_website():
    """Scrape content from a website URL."""
    try:
        data = request.get_json()
        if not data:
            current_app.logger.warning("No JSON data provided in scrape request")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url')
        if not url:
            current_app.logger.warning("No URL provided in scrape request")
            return jsonify({'error': 'No URL provided'}), 400
        
        current_app.logger.info(f"Processing website scrape request: {url}")
        
        # Process the website scraping
        loader = DocumentLoader()
        result = loader.load_document_from_url(url)
        
        if result.get('status') == 'success':
            # Add website content to knowledge base
            content = result.get('content', '')
            source_name = result.get('url', url)
            
            add_result = loader.add_document_to_knowledge_base(content, source_name)
            
            if add_result.get('status') == 'success':
                current_app.logger.info("Website scraped and added to knowledge base successfully")
                return jsonify({
                    'message': 'Website scraped and added to knowledge base successfully',
                    'result': add_result,
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
            else:
                current_app.logger.error(f"Error adding website to knowledge base: {add_result.get('error')}")
                return jsonify({
                    'error': 'Failed to add website to knowledge base',
                    'message': add_result.get('error', 'Unknown error')
                }), 500
        else:
            current_app.logger.error(f"Error scraping website: {result.get('error')}")
            return jsonify({
                'error': 'Website scraping failed',
                'message': result.get('error', 'Unknown error')
            }), 500
        
    except Exception as e:
        current_app.logger.error(f"Error in scrape endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500