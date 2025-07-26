import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import json

# Add the parent directory to the path to import rag_retriever
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes import api_bp
from backend.config import Config

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Set up logging
    if not app.debug:
        logging.basicConfig(
            level=getattr(logging, app.config['LOG_LEVEL'], logging.INFO),
            format='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
    else:
        # More detailed logging for debug mode
        logging.basicConfig(
            level=getattr(logging, app.config['LOG_LEVEL'], logging.DEBUG),
            format='%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s'
        )
    
    app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL'], logging.INFO))
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            'message': 'Healthcare AI Assistant API',
            'version': '1.0.0',
            'docs': '/api/docs'
        }), 200
    
    # Log all requests
    @app.before_request
    def log_request_info():
        app.logger.info(f"Request: {request.method} {request.url}")
        app.logger.info(f"Headers: {dict(request.headers)}")
        if request.is_json:
            app.logger.info(f"Body: {request.get_json()}")
    
    # Log all responses
    @app.after_request
    def log_response_info(response):
        app.logger.info(f"Response: {response.status}")
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )