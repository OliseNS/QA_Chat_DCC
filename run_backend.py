#!/usr/bin/env python3
"""
Main entry point for the Healthcare AI Assistant Flask backend.
"""

import os
import sys

# Add the current directory to the path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import create_app

if __name__ == '__main__':
    app = create_app()
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Healthcare AI Assistant backend...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Press CTRL+C to stop the server")
    
    app.run(host=host, port=port, debug=debug)