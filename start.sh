#!/bin/bash

# Healthcare AI Assistant Startup Script

echo "🏥 Starting Healthcare AI Assistant..."

# Check if we're in the right directory
if [ ! -f "run_backend.py" ]; then
    echo "❌ Error: run_backend.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    # Activate virtual environment
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your OpenRouter API key"
fi

# Start the backend server
echo "🚀 Starting backend server..."
python run_backend.py