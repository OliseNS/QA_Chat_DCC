@echo off
title Healthcare AI Assistant

echo 🏥 Starting Healthcare AI Assistant...

:: Check if we're in the right directory
if not exist "run_backend.py" (
    echo ❌ Error: run_backend.py not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv venv
    
    :: Activate virtual environment
    call venv\Scripts\activate
    
    :: Install dependencies
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
) else (
    :: Activate virtual environment
    call venv\Scripts\activate
)

:: Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Creating from .env.example...
    copy .env.example .env
    echo 📝 Please edit .env and add your OpenRouter API key
)

:: Start the backend server
echo 🚀 Starting backend server...
python run_backend.py

pause