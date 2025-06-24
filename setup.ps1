# setup.ps1
# PowerShell script to set up the Python environment for the RAG chatbot

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Green
pip install -r requirements.txt


Write-Host "Setup complete!" -ForegroundColor Green
