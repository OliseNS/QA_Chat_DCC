# This script sets up and runs the RAG Chatbot for DCC Dialysis

# First we install dependencies
function Install-Dependencies {
    Write-Host "Checking and installing required packages..." -ForegroundColor Cyan
    
    # Check if virtual environment exists, create if not
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate
    
    # Install requirements
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}

# Function to check if we have all necessary data processed
function Check-Data {
    $chunksDir = ".\data\chunks"
    $embeddingsDir = ".\data\embeddings\faiss"
    
    # Check if data is already processed
    $chunksExist = Test-Path $chunksDir
    $embeddingsExist = Test-Path "$embeddingsDir\embeddings.npy"
    
    return @{
        ChunksExist = $chunksExist
        EmbeddingsExist = $embeddingsExist
    }
}

# Function to check if .env file exists and create it if needed
function Setup-Environment {
    $envTemplatePath = ".\.env.template"
    $envPath = ".\.env"
    
    if (-not (Test-Path $envPath) -and (Test-Path $envTemplatePath)) {
        Write-Host "No .env file found. Creating one from template..." -ForegroundColor Yellow
        
        # Read the template
        $envTemplate = Get-Content $envTemplatePath -Raw
        
        # Ask user if they want to add a Hugging Face token
        $addToken = Read-Host "Do you want to add a Hugging Face token for accessing Gemma-2B? (y/n)"
        
        if ($addToken -eq "y" -or $addToken -eq "Y") {
            $token = Read-Host "Enter your Hugging Face token (from https://huggingface.co/settings/tokens)"
            $envContent = $envTemplate -replace "your_token_here", $token
            
            # Save with token
            Set-Content -Path $envPath -Value $envContent
            Write-Host "Created .env file with your Hugging Face token." -ForegroundColor Green
        } else {
            # Save without token (will use fallback model)
            Set-Content -Path $envPath -Value $envTemplate
            Write-Host "Created .env file. You can edit it later to add your token." -ForegroundColor Yellow
            Write-Host "Without a token, the system will use a smaller fallback model." -ForegroundColor Yellow
        }
    } elseif (Test-Path $envPath) {
        Write-Host ".env file already exists." -ForegroundColor Green
    } else {
        Write-Host "No .env.template file found. Continuing without environment configuration." -ForegroundColor Yellow
    }
}

# Function to process data
function Process-Data {
    param (
        [bool]$runScraper = $false,
        [bool]$generateChunks = $false,
        [bool]$generateEmbeddings = $false
    )
    
    # Run scraper if needed
    if ($runScraper) {
        Write-Host "Running web scraper..." -ForegroundColor Cyan
        python .\src\scrape_dcc.py
    }
    
    # Generate chunks if needed
    if ($generateChunks) {
        Write-Host "Processing content into chunks..." -ForegroundColor Cyan
        python .\src\process_to_chunks.py
    }
    
    # Generate embeddings if needed
    if ($generateEmbeddings) {
        Write-Host "Generating embeddings..." -ForegroundColor Cyan
        python .\src\generate_embeddings.py
    }
}

# Function to run the chatbot
function Run-Chatbot {
    param (
        [string]$mode = "cli"  # cli or web
    )
    
    if ($mode -eq "cli") {
        Write-Host "Starting command-line chatbot..." -ForegroundColor Cyan
        python .\src\rag_chatbot.py
    }
    elseif ($mode -eq "web") {
        Write-Host "Starting web interface chatbot..." -ForegroundColor Cyan
        streamlit run app.py
    }
    else {
        Write-Host "Invalid mode specified. Use 'cli' or 'web'." -ForegroundColor Red
    }
}

# Main script execution
Write-Host "=== DCC Dialysis RAG Chatbot Setup ===" -ForegroundColor Magenta

# Install dependencies
Install-Dependencies

# Set up environment file (.env) for Hugging Face token
Setup-Environment

# Check data status
$dataStatus = Check-Data
$runScraper = $false
$generateChunks = $false
$generateEmbeddings = $false

# Determine what needs to be processed
if (-not $dataStatus.ChunksExist) {
    Write-Host "No chunks found. Need to run scraper and process content." -ForegroundColor Yellow
    $runScraper = $true
    $generateChunks = $true
    $generateEmbeddings = $true
}
elseif (-not $dataStatus.EmbeddingsExist) {
    Write-Host "Chunks found but no embeddings. Need to generate embeddings." -ForegroundColor Yellow
    $generateEmbeddings = $true
}
else {
    Write-Host "Data is already processed and ready to use." -ForegroundColor Green
}

# Process data if needed
if ($runScraper -or $generateChunks -or $generateEmbeddings) {
    Process-Data -runScraper $runScraper -generateChunks $generateChunks -generateEmbeddings $generateEmbeddings
}

# Ask user for interface preference
$validResponse = $false
while (-not $validResponse) {
    $interfaceChoice = Read-Host "Choose interface mode (cli/web)"
    if ($interfaceChoice -eq "cli" -or $interfaceChoice -eq "web") {
        $validResponse = $true
    }
    else {
        Write-Host "Invalid choice. Please enter 'cli' or 'web'." -ForegroundColor Red
    }
}

# Run the chatbot
Run-Chatbot -mode $interfaceChoice

# Keep the console window open if running CLI mode
if ($interfaceChoice -eq "cli") {
    Read-Host "Press Enter to exit"
}
