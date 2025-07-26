# 🏥 Healthcare AI Assistant

A full-stack Healthcare AI Assistant using Retrieval-Augmented Generation (RAG) and agentic reasoning to provide accurate medical information and assistance.

## 🚀 Features

- **Intelligent Q&A**: Ask questions about medical policies, treatments, and procedures
- **Document Processing**: Load and process PDF, HTML, and Markdown medical documents
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Medical Calculations**: Built-in calculator for BMI, dosage, and other medical calculations
- **Document Summarization**: Summarize medical documents for quick understanding
- **RESTful API**: Clean JSON API for integration with other applications
- **Docker Support**: Containerized deployment for easy scaling

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask
- **AI/ML**: Sentence Transformers, OpenRouter API (Gemma-3)
- **Search**: FAISS vector database with semantic embeddings
- **Data Processing**: NumPy, Scikit-learn, PyPDF2, BeautifulSoup4
- **Containerization**: Docker, Docker Compose

### Frontend
- **Framework**: Vanilla JavaScript (Single HTML file)
- **UI Library**: Custom CSS with Material Icons
- **Styling**: Modern gradient design with responsive layout

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional but recommended)
- OpenRouter API key (get one free at [openrouter.ai](https://openrouter.ai))

### Quick Setup with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd healthcare-ai-assistant
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Backend API: http://localhost:5000
   - Frontend UI: http://localhost:5000 (served from the backend)

### Manual Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd healthcare-ai-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

5. **Run the application**
   ```bash
   python run_backend.py
   ```

## 📁 Project Structure

```
healthcare-ai-assistant/
├── backend/                 # Backend Flask application
│   ├── agents/             # AI agents with RAG and tools
│   ├── routes/             # API routes and endpoints
│   ├── utils/              # Utility functions
│   ├── app.py             # Flask application factory
│   └── config.py          # Configuration settings
├── data/                  # Data directory
│   ├── embeddings/        # Generated embeddings
│   ├── chunks/            # Text chunks
│   └── raw/               # Raw source files
├── docs/                  # Medical documents for processing
├── examples/              # Example API request payloads
├── frontend/              # Frontend static files
│   └── public/            # Publicly served files
│       └── index.html     # Main frontend application
├── uploads/               # Uploaded documents
├── utils/                 # Data processing utilities
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
├── Dockerfile            # Docker configuration for backend
├── Dockerfile.frontend   # Docker configuration for frontend (coming soon)
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
├── run_backend.py        # Backend entry point
└── README.md             # This file
```

## 🔄 API Endpoints

### Health Check
```
GET /api/health
```

### Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "What are the symptoms of diabetes?",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help you today?"}
  ]
}
```

### Document Search
```
POST /api/search
Content-Type: application/json

{
  "query": "treatment options for kidney disease",
  "top_k": 5
}
```

### Medical Calculator
```
POST /api/calculate
Content-Type: application/json

{
  "type": "bmi",
  "parameters": {
    "weight_kg": 70,
    "height_m": 1.75
  }
}
```

### Document Summarization
```
POST /api/summarize
Content-Type: application/json

{
  "text": "Long medical document text here..."
}
```

### Document Upload
```
POST /api/upload
Content-Type: multipart/form-data

file: [PDF/HTML/Markdown document]
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# OpenRouter API Key (required)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model settings
LLM_MODEL=google/gemma-3-27b-it:free

# Data directories
DOCS_DIR=docs
DATA_DIR=data
EMBEDDINGS_DIR=data/embeddings/faiss

# Flask settings
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=False
HOST=127.0.0.1
PORT=5000

# Logging settings
LOG_LEVEL=INFO

# Upload settings
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# RAG settings
CHUNK_SIZE=200
OVERLAP_SIZE=50
TOP_K_RESULTS=4
SIMILARITY_THRESHOLD=0.15
```

## 📖 Usage

### Using the Frontend UI

The application features a modern web interface that allows you to:

1. **Upload Medical Documents**:
   - Click "Choose PDF File" to select a PDF document from your computer
   - After selecting a file, click "Process PDF" to add it to the knowledge base
   - You can also scrape medical websites by entering a URL and clicking "Scrape Website"

2. **Chat with the AI Assistant**:
   - Type your medical questions in the chat input at the bottom
   - Press Enter or click "Send" to submit your question
   - The assistant will provide answers based on the medical documents in the knowledge base

3. **View Responses**:
   - Your messages appear on the right with a purple background
   - Assistant responses appear on the left with a white background
   - Timestamps help you track the conversation history

### Adding Medical Documents

1. Place PDF, HTML, or Markdown documents in the `docs/` directory
2. The system will automatically process these documents when the application starts
3. Documents are chunked and converted to embeddings for RAG

### Making API Requests

See the `examples/` directory for sample request payloads:

- `examples/chat_request.json` - Chat with the assistant
- `examples/search_request.json` - Search for relevant documents
- `examples/calculate_request.json` - Perform medical calculations
- `examples/summarize_request.json` - Summarize medical text

### Example API Calls

```bash
# Health check
curl http://localhost:5000/api/health

# Chat with the assistant
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d @examples/chat_request.json

# Search for documents
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d @examples/search_request.json

# Calculate BMI
curl -X POST http://localhost:5000/api/calculate \
  -H "Content-Type: application/json" \
  -d @examples/calculate_request.json
```

## 🔍 Troubleshooting

### Common Issues

**1. "API key not configured" error**
- Ensure you have set your OpenRouter API key in the `.env` file
- Restart the application after updating environment variables

**2. "Knowledge base not accessible" error**
- Verify that the embeddings have been generated in `data/embeddings/faiss/`
- Run the data processing utilities if needed

**3. Docker build failures**
- Ensure Docker is running and you have sufficient permissions
- Check Docker logs for specific error messages

### Getting Help

If you encounter issues:
1. Check the application logs for error messages
2. Verify all dependencies are installed
3. Ensure environment variables are properly configured
4. Test individual components with example requests

## 📋 Development

### Regenerating Embeddings

If you update the source documents:
```bash
python utils/generate_embeddings.py
```

### Testing Components

Test the retriever:
```bash
python rag_retriever.py
```

### Adding New Features

1. Add new routes in `backend/routes/api.py`
2. Implement new functionality in `backend/agents/` or `backend/utils/`
3. Update the API documentation in this README
4. Add example requests in the `examples/` directory

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for educational and healthcare assistance purposes.

## 📞 Support

For technical support or questions, please contact the development team.

---

Made with ❤️ for better healthcare accessibility
