# 💬 Dialysis Care Center Assistant

An AI-powered chatbot that helps patients understand dialysis care and treatment options using information from Dialysis Care Center (DCC). The application uses Retrieval-Augmented Generation (RAG) to provide accurate, contextual responses based on real DCC documentation.

## 🚀 Features

- **Intelligent Q&A**: Ask questions about dialysis treatments, procedures, and care options
- **Real-time Search**: Semantic search through DCC knowledge base using sentence transformers
- **Context Visualization**: View source documents and relevance scores for transparency
- **Sample Questions**: Pre-built questions to get started quickly
- **Modern UI**: Clean, responsive interface with DCC branding

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Sentence Transformers, OpenRouter API (Gemma-3)
- **Search**: FAISS vector database with semantic embeddings
- **Backend**: Python 3.10+
- **Data Processing**: NumPy, Scikit-learn

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- OpenRouter API key (get one free at [openrouter.ai](https://openrouter.ai))

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd QA_Chat
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Open the `.env` file
   - Replace `your_openrouter_api_key_here` with your actual OpenRouter API key
   ```env
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

4. **Generate embeddings** (if not already present)
   ```bash
   python utils/generate_embeddings.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Alternative Setup (Windows)

You can also use the provided setup script:
```cmd
setup.bat
```

## 🔧 Configuration

### API Key Setup

1. Visit [openrouter.ai](https://openrouter.ai) and create an account
2. Navigate to your dashboard and copy your API key
3. Update the `.env` file with your key
4. Restart the application

### Data Structure

The application expects this directory structure:
```
data/
├── embeddings/
│   └── faiss/
│       ├── metadata.json
│       └── embeddings.npy
├── chunks/
│   └── [category folders with text chunks]
└── raw/
    └── [raw text files]
```

## 📖 Usage

### Basic Usage

1. Open the application in your browser (usually `http://localhost:8501`)
2. Use the test retrieval button to verify the system is working
3. Click sample questions or type your own questions about dialysis care
4. View source information in the expandable section below responses

### Sample Questions

- "What types of dialysis treatment does DCC offer?"
- "How do I prepare for my first dialysis session?"
- "What are the benefits of home hemodialysis?"
- "Tell me about DCC's mission and values"
- "How can I contact DCC?"

### Testing Without API Key

You can test the retrieval system without an API key by:
1. Clicking the "🧪 Test Retrieval" button
2. Using sample questions (they will show system messages instead of AI responses)

## 🔍 Troubleshooting

### Common Issues

**1. "Knowledge base not accessible" error**
- Run `python utils/generate_embeddings.py` to rebuild embeddings
- Check that `data/embeddings/faiss/` contains `metadata.json` and `embeddings.npy`

**2. "API key not configured" message**
- Update the `.env` file with your OpenRouter API key
- Ensure the key is not wrapped in quotes
- Restart the application

**3. Import errors**
- Install all requirements: `pip install -r requirements.txt`
- Check Python version (3.10+ required)

**4. Retrieval returns no results**
- Verify embeddings are generated: `python rag_retriever.py`
- Check that chunk files exist in `data/chunks/`

### Getting Help

If you encounter issues:
1. Check the terminal/console for error messages
2. Verify all dependencies are installed
3. Ensure data files are present and accessible
4. Test the retriever independently with `python rag_retriever.py`

## 📁 Project Structure

```
QA_Chat/
├── app.py                 # Main Streamlit application
├── rag_retriever.py       # RAG retrieval system
├── config.json           # Application configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── setup.bat            # Windows setup script
├── utils/               # Utility scripts
│   ├── generate_embeddings.py
│   ├── process_to_chunks.py
│   └── scrape_dcc.py
└── data/               # Data directory
    ├── embeddings/     # Generated embeddings
    ├── chunks/         # Text chunks
    └── raw/           # Raw source files
```

## 🔄 Development

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

### Customization

- Modify `config.json` for UI settings and sample questions
- Update styling in `app.py` for visual customizations
- Adjust retrieval parameters in `rag_retriever.py`

## 📋 Version Info

- **Version**: 1.0.0
- **Developer**: Olisemeka Nmarkwe
- **AI Model**: Google Gemma-3 (via OpenRouter)
- **Embedding Model**: all-MiniLM-L6-v2

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for Dialysis Care Center to help patients access information about dialysis care and treatment options.

---

For technical support or questions, please contact the development team.
- Responsive design

## Technical Details

- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Search**: Cosine similarity with scikit-learn
- **LLM**: Google Gemma 3 27B
- **Frontend**: Streamlit
- **Data**: Chunked text and embeddings from DCC website content

## Installation

1. Clone this repository
2. Run the setup script:

```bash
# Windows
setup.bat

# Or manually install dependencies
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenRouter API key:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)
3. Ask questions about dialysis care and treatments

## Project Structure

```
├── config.json                # Application configuration
├── requirements.txt           # Python dependencies
├── app.py                     # Main Streamlit application
├── rag_retriever.py          # RAG retrieval system
├── setup.bat                 # Windows setup script
├── .env                      # Environment variables (API keys)
├── utils/                    # Utility scripts
│   ├── generate_embeddings.py # Generate embeddings from text
│   ├── process_to_chunks.py   # Process text into chunks
│   └── scrape_dcc.py          # Website scraping utility
└── data/
    ├── raw/                   # Raw text data
    ├── chunks/                # Chunked text data
    └── embeddings/
        └── faiss/             # Embeddings and metadata
            ├── embeddings.npy # Precomputed embeddings
            └── metadata.json  # Chunk metadata
```

## OpenRouter API Configuration

This application uses an API key that has been configured in the backend. No additional configuration is needed for the API key.

## Development

- Modify `config.json` to customize the application settings
- Add new content by running the scraping, chunking, and embedding generation scripts

## License

This project is created for educational purposes.

## Author

Made by [Olisemeka Nmarkwe](https://olisemeka.dev)
