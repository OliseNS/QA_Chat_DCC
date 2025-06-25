# RAG Chatbot for DCC Dialysis

A simple chatbot that answers questions using content from https://dccdialysis.com/ using RAG (Retrieval-Augmented Generation) and Hugging Face models.

## Project Structure

```
Rag_chat/
├── data/
│   ├── processed/ # Processed data (chunks, embeddings)
│   └── raw/       # Raw scraped content
├── src/
│   ├── main.py    # Main entry point
│   └── scraper.py # Web scraper module
├── plan.md        # Project plan
├── README.md      # This file
├── requirements.txt # Python dependencies
├── setup.ps1      # Setup script
└── todo.md        # Todo list
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- wkhtmltopdf (for PDF generation): [Download here](https://wkhtmltopdf.org/downloads.html)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Rag_chat.git
   cd Rag_chat
   ```

2. Run the setup script (Windows):
   ```
   .\setup.ps1
   ```

   This will:
   - Create a virtual environment
   - Install required packages
   - Remind you to install wkhtmltopdf

3. Activate the virtual environment:
   ```
   .\venv\Scripts\Activate
   ```

## Usage

### Scraping the Website

To scrape the DCC Dialysis website and save content as text and PDF files:

```
python scrape_dcc
```

This will:
1. Scrape all pages from https://dccdialysis.com/
2. Save the raw content as text files in `data/raw/`
4. Process and chunk the content for later use in `data/processed/`

### Generating Embeddings

To generate embeddings for the chunked content:

```
python src/generate_embeddings.py
```

This will:
1. Process all chunks in the `data/chunks/` directory
2. Generate embeddings using the SentenceTransformer model
3. Save individual embeddings in `data/embeddings/`
4. Create FAISS-compatible embeddings in `data/embeddings/faiss/`

## Using Hugging Face Models

The chatbot uses Google's Gemma-2B model by default. If you want to use this model, you'll need to:

1. Create a Hugging Face account if you don't have one: https://huggingface.co/join
2. Accept the model license at https://huggingface.co/google/gemma-2b
3. Create an access token at https://huggingface.co/settings/tokens
4. Copy the `.env.template` file to `.env` and add your token:
   ```
   cp .env.template .env
   ```
   Then edit the `.env` file to add your token:
   ```
   HUGGINGFACE_TOKEN=your_actual_token_here
   ```

If you don't provide a token, the system will automatically fall back to TinyLlama, which is a smaller, non-gated model.

### Running the RAG Chatbot

To run the chatbot from the command line:

```
python src/rag_chatbot.py
```

This will start a simple CLI interface where you can ask questions about DCC Dialysis.

### Running the Web UI

To run the chatbot with a web interface:

```
streamlit run app.py
```

This will start a Streamlit web server, and you can access the chatbot through your browser.

## License

This project is for educational purposes only. The content is gotten from scraping https://dccdialysis.com/ .
