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

## License

This project is for educational purposes only. The content jis gotten from scraping https://dccdialysis.com/ .
