# Simple RAG Chatbot for DCC Dialysis Website - TODO

## Project Goal
Create a simple chatbot that answers questions using content from https://dccdialysis.com/ using RAG (Retrieval-Augmented Generation) and Hugging Face models.

## Basic Setup
- [x] Set up Python virtual environment
- [x] Install required packages (requests, beautifulsoup4, sentence-transformers, chromadb, transformers, torch, streamlit)
- [x] Create basic project structure

## Data Collection
- [x] Create simple web scraper for dccdialysis.com
  - [x] Extract content from main pages
  - [x] Clean HTML and extract text
  - [x] Chunk text into manageable segments
- [x] Save processed data for reuse
- [x] Save PDF files for scraped pages

## Vector Database
- [ ] Set up ChromaDB for vector storage
- [ ] Select embedding model (sentence-transformers/all-MiniLM-L6-v2)
- [ ] Generate embeddings for website content
- [ ] Implement basic similarity search

## Language Model
- [ ] Choose appropriate Hugging Face model (mistralai/Mistral-7B-Instruct-v0.1)
- [ ] Set up model with optimizations for local use
- [ ] Create prompt template for question answering
- [ ] Implement response generation function

## RAG Pipeline
- [ ] Create function to retrieve relevant context based on query
- [ ] Integrate retrieval with language model
- [ ] Format responses with appropriate disclaimers
- [ ] Implement basic error handling

## Simple UI
- [ ] Create Streamlit interface with:
  - [ ] Input field for user questions
  - [ ] Display area for chatbot responses
  - [ ] Optional: display sources of information
- [ ] Make UI responsive and user-friendly

## Testing
- [ ] Test with common dialysis-related questions
- [ ] Check response quality and relevance
- [ ] Verify source attribution
- [ ] Optimize for better performance if needed

## Documentation
- [ ] Document setup instructions
- [ ] Create simple user guide
- [ ] Add comments to code for maintainability

This simplified todo list focuses on the essential components needed to create a functional RAG chatbot with a clean interface while keeping the implementation straightforward.
