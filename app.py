#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import json
import os
import requests
import time
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Dialysis Care Center Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set clean styling
st.markdown("""
<style>
    /* Brand colors */
    :root {
        --dcc-red: rgb(171, 35, 40);
        --light-gray: #f8f9fa;
    }
    
    /* Main app background */
    .stApp {
        background: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: white !important;
    }
    
    /* Logo container margin */
    .logo-container {
        margin: 1.5rem 0 2rem 0 !important;
        text-align: center;
    }
    
    .logo-container img {
        margin: 1rem 0;
        max-width: 100%;
        height: auto;
    }
    
    /* All buttons - unified styling with higher specificity */
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        background: white !important;
        color: black !important;
        border: 2px solid black !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        width: 100% !important;
        text-align: left !important;
        transition: all 0.2s !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        background: #f8f9fa !important;
        color: black !important;
        border: 2px solid black !important;
    }
    
    /* Force button text color */
    .stButton > button span,
    .stButton > button p,
    .stButton > button div {
        color: black !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    
    /* Chat input - use default Streamlit styling */
    
    /* Headers */
    h1, h2, h3 {
        color: var(--dcc-red);
    }
    
    /* Text visibility */
    p, div, span, label {
        color: black !important;
    }
    
    /* Tooltips - better visibility */
    .stTooltip,
    [data-testid="stTooltipHoverTarget"],
    [data-baseweb="tooltip"] {
        color: black !important;
        background: white !important;
        border: 2px solid black !important;
        border-radius: 4px !important;
        padding: 0.5rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        font-weight: 500 !important;
        z-index: 9999 !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 8px;
        background: white;
    }
    
    .streamlit-expanderHeader {
        background: var(--light-gray);
        color: var(--dcc-red);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# UI Settings
ui_settings = config["ui_settings"]
app_info = config["app_info"]
sample_questions = config["sample_questions"]

# Function to load the improved retriever
@st.cache_resource
def load_retriever():
    try:
        from rag_retriever import RAGRetriever
        retriever = RAGRetriever()
        return retriever
    except Exception as e:
        st.error(f"Error loading RAG retriever: {str(e)}")
        return None

# Function to search for relevant documents using improved retriever
def search_documents_improved(query, retriever, top_k=3):
    """Search for relevant documents using the RAG retriever."""
    if retriever is None:
        return []
    
    try:
        results = retriever.search(query, top_k=top_k)
        
        # Convert to expected format for compatibility
        formatted_results = []
        for result in results:
            formatted_result = {
                "text": result['content'],  # Use 'content' field from retriever
                "source": result['category'],  # Use 'category' as source
                "distance": 1.0 - result['similarity'],  # Convert similarity to distance
                "similarity": result['similarity'],
                "rerank_score": result['similarity']
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    except Exception as e:
        st.error(f"Error in improved search: {e}")
        return []

# Function to get embedding for a query
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def get_embedding(query):
    model = load_embedding_model()
    if model is None:
        st.error("Embedding model failed to load. Please check your installation.")
        return None
    
    try:
        return model.encode(query)
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Function to call OpenRouter API with streaming support
def call_openrouter_api(messages, temperature=0.7, stream=False):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return """
🔑 **API Key Required**

To use this assistant, you need to set up your OpenRouter API key:

1. **Get an API key**: Visit https://openrouter.ai and sign up for an account
2. **Copy your API key** from the dashboard
3. **Update the .env file** in the project directory:
   - Open the `.env` file
   - Replace `your_openrouter_api_key_here` with your actual API key
   - Save the file and refresh this page

**Note**: OpenRouter provides free credits for testing, and this app uses the free Gemma-3 model.
        """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://dialysisassistant.com",
        "X-Title": "DCC Dialysis Assistant"
    }
    
    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": stream
    }
    
    try:
        if stream:
            # For streaming responses
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()
            return response
        else:
            # For non-streaming responses
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error calling OpenRouter API: {e}")
        return f"Error generating response: {str(e)}"

# Function to parse streaming response
def parse_streaming_response(response):
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == '[DONE]':
                        break
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        st.error(f"Streaming error: {e}")
        yield ""

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "context_data" not in st.session_state:
    st.session_state.context_data = []

# Load retriever
try:
    retriever = load_retriever()
    if retriever is None:
        st.error("⚠️ **Knowledge Base Error**: The retriever could not be loaded. Please ensure all data files are present in the data/embeddings/faiss directory.")
except Exception as e:
    st.error(f"**Initialization Error**: {e}")
    retriever = None

# Sidebar
with st.sidebar:
    # Show the logo with modern styling
    st.markdown("""
    <div class="logo-container">
        <img src="https://eadn-wc01-6859330.nxedge.io/wp-content/uploads/2021/07/DCC-Logo-Clause_Rebranded_888404-600x227.png" alt="Dialysis Care Center Logo">
    </div>
    """, unsafe_allow_html=True)
    
    # App title and description with improved contrast
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; 
               background: linear-gradient(135deg, #ffffff, #f8f9fa); 
               border-radius: 12px; 
               border: 2px solid rgba(171, 35, 40, 0.15);">
        <h2 style="font-size: 1.6rem; 
                  font-weight: 700; 
                  color: rgb(171, 35, 40); 
                  margin-bottom: 0.8rem;
                  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
            {app_info['title']}
        </h2>
        <p style="color: #495057; 
                 font-size: 1rem; 
                 font-weight: 500; 
                 line-height: 1.5; 
                 margin: 0;">
            {app_info['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Questions section with improved styling
    st.markdown("""
    <h3 style="font-size: 1.3rem; 
              font-weight: 700; 
              margin-bottom: 1.5rem; 
              color: rgb(171, 35, 40);
              text-align: center;
              text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
        💡 Sample Questions
    </h3>
    """, unsafe_allow_html=True)
    
    # Sample questions with improved buttons
    for i, question in enumerate(sample_questions):
        if st.button(
            question, 
            key=f"sample_btn_{i}",
            use_container_width=True
        ):
            # Add the question to messages and chat history immediately
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Set flag to process the response
            st.session_state.process_sample_question = True
            st.session_state.sample_question_text = question
            
            # Rerun to show the message and process response
            st.rerun()
    
    # Divider with gradient effect
    st.markdown("""
    <hr style="height: 3px; 
              border: none; 
              background: linear-gradient(to right, transparent, rgba(171, 35, 40, 0.4), transparent); 
              margin: 2.5rem 0; 
              border-radius: 2px;">
    """, unsafe_allow_html=True)
    
    # Info box with better contrast
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffffff, #f8f9fa); 
                border-radius: 16px; 
                padding: 1.8rem; 
                margin-top: 1.5rem; 
                border: 3px solid rgba(171, 35, 40, 0.25);
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);">
        <h4 style="font-size: 1.2rem; 
                  font-weight: 700; 
                  color: rgb(171, 35, 40); 
                  margin-bottom: 1rem;
                  text-align: center;
                  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
            📋 About This Assistant
        </h4>
        <p style="font-size: 1rem; 
                 color: #495057; 
                 margin-bottom: 0; 
                 line-height: 1.6;
                 font-weight: 500;
                 text-align: center;">
            This AI assistant helps answer questions about dialysis care and services offered by Dialysis Care Center. The assistant uses real information from DCC resources.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer info with modern styling and better visibility
    st.markdown(f"""
    <div style="text-align: center; margin-top: 3rem;">
        <div style="background: linear-gradient(135deg, #ffffff, #f8f9fa); 
                    border-radius: 12px; 
                    padding: 1.5rem; 
                    box-shadow: 0 4px 16px rgba(171, 35, 40, 0.15);
                    border: 3px solid rgba(171, 35, 40, 0.2);">
            <p style="font-size: 1rem; 
                     font-weight: 700; 
                     color: rgb(171, 35, 40); 
                     margin-bottom: 0.5rem;
                     text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
                🚀 Version: {app_info['version']}
            </p>
            <p style="font-size: 0.95rem; 
                     color: #495057; 
                     margin-bottom: 0; 
                     font-weight: 600;">
                👨‍💻 Developed by: {app_info['developer']['name']}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content area with improved visibility
st.markdown("""
<div style="background: white; 
           border-radius: 20px; 
           box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12); 
           padding: 2.5rem; 
           margin-bottom: 2rem; 
           border: 3px solid rgba(171, 35, 40, 0.2);
           text-align: center;">
    <h1 style="font-size: 2.5rem; 
              font-weight: 800; 
              margin-bottom: 1rem;
              color: rgb(171, 35, 40);">
        💬 Dialysis Care Center Assistant
    </h1>
    <p style="color: black; 
             margin-top: 1rem; 
             font-size: 1.3rem; 
             font-weight: 500;">
        Your AI-powered guide to dialysis care and treatment options
    </p>
    <div style="margin-top: 1.5rem; 
               padding: 1rem; 
               background: rgba(171, 35, 40, 0.05); 
               border-radius: 12px; 
               border: 2px solid rgba(171, 35, 40, 0.1);">
        <p style="color: rgb(171, 35, 40); 
                 font-weight: 600; 
                 margin: 0;
                 font-size: 1.1rem;">
            🎯 Ask me anything about dialysis treatments, procedures, and care options!
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Add clear chat button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.context_data = []
        st.rerun()# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show context information after assistant messages
        if (message["role"] == "assistant" and 
            st.session_state.context_data and 
            i == len(st.session_state.messages) - 1):  # Only for the latest assistant message
            
            st.markdown("---")
            with st.expander("📚 View Source Information", expanded=False):
                for j, context in enumerate(st.session_state.context_data):
                    st.markdown(f"""
                    <div style="background: white; border-radius: 10px; padding: 1.2rem; 
                                margin-bottom: 1.2rem; border: 2px solid rgba(171, 35, 40, 0.2);">
                        <h4 style="margin-top: 0; color: rgb(171, 35, 40); font-size: 1.1rem; font-weight: 600;">{context['source']}</h4>
                        <span style="display: inline-block; background: rgba(171, 35, 40, 0.15); 
                                    color: rgb(171, 35, 40); border-radius: 999px; padding: 0.3rem 0.8rem;
                                    font-size: 0.85rem; font-weight: 600; margin-bottom: 0.6rem;">
                            Relevance: {1.0 - context['distance']:.2f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use a cleaner text area without tooltips
                    st.text_area("Source Content", 
                                 context["text"], 
                                 height=150, 
                                 key=f"ctx_{i}_{j}")
                    
                    if j < len(st.session_state.context_data) - 1:
                        st.markdown("<hr style='margin: 1rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

# Enhanced system prompt with better prompt engineering
ENHANCED_SYSTEM_PROMPT = """You are an expert dialysis care specialist and patient advocate at Dialysis Care Center (DCC). Your role is to provide comprehensive, accurate, and compassionate guidance about dialysis treatments and care options.

## Core Identity & Expertise:
- Expert knowledge in all dialysis modalities (hemodialysis, peritoneal dialysis, home treatments)
- Deep understanding of patient care, treatment planning, and quality of life considerations
- Empathetic advocate who prioritizes patient education and empowerment
- Professional healthcare communicator with 15+ years of experience

## Knowledge Base Context:
{context_str}

## Response Framework (Use this structure):

### 1. IMMEDIATE RESPONSE
- Directly address the user's question with key information upfront
- Use clear, jargon-free language that patients can understand

### 2. DETAILED EXPLANATION
- Provide comprehensive details organized with headers and bullet points
- Include relevant medical context when appropriate
- Explain "why" behind recommendations, not just "what"

### 3. PRACTICAL GUIDANCE
- Offer actionable next steps or considerations
- Include lifestyle impact information when relevant
- Mention when to consult healthcare providers

### 4. SUPPORTIVE CLOSURE
- Acknowledge the emotional aspect of dialysis decisions
- Reinforce that help and support are available
- End with encouragement or reassurance when appropriate

## Communication Guidelines:
✅ DO:
- Use warm, professional, and empathetic tone
- Explain medical terms in simple language
- Provide specific, actionable information
- Acknowledge uncertainty when information is incomplete
- Use "you" to make responses personal and direct
- Include emotional support and encouragement
- Structure responses for easy scanning (headers, bullets)

❌ DON'T:
- Create fake contact information, phone numbers, or specific scheduling details
- Make definitive medical diagnoses or treatment decisions
- Use overly technical medical jargon without explanation
- Provide outdated or generalized information
- Rush through explanations - take time to be thorough
- Ignore the emotional impact of dialysis on patients and families

## Response Quality Standards:
- Accuracy: Base all information on provided knowledge base
- Clarity: Write at an 8th-grade reading level
- Completeness: Address all aspects of the user's question
- Compassion: Acknowledge the challenges patients face
- Practicality: Include actionable information when possible

Remember: You're not just providing information - you're supporting someone through one of the most significant healthcare decisions they may face. Every response should reflect expertise, empathy, and genuine care for the patient's wellbeing."""

# Function to create enhanced system message
def create_enhanced_system_message(context_str):
    return ENHANCED_SYSTEM_PROMPT.format(context_str=context_str)

# Process sample question if flag is set
if "process_sample_question" in st.session_state and st.session_state.process_sample_question:
    # Get the sample question
    user_input = st.session_state.sample_question_text
    
    # Reset the flag
    st.session_state.process_sample_question = False
    del st.session_state.sample_question_text
    
    # Show a spinner while processing
    with st.spinner("Thinking..."):
        # Check if retriever is loaded
        if retriever is None:
            response = """
🔧 **System Error**

The knowledge base is not accessible. This usually means:

1. **Missing data files**: The semantic embeddings may not be generated yet
2. **Installation issue**: Required Python packages may not be installed  
3. **Data corruption**: The embedding files may be corrupted

**To fix this**:
- Run `python utils/generate_embeddings.py` to rebuild the knowledge base
- Make sure all required packages are installed: `pip install -r requirements.txt`
- Check that the `data/embeddings/faiss/` directory contains `metadata.json` and `embeddings.npy`
            """
        else:
            # Search for relevant context using improved retriever
            context_data = search_documents_improved(user_input, retriever, top_k=3)
            st.session_state.context_data = context_data
            
            # Prepare context for the LLM
            context_texts = [item["text"] for item in context_data]
            context_str = "\n\n".join(context_texts)
            
            # Prepare messages for the LLM - SAMPLE QUESTION PROCESSING
            messages = [
                {"role": "system", "content": create_enhanced_system_message(context_str)},
            ]
            
            # Add chat history (limited to last 4 exchanges to save tokens)
            for msg in st.session_state.chat_history[-8:]:
                messages.append(msg)
            
            # Get streaming response from OpenRouter
            response_stream = call_openrouter_api(messages, temperature=0.7, stream=True)
            
            # Display assistant message with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in parse_streaming_response(response_stream):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # Remove cursor and show final response
                message_placeholder.markdown(full_response)
                response = full_response
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Input for new message
user_input = st.chat_input("Ask me about dialysis treatments and care options...")

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Show a spinner while processing
    with st.spinner("Thinking..."):
        # Check if retriever is loaded
        if retriever is None:
            response = """
🔧 **System Error**

The knowledge base is not accessible. This usually means:

1. **Missing data files**: The semantic embeddings may not be generated yet
2. **Installation issue**: Required Python packages may not be installed  
3. **Data corruption**: The embedding files may be corrupted

**To fix this**:
- Run `python utils/generate_embeddings.py` to rebuild the knowledge base
- Make sure all required packages are installed: `pip install -r requirements.txt`
- Check that the `data/embeddings/faiss/` directory contains `metadata.json` and `embeddings.npy`
            """
        else:
            # Search for relevant context using improved retriever
            context_data = search_documents_improved(user_input, retriever, top_k=3)
            st.session_state.context_data = context_data
            
            # Prepare context for the LLM
            context_texts = [item["text"] for item in context_data]
            context_str = "\n\n".join(context_texts)
            
            # Prepare messages for the LLM - USER INPUT PROCESSING
            messages = [
                {"role": "system", "content": create_enhanced_system_message(context_str)},
            ]
            
            # Add chat history (limited to last 4 exchanges to save tokens)
            for msg in st.session_state.chat_history[-8:]:
                messages.append(msg)
            
            # Try streaming response first, fall back to regular if it fails
            try:
                # Get streaming response from OpenRouter
                response_stream = call_openrouter_api(messages, temperature=0.7, stream=True)
                
                # Display assistant message with streaming
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in parse_streaming_response(response_stream):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "\\")
                        time.sleep(0.01)  # Small delay for smoother streaming effect
                    
                    # Remove cursor and show final response
                    message_placeholder.markdown(full_response)
                    response = full_response
            except Exception as e:
                # Fall back to regular API call if streaming fails
                st.warning("Streaming failed, using regular response...")
                response = call_openrouter_api(messages, temperature=0.7, stream=False)
                with st.chat_message("assistant"):
                    st.write(response)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Enhanced footer with improved visibility
st.markdown("""
<div style="margin-top: 3.5rem;">
    <div style="background: white; 
               border-radius: 12px; padding: 1.5rem; text-align: center; 
               box-shadow: 0 5px 15px rgba(171, 35, 40, 0.08);
               border: 3px solid rgba(171, 35, 40, 0.2);">
        <p style="color: rgb(171, 35, 40); font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem;">
            © 2025 Dialysis Care Center Assistant
        </p>
        <p style="color: #007bff; font-size: 0.9rem; margin-bottom: 0; font-weight: 500;">
            Powered by Google Gemma 3 - Helping patients understand dialysis care options
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
