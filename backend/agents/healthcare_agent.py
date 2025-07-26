import os
import sys
import json
import requests
from typing import List, Dict, Any
import logging

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import Config
from backend.utils.calculator import CalculatorTool

class HealthcareAgent:
    """Healthcare AI Assistant agent with RAG, calculator, and summarization capabilities."""
    
    def __init__(self, retriever):
        """Initialize the healthcare agent."""
        self.retriever = retriever
        self.calculator = CalculatorTool()
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for the LLM
        self.system_prompt = """You are a knowledgeable healthcare assistant specializing in medical policies, treatments, and patient care.
        
        RULES:
        - Answer using ONLY the knowledge base provided - this contains real medical information
        - Be thorough and detailed when information IS available in the sources
        - If sources lack specific details: "For more details about [topic], please consult with a healthcare professional"
        - Use clear formatting: bullet points, **bold** for emphasis
        - Explain medical terms simply
        - For calculations, use the calculator tool when appropriate
        
        KNOWLEDGE BASE:
        {context_str}
        
        Answer comprehensively using the above sources."""
    
    def process_query(self, query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's question
            history: Conversation history
            
        Returns:
            Dictionary with response and context information
        """
        try:
            # Search for relevant context using RAG
            context_data = self.retriever.search(
                query, 
                top_k=self.config.TOP_K_RESULTS, 
                similarity_threshold=self.config.SIMILARITY_THRESHOLD
            )
            
            # Prepare context for the LLM
            context_texts = [item["content"] for item in context_data]
            context_str = "\n\n".join(context_texts)
            
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": self.system_prompt.format(context_str=context_str)},
            ]
            
            # Add conversation history (limited to last 4 exchanges to save tokens)
            if history:
                for msg in history[-8:]:  # Last 4 exchanges (8 messages)
                    messages.append(msg)
            
            # Add current user query
            messages.append({"role": "user", "content": query})
            
            # Get response from LLM
            llm_response = self._call_llm_api(messages)
            
            return {
                "response": llm_response,
                "context": context_data,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing your query: {str(e)}",
                "context": [],
                "query": query
            }
    
    def summarize_text(self, text: str) -> str:
        """
        Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary of the text
        """
        try:
            # Create a summarization prompt
            summary_prompt = f"""Please provide a concise summary of the following medical text:
            
            {text}
            
            Summary:"""
            
            messages = [
                {"role": "system", "content": "You are a medical text summarization assistant. Provide concise, accurate summaries of medical information."},
                {"role": "user", "content": summary_prompt}
            ]
            
            # Get summary from LLM
            summary = self._call_llm_api(messages)
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return f"Error generating summary: {str(e)}"
    
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the OpenRouter API with the provided messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response from the LLM
        """
        if not self.config.OPENROUTER_API_KEY or self.config.OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            return "API key not configured. Please set your OpenRouter API key in the environment variables."
        
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://healthcare-assistant.com",
            "X-Title": "Healthcare AI Assistant"
        }
        
        payload = {
            "model": self.config.LLM_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            self.logger.error(f"Error calling OpenRouter API: {e}")
            return f"Error generating response: {str(e)}"
    
    def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tool calls from the LLM.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            
        Returns:
            Result from the tool
        """
        try:
            if tool_name == "calculate_bmi":
                return self.calculator.calculate_bmi(
                    parameters.get("weight_kg"),
                    parameters.get("height_m")
                )
            elif tool_name == "calculate_dosage":
                return self.calculator.calculate_dosage(
                    parameters.get("weight_kg"),
                    parameters.get("dosage_per_kg")
                )
            elif tool_name == "summarize":
                return {"summary": self.summarize_text(parameters.get("text", ""))}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            self.logger.error(f"Error handling tool call: {e}")
            return {"error": f"Error executing tool: {str(e)}"}