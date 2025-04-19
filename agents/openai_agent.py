from typing import List
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import os
from dotenv import load_dotenv
from .base_agent import BaseSheetAgent

class OpenAISheetAgent(BaseSheetAgent):
    """Sheet agent implementation using OpenAI models."""
    
    def _initialize_llm(self):
        """Initialize the OpenAI language model."""
        try:
            load_dotenv()  # Load environment variables
            
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",  # Using GPT-4 for better reasoning
                temperature=0,
                verbose=True
            )
            
        except Exception as e:
            print(f"Error initializing OpenAI LLM: {str(e)}")
            raise

    @classmethod
    def create(cls, tools: List[Tool]) -> 'OpenAISheetAgent':
        """Factory method to create an OpenAI sheet agent."""
        return cls(tools) 