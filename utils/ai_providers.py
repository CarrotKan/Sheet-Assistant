from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Optional
import google.generativeai as genai
import together

class AIProvider(ABC):
    @abstractmethod
    def initialize(self) -> None:
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class GoogleAIProvider(AIProvider):
    def __init__(self, model_name: str = 'gemini-2.0-flash'):
        self.model_name = model_name
        self.model = None
    
    def initialize(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate_response(self, prompt: str) -> str:
        if not self.model:
            self.initialize()
        response = self.model.generate_content(prompt)
        return response.text

class TogetherAIProvider(AIProvider):
    def __init__(self, model_name: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1'):
        self.model_name = model_name
        self.initialized = False
    
    def initialize(self) -> None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        together.api_key = api_key
        self.initialized = True
    
    def generate_response(self, prompt: str) -> str:
        if not self.initialized:
            self.initialize()
        
        output = together.Complete.create(
            prompt=prompt,
            model=self.model_name,
            max_tokens=1024,
            temperature=0.7,
        )
        
        return output['output']['choices'][0]['text']

def get_ai_provider(provider_name: str = "google", model_name: Optional[str] = None) -> AIProvider:
    providers = {
        "google": (GoogleAIProvider, "gemini-2.0-flash"),
        "together": (TogetherAIProvider, "mistralai/Mixtral-8x7B-Instruct-v0.1")
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {list(providers.keys())}")
    
    provider_class, default_model = providers[provider_name]
    return provider_class(model_name or default_model) 