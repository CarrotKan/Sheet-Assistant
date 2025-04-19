import os
from dotenv import load_dotenv
from utils.ai_providers import get_ai_provider

def test_ai_provider(provider_name: str = "google"):
    print(f"Testing {provider_name.capitalize()} AI Connection...")
    try:
        # Load environment variables
        load_dotenv()
        
        # Get the provider
        provider = get_ai_provider(provider_name)
        
        # Test with a simple query
        prompt = "What is 2+2? Give a simple numeric answer."
        print(f"\nSending prompt: {prompt}")
        
        response = provider.generate_response(prompt)
        print(f"\nResponse from {provider_name.capitalize()}:", response)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

if __name__ == "__main__":
    # Test Google AI
    try:
        test_ai_provider("google")
    except Exception as e:
        print("Google AI test failed")
    
    print("\n" + "="*50 + "\n")
    
    # Test Together AI
    try:
        test_ai_provider("together")
    except Exception as e:
        print("Together AI test failed") 