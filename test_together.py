import os
from dotenv import load_dotenv
import together
import json

def test_together_connection():
    print("\nTesting Together AI Connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Set API key using environment variable
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "")
    
    print("\nTesting model completion...")
    try:
        response = together.Complete.create(
            prompt="What is 2+2? Respond with just the number.",
            model="meta-llama/Llama-3-70b-chat-hf",
            max_tokens=128,
        )
        
        print(f"\nRaw response: {json.dumps(response, default=str)}")
        
        if isinstance(response, dict):
            # Check for new response structure
            if 'choices' in response:
                text = response['choices'][0]['text'].strip()
                print(f"Response received: {text}")
                return
            
            # Check for legacy response structure
            if 'output' in response and 'choices' in response['output']:
                text = response['output']['choices'][0]['text'].strip()
                print(f"Response received: {text}")
                return
                
            print("Unexpected response structure:", json.dumps(response, default=str))
        else:
            print("Response is not a dictionary:", type(response))
            
    except Exception as e:
        print(f"Error during API call: {str(e)}")

if __name__ == "__main__":
    test_together_connection() 