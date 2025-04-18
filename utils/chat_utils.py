from openai import OpenAI
import pandas as pd
from typing import List

def generate_context(df: pd.DataFrame, similarities: List[float], k: int = 5) -> str:
    """Generate context from top k similar rows."""
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    context_parts = []
    
    for idx in top_indices:
        row = df.iloc[idx]
        row_text = " | ".join(f"{col}: {str(val)}" for col, val in row.items())
        context_parts.append(row_text)
    
    return "\n".join(context_parts)

def get_chat_response(client: OpenAI, query: str, context: str) -> str:
    """Get response from GPT-4 based on the query and context."""
    system_prompt = """You are a helpful assistant that answers questions about spreadsheet data. 
    Use the provided context to answer questions accurately. If you're not sure about something, 
    say so rather than making assumptions."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content 