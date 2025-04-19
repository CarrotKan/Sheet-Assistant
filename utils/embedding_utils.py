from openai import OpenAI
import numpy as np
from typing import List, Dict
import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_PARALLEL_REQUESTS = 5  # Limit parallel requests to avoid rate limits

def get_embedding(text: str, api_key: str) -> List[float]:
    """Get embeddings for a single text using OpenAI's API."""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def create_embeddings(texts: List[str], client: OpenAI, progress_callback=None) -> List[List[float]]:
    """Create embeddings for texts with individual progress tracking."""
    embeddings = []
    total_texts = len(texts)
    
    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(get_embedding, text, client.api_key): idx 
            for idx, text in enumerate(texts)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                embedding = future.result()
                embeddings.append((idx, embedding))
                
                # Update progress
                if progress_callback:
                    progress_callback(idx + 1, total_texts)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error processing item {idx + 1}: {str(e)}")
                raise e
    
    # Sort embeddings by original index
    embeddings.sort(key=lambda x: x[0])
    return [emb for _, emb in embeddings]

def compute_similarity(query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
    """Compute cosine similarity between query and all documents."""
    query_embedding = np.array(query_embedding)
    document_embeddings = np.array(document_embeddings)
    
    # Normalize the embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norm = document_embeddings / np.linalg.norm(document_embeddings, axis=1)[:, np.newaxis]
    
    # Compute cosine similarity
    similarities = np.dot(doc_norm, query_norm)
    return similarities.tolist() 