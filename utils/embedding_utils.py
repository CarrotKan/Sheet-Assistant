from openai import OpenAI
import numpy as np
from typing import List, Dict
import streamlit as st
from multiprocessing import Pool, cpu_count
import os
from functools import partial
import time

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # Number of texts to process in each batch

def get_embedding(text: str, api_key: str) -> List[float]:
    """Get embeddings for a single text using OpenAI's API."""
    # Create a new client for each process
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def process_batch(texts: List[str], api_key: str) -> List[List[float]]:
    """Process a batch of texts to get their embeddings."""
    return [get_embedding(text, api_key) for text in texts]

def create_embeddings_batch(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Create embeddings for a batch of texts with parallel processing."""
    api_key = client.api_key
    total_texts = len(texts)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate optimal number of processes
    num_processes = min(cpu_count(), 4)  # Limit to 4 processes to avoid API rate limits
    
    # Split texts into batches
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    total_batches = len(batches)
    
    embeddings = []
    with Pool(processes=num_processes) as pool:
        # Create a partial function with the API key
        process_func = partial(process_batch, api_key=api_key)
        
        # Process batches with progress tracking
        for i, batch_embeddings in enumerate(pool.imap(process_func, batches)):
            embeddings.extend(batch_embeddings)
            
            # Update progress
            progress = (i + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {i + 1} of {total_batches} ({len(embeddings)} of {total_texts} rows completed)")
            
            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.1)
    
    # Clear the status text and progress bar
    status_text.empty()
    progress_bar.empty()
    
    st.success(f"Successfully created embeddings for {total_texts} rows using {EMBEDDING_MODEL}")
    return embeddings

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