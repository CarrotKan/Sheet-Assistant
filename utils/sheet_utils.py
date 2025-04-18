import pandas as pd
from typing import List, Dict, Tuple, Union
import numpy as np

def read_excel_file(file_path: str) -> pd.DataFrame:
    """Read an Excel file and return a pandas DataFrame."""
    return pd.read_excel(file_path)

def prepare_row_texts(df: pd.DataFrame) -> List[str]:
    """Convert DataFrame rows to text format for embedding."""
    texts = []
    for _, row in df.iterrows():
        # Convert row to string, handling non-string types
        row_text = " ".join(f"{col}: {str(val)}" for col, val in row.items())
        texts.append(row_text)
    return texts

def find_exact_matches(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, List[float]]:
    """Find exact matches of the query in any column of the DataFrame."""
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Initialize a mask for exact matches
    exact_matches = pd.Series(False, index=df.index)
    
    # Search through each column
    for column in df.columns:
        # Convert column values to string and lowercase for comparison
        column_values = df[column].astype(str).str.lower()
        # Update mask for exact matches
        exact_matches |= column_values.str.contains(query_lower, regex=False, na=False)
    
    # Get matching rows
    matching_rows = df[exact_matches]
    
    # If we found exact matches, assign them high similarity scores
    similarities = [1.0] * len(matching_rows) if not matching_rows.empty else []
    
    return matching_rows, similarities

def get_top_k_similar_rows(similarities: List[float], df: pd.DataFrame, k: int = 5) -> Tuple[pd.DataFrame, List[float]]:
    """Get the top k most similar rows and their similarity scores."""
    if not similarities:
        return pd.DataFrame(), []
        
    # Get indices of top k similar rows
    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    
    # Get the corresponding rows and similarities
    top_k_rows = df.iloc[top_k_indices]
    top_k_similarities = [similarities[i] for i in top_k_indices]
    
    return top_k_rows, top_k_similarities

def combine_search_results(exact_matches: Tuple[pd.DataFrame, List[float]], 
                         semantic_matches: Tuple[pd.DataFrame, List[float]], 
                         k: int = 5) -> Tuple[pd.DataFrame, List[float]]:
    """Combine exact and semantic matches, prioritizing exact matches."""
    exact_df, exact_scores = exact_matches
    semantic_df, semantic_scores = semantic_matches
    
    # If we have exact matches, prioritize them
    if not exact_df.empty:
        if len(exact_df) >= k:
            return exact_df.head(k), exact_scores[:k]
        
        # If we have some exact matches but need more results
        remaining_k = k - len(exact_df)
        # Remove exact matches from semantic results to avoid duplicates
        semantic_df = semantic_df[~semantic_df.index.isin(exact_df.index)]
        
        # Combine results
        combined_df = pd.concat([exact_df, semantic_df.head(remaining_k)])
        combined_scores = exact_scores + semantic_scores[:remaining_k]
        
        return combined_df, combined_scores
    
    # If no exact matches, return semantic matches
    return semantic_df.head(k), semantic_scores[:k] 