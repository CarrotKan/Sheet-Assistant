from typing import List
import pandas as pd
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from utils.embedding_utils import get_embedding, compute_similarity
from utils.sheet_utils import get_top_k_similar_rows
import os
from openai import OpenAI

class SheetTools:
    """Collection of tools for working with spreadsheet data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def exact_match_search(self, query: str) -> str:
        """Search for exact matches of a value in the DataFrame."""
        try:
            matches = []
            for column in self.df.columns:
                mask = self.df[column].astype(str).str.contains(query, case=False, na=False)
                if mask.any():
                    matches.extend(self.df[mask].to_dict('records'))
            
            if not matches:
                return "No exact matches found."
            
            return str(matches)
        except Exception as e:
            return f"Error performing exact match search: {str(e)}"

    def similarity_search(self, query: str, k: int = 5) -> str:
        """Search for semantically similar rows using embeddings."""
        try:
            # Get query embedding
            query_embedding = get_embedding(query, self.client.api_key)
            
            # Get row embeddings if not already computed
            if not hasattr(self, 'row_embeddings'):
                from utils.sheet_utils import prepare_row_texts
                from utils.embedding_utils import create_embeddings_batch
                texts = prepare_row_texts(self.df)
                self.row_embeddings = create_embeddings_batch(texts, self.client)
            
            # Compute similarities
            similarities = compute_similarity(query_embedding, self.row_embeddings)
            
            # Get top k similar rows
            similar_rows, scores = get_top_k_similar_rows(similarities, self.df, k)
            
            # Format results
            results = []
            for (_, row), score in zip(similar_rows.iterrows(), scores):
                result = {
                    'similarity_score': f"{score:.2f}",
                    'data': row.to_dict()
                }
                results.append(result)
            
            return str(results)
        except Exception as e:
            return f"Error performing similarity search: {str(e)}"

    def get_column_values(self, column_name: str) -> str:
        """Get unique values from a specified column in the DataFrame."""
        try:
            if column_name not in self.df.columns:
                return f"Column '{column_name}' not found. Available columns: {list(self.df.columns)}"
            
            unique_values = self.df[column_name].unique().tolist()
            return str(unique_values)
        except Exception as e:
            return f"Error getting column values: {str(e)}"

    def get_tools(self) -> List[Tool]:
        """Get all available sheet tools."""
        return [
            Tool(
                name="exact_match_search",
                description="""Find rows containing an exact text match in any column. 
                Returns all rows where any column contains the search text (case-insensitive).
                Input: A text string to search for across all columns.""",
                func=self.exact_match_search
            ),
            Tool(
                name="similarity_search",
                description="""Find semantically similar rows using natural language understanding.
                Returns the top 5 most relevant rows based on the meaning of your query, not just exact matches.
                Each result includes a similarity score and the full row data.
                Input: A natural language description of what you're looking for.""",
                func=self.similarity_search
            ),
            Tool(
                name="get_column_values",
                description="""List all unique values present in a specific column.
                Useful for understanding the range of values in a column or finding available options.
                Input: The exact name of the column you want to inspect.""",
                func=self.get_column_values
            )
        ]

    @staticmethod
    def create(df: pd.DataFrame) -> 'SheetTools':
        """Create a new instance of SheetTools with the given DataFrame."""
        return SheetTools(df) 