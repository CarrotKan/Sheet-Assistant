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

    def query_dataframe(self, query: str) -> str:
        """Run a pandas query on the DataFrame.
        The query should be a valid Python expression that can be evaluated on a DataFrame.
        Examples:
        - 'df["column_name"].mean()'
        - 'df.groupby("category")["value"].sum()'
        - 'df[df["age"] > 30]["name"].tolist()'
        """
        try:
            # Create a safe copy of the DataFrame
            df = self.df.copy()
            
            # Execute the query
            result = eval(query)
            
            # Convert different types of results to string representation
            if isinstance(result, pd.DataFrame):
                if len(result) > 10:
                    return f"DataFrame with {len(result)} rows:\n{result.head(10).to_string()}\n... ({len(result)-10} more rows)"
                return result.to_string()
            elif isinstance(result, pd.Series):
                if len(result) > 10:
                    return f"Series with {len(result)} items:\n{result.head(10).to_string()}\n... ({len(result)-10} more items)"
                return result.to_string()
            elif isinstance(result, (list, tuple)):
                if len(result) > 10:
                    return f"List with {len(result)} items: {str(result[:10])}... ({len(result)-10} more items)"
                return str(result)
            else:
                return str(result)
                
        except Exception as e:
            return f"Error executing query: {str(e)}\nPlease ensure the query is valid Python code and uses 'df' as the DataFrame variable."

    def get_column_values(self, column_name: str) -> str:
        """Get unique values from a specified column in the DataFrame."""
        try:
            if column_name not in self.df.columns:
                return f"Column '{column_name}' not found. Available columns: {list(self.df.columns)}"
            
            unique_values = self.df[column_name].unique().tolist()
            return str(unique_values)
        except Exception as e:
            return f"Error getting column values: {str(e)}"

    def sample_columns(self, num_columns: str = "5") -> str:
        """Get a sample of random columns with their data types and example values.
        This helps understand the structure and content of the DataFrame."""
        try:
            # Convert input to integer and validate
            try:
                num_columns = int(num_columns)
                if num_columns < 1:
                    return "Please provide a positive number of columns to sample (minimum 1)."
            except ValueError:
                return "Please provide a valid number for columns to sample."
            
            # Get all columns
            all_columns = list(self.df.columns)
            
            # If num_columns is greater than available columns, use all columns
            num_columns = min(num_columns, len(all_columns))
            
            # Randomly sample columns if we have more than requested
            import random
            sampled_columns = random.sample(all_columns, num_columns)
            
            # Build the response
            response = [f"Here's a sample of {num_columns} column(s) from the DataFrame ({len(all_columns)} total columns):"]
            
            for col in sampled_columns:
                # Get column info
                dtype = str(self.df[col].dtype)
                non_null = self.df[col].count()
                total = len(self.df)
                null_count = total - non_null
                
                # Get sample values (up to 3)
                sample_values = self.df[col].dropna().sample(min(3, non_null)).tolist()
                
                # Add some basic statistics based on data type
                stats = []
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    stats.extend([
                        f"Min: {self.df[col].min():.2f}",
                        f"Max: {self.df[col].max():.2f}",
                        f"Mean: {self.df[col].mean():.2f}"
                    ])
                elif pd.api.types.is_string_dtype(self.df[col]):
                    value_counts = self.df[col].value_counts()
                    if not value_counts.empty:
                        stats.append(f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
                
                # Format the response
                col_info = f"\n• {col} (Type: {dtype})"
                col_info += f"\n  - Non-null count: {non_null}/{total} ({null_count} null values)"
                if stats:
                    col_info += f"\n  - Statistics: {', '.join(stats)}"
                col_info += f"\n  - Example values: {sample_values}"
                
                response.append(col_info)
            
            return "\n".join(response)
            
        except Exception as e:
            return f"Error sampling columns: {str(e)}"

    def get_tools(self) -> List[Tool]:
        """Get all available sheet tools."""
        return [
            Tool(
                name="sample_columns",
                description="""Get a random sample of columns from the DataFrame to understand its structure.
                Shows column names, data types, null value counts, and example values.
                For numeric columns, includes basic statistics (min, max, mean).
                For text columns, shows the most common value.
                Input: Number of columns to sample (default is 5). Must be a positive number.
                Example inputs: "3", "5", "10" """,
                func=self.sample_columns
            ),
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
                name="query_dataframe",
                description="""Run a pandas query on the DataFrame to perform data analysis.
                Input should be a valid Python expression using 'df' as the DataFrame variable.
                Examples:
                - Basic statistics: df["column_name"].mean()
                - Grouping: df.groupby("category")["value"].sum()
                - Filtering: df[df["age"] > 30]["name"].tolist()
                - Column operations: df["column_name"].value_counts()
                The query must be valid Python code and can use any pandas DataFrame methods.""",
                func=self.query_dataframe
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