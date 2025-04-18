import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import time

from utils.embedding_utils import get_embedding, create_embeddings_batch, compute_similarity
from utils.sheet_utils import (
    read_excel_file, prepare_row_texts, get_top_k_similar_rows,
    find_exact_matches, combine_search_results
)
from utils.chat_utils import generate_context, get_chat_response

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'row_embeddings' not in st.session_state:
    st.session_state.row_embeddings = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

def process_file(uploaded_file):
    """Process the uploaded file and create embeddings."""
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Read the Excel file
            with st.spinner("Reading Excel file..."):
                df = read_excel_file(tmp_path)
            st.success("File uploaded successfully!")
            
            # Store DataFrame in session state
            st.session_state.df = df
            
            # Display the dataframe
            st.write("Preview of your data:")
            st.dataframe(df.head())

            # Create embeddings with parallel processing
            st.info("Creating embeddings for your data using parallel processing. This might take a moment...")
            texts = prepare_row_texts(df)
            
            # Add warning for large datasets
            if len(texts) > 1000:
                st.warning(f"Processing {len(texts)} rows. This might take a while and could hit API rate limits. Consider reducing the dataset size.")
            
            try:
                st.session_state.row_embeddings = create_embeddings_batch(texts, client)
                st.session_state.file_processed = True
            except Exception as e:
                if "rate limit" in str(e).lower():
                    st.error("Hit OpenAI API rate limit. Please wait a moment and try again, or reduce the dataset size.")
                else:
                    raise e

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.file_processed = False
            raise e
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"Error handling file: {str(e)}")
        st.session_state.file_processed = False

def main():
    st.title("📊 Sheet Assistant")
    st.write("Upload your Excel file and ask questions about your data!")

    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'], key='file_uploader')

    # Process the file if it's newly uploaded
    if uploaded_file is not None and not st.session_state.file_processed:
        process_file(uploaded_file)

    # Only show query input if file has been processed successfully
    if st.session_state.file_processed and st.session_state.df is not None:
        # Query input
        query = st.text_input("Ask a question about your data:")
        
        if query:
            try:
                with st.spinner("Processing your question..."):
                    # First, try exact matching
                    exact_matches = find_exact_matches(st.session_state.df, query)
                    
                    # Then do semantic search
                    query_embedding = get_embedding(query, client.api_key)  # Updated to pass api_key
                    similarities = compute_similarity(query_embedding, st.session_state.row_embeddings)
                    semantic_matches = get_top_k_similar_rows(similarities, st.session_state.df)
                    
                    # Combine results
                    final_matches, final_similarities = combine_search_results(exact_matches, semantic_matches)
                    
                    if not final_matches.empty:
                        # Get relevant context from combined matches
                        context = generate_context(final_matches, final_similarities)
                        
                        # Get response from GPT-4
                        response = get_chat_response(client, query, context)
                        
                        # Display response
                        st.write("Answer:")
                        st.write(response)
                        
                        # Display relevant rows
                        st.write("Matching rows from your data:")
                        for (_, row), similarity in zip(final_matches.iterrows(), final_similarities):
                            match_type = "Exact Match" if similarity >= 1.0 else f"Similarity: {similarity:.2f}"
                            with st.expander(match_type):
                                st.write(row.to_dict())
                    else:
                        st.warning("No matches found in the data. Try rephrasing your query.")
                        
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Add a reset button
    if st.session_state.file_processed:
        if st.button("Reset"):
            st.session_state.df = None
            st.session_state.row_embeddings = None
            st.session_state.file_processed = False
            st.rerun()

if __name__ == "__main__":
    main() 