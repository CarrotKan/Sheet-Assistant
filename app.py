import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import time

from utils.embedding_utils import get_embedding, create_embeddings, compute_similarity
from utils.sheet_utils import (
    read_excel_file, prepare_row_texts, get_top_k_similar_rows,
    find_exact_matches, combine_search_results
)
from utils.chat_utils import generate_context, get_chat_response
from agents import SheetTools, OpenAISheetAgent

# Set page configuration
st.set_page_config(
    page_title="Sheet Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stAlert {
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'row_embeddings' not in st.session_state:
    st.session_state.row_embeddings = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'agent' not in st.session_state:
    st.session_state.agent = None

def process_query_with_agents(query: str, df: pd.DataFrame) -> str:
    """Process a query using Langchain agent with OpenAI."""
    try:
        # Initialize or get agent from session state
        if st.session_state.agent is None:
            # Initialize tools and agent
            sheet_tools = SheetTools.create(df)
            tools = sheet_tools.get_tools()
            st.session_state.agent = OpenAISheetAgent.create(tools)
        
        # Process the query
        with st.status("🤖 AI Assistant is analyzing your data...", expanded=True) as status:
            st.write("Processing your query...")
            response = st.session_state.agent.process_query(query)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            return response
    except Exception as e:
        st.error(f"Error in agent processing: {str(e)}")
        return f"Error: {str(e)}"

def process_file(uploaded_file):
    """Process the uploaded file and create embeddings."""
    try:
        progress = st.progress(0, "Starting file processing...")
        steps = ["Saving file", "Reading Excel", "Processing data", "Creating embeddings"]
        current_step = 0
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            current_step += 1
            progress.progress(current_step/len(steps), f"✓ File saved | {steps[current_step]}...")

        try:
            # Read the Excel file
            with st.status("📑 Reading Excel file...", expanded=True) as status:
                df = read_excel_file(tmp_path)
                current_step += 1
                progress.progress(current_step/len(steps), f"✓ Excel read | {steps[current_step]}...")
                status.update(label="✅ File read successfully!", state="complete")
            
            # Store DataFrame in session state
            st.session_state.df = df
            
            # Process the data
            current_step += 1
            progress.progress(current_step/len(steps), f"✓ Data processed | {steps[current_step]}...")
            
            # Display the dataframe in an expander
            with st.expander("📊 Preview your data", expanded=True):
                total_rows = len(df)
                total_cols = len(df.columns)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", f"{total_rows:,}")
                with col2:
                    st.metric("Total Columns", total_cols)
                
                st.dataframe(
                    df.head(),
                    use_container_width=True,
                    hide_index=True
                )

            # Create embeddings
            with st.status("🔄 Creating embeddings...", expanded=True) as status:
                texts = prepare_row_texts(df)
                total_texts = len(texts)
                
                # Add warning for large datasets
                if total_texts > 1000:
                    st.warning("⚠️ Large dataset detected! Processing might take longer and could hit API rate limits. Consider reducing the dataset size.")
                
                try:
                    # Create embedding progress bar
                    embedding_progress = st.progress(0, "Starting embedding creation...")
                    status_text = st.empty()
                    
                    def update_progress(current, total):
                        """Update progress bar and status text."""
                        progress_val = current / total
                        embedding_progress.progress(
                            progress_val,
                            f"Processing item {current:,} of {total:,} ({progress_val:.1%})"
                        )
                        status_text.text(f"Created embeddings for {current:,} of {total:,} items...")
                    
                    # Create embeddings with progress tracking
                    st.session_state.row_embeddings = create_embeddings(
                        texts,
                        client,
                        progress_callback=update_progress
                    )
                    
                    # Clear progress indicators
                    embedding_progress.empty()
                    status_text.empty()
                    
                    # Update main progress
                    current_step += 1
                    progress.progress(1.0, "✅ All steps completed!")
                    status.update(label="✅ Embeddings created successfully!", state="complete")
                    
                    # Set processing flag
                    st.session_state.file_processed = True
                    
                    # Show success message
                    st.success(f"""
                        ✅ File processed successfully!
                        - {total_rows:,} rows processed
                        - {total_cols} columns analyzed
                        - Embeddings created for all entries
                    """)
                    
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        st.error("🚫 Hit OpenAI API rate limit. Please wait a moment and try again, or reduce the dataset size.")
                    else:
                        raise e

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.session_state.file_processed = False
            raise e
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"❌ Error handling file: {str(e)}")
        st.session_state.file_processed = False

def display_chat_messages():
    """Display chat messages from the agent's memory."""
    if st.session_state.agent:
        messages = st.session_state.agent.get_memory()
        for msg in messages:
            # Skip system messages
            if msg.type == "system":
                continue
            # Display user messages
            if msg.type == "human":
                with st.chat_message("user"):
                    st.write(msg.content)
            # Display assistant messages
            elif msg.type == "ai":
                with st.chat_message("assistant"):
                    st.write(msg.content)

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 20px'>
                <h1 style='color: #4A90E2; margin-bottom: 0;'>📊</h1>
                <h2 style='color: #4A90E2; margin-top: 0;'>Sheet Assistant</h2>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Upload your Excel file 📄
        2. Wait for processing ⚙️
        3. Ask questions about your data 💬
        """)
        st.markdown("---")
        
        # Reset button in sidebar
        if st.session_state.file_processed:
            if st.button("🔄 Reset Session", use_container_width=True):
                if st.session_state.agent:
                    st.session_state.agent.clear_memory()
                st.session_state.df = None
                st.session_state.row_embeddings = None
                st.session_state.file_processed = False
                st.session_state.agent = None
                st.rerun()

    # Main content area
    st.markdown("<h1 style='text-align: center;'>📊 Excel Data Analysis Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your Excel file and ask questions about your data in natural language!</p>", unsafe_allow_html=True)

    # File upload section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        uploaded_file = st.file_uploader("", type=['xlsx'], key='file_uploader')

    # Process the file if it's newly uploaded
    if uploaded_file is not None and not st.session_state.file_processed:
        process_file(uploaded_file)

    # Query section
    if st.session_state.file_processed and st.session_state.df is not None:
        st.markdown("---")
        
        # Chat interface
        st.markdown("<h3 style='text-align: center;'>💬 Chat with your Data</h3>", unsafe_allow_html=True)
        
        # Display chat history from agent's memory
        display_chat_messages()
        
        # Query input
        query = st.chat_input("Ask a question about your data...")
        
        if query:
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                try:
                    # Process query
                    response = process_query_with_agents(query, st.session_state.df)
                    st.write(response)
                    
                    # Show similarity results in an expander
                    with st.expander("🔍 View similar entries in your data"):
                        try:
                            query_embedding = get_embedding(query, client.api_key)
                            similarities = compute_similarity(query_embedding, st.session_state.row_embeddings)
                            matches_df, scores = get_top_k_similar_rows(similarities, st.session_state.df)
                            
                            if not matches_df.empty and len(scores) > 0:
                                for idx, (_, row) in enumerate(matches_df.iterrows()):
                                    if idx < len(scores):  # Ensure we have a score for this row
                                        st.metric(
                                            label=f"Match #{idx + 1}",
                                            value=f"{scores[idx]:.2%}"
                                        )
                                        # Convert row to dictionary safely
                                        try:
                                            row_data = row.to_dict()
                                        except:
                                            row_data = {col: str(val) for col, val in row.items()}
                                        st.json(row_data)
                                        st.markdown("---")
                            else:
                                st.info("No similar entries found in the data.")
                        except Exception as e:
                            st.warning(f"Could not compute similarity results: {str(e)}")
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 