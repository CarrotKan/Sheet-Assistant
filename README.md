# Sheet Assistant

A Streamlit-based application that allows users to upload Excel sheets and ask questions about their data using OpenAI's GPT-4 and embeddings.

## Features

- Excel file upload support
- Semantic search using OpenAI embeddings
- Natural language querying with GPT-4
- Interactive UI with Streamlit
- Display of relevant rows with similarity scores

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)
3. Upload an Excel file (.xlsx)
4. Ask questions about your data in natural language
5. View the AI-generated responses and relevant rows from your data

## Project Structure

- `app.py`: Main Streamlit application
- `utils/`
  - `embedding_utils.py`: Functions for handling OpenAI embeddings
  - `sheet_utils.py`: Excel file processing utilities
  - `chat_utils.py`: GPT-4 interaction utilities
- `requirements.txt`: Project dependencies
- `.env.example`: Template for environment variables

## Requirements

- Python 3.8+
- OpenAI API key
- Excel files (.xlsx format) 