# PDF Query System with Ollama

This project implements a Python-based query system that allows users to ask questions about the content of PDF documents. It uses a local Ollama server for language processing and a vector database to efficiently search through PDF contents.

## Usage

1. Run `make` to setup your Python environment.
1. Ensure you have Ollama installed and running locally.
2. Place your PDF files in a directory.
3. Run the `ollama_query.py` script with the `--pdf-dir` argument pointing to your PDF directory.
4. Enter questions when prompted. The system will search the PDFs for relevant information and use Ollama to generate answers.

Example:

```bash
  # Setup the Python environment
  make

  # Setup Ollama
  ollama run qwen2.5-coder

  # Start the program
  ./venv/bin/python3 ollama_query.py --pdf-dir ./PDFs --vector-db-dir ./vector_db
```

## Components

### 1. PDF Vector Database (`pdf_vdb.py`)

- `PDFVectorDatabase` class:
  - Loads PDF files from a specified directory
  - Splits text into chunks
  - Creates a Chroma vector database using HuggingFace embeddings
  - Provides a method to query the database for similar text chunks

### 2. Ollama Query System (`ollama_query.py`)

- `OllamaQuerySystem` class:
  - Initializes the PDF vector database and Ollama LLM
  - Sets up a prompt template for formatting context and questions
  - Creates an LLM chain that:
    1. Queries the PDF database for relevant context
    2. Passes the context and user's question to the Ollama LLM
  - Provides a `query` method to get responses from the LLM chain

### 3. Runnable Parsers (`runnable_parsers.py`)

- `DocumentMessageToString` class:
  - Converts `Document` objects to strings
  - Used in the query chain to process vector database results



Example:

```bash
python ollama_query.py --pdf-dir /path/to/your/pdfs
```

## Dependencies

- langchain
- PyPDF2
- Chroma
- HuggingFace Transformers
- Rich (for formatted console output)

Make sure to install all required dependencies before running the script.

## Note

This system uses a local Ollama server for language processing. Ensure that Ollama is properly set up and running on your machine before using this query system.
