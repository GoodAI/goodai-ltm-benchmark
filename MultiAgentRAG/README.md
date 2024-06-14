# Useful commands:
python3.12 ./src/app.py
./generate_tree.sh
source rag-venv/bin/activate

# Multi-Agent RAG System with Continual Learning

This project implements a vertical slice of a multi-agent RAG (Retrieval-Augmented Generation) system with continual learning capabilities using LangChain, LangGraph, and LLM API calls.

## Setup

1. Install the required dependencies: "pip install -r .\MultiAgentRAG\requirements.txt"
2. Set up your OpenAI API key: "export OPENAI_API_KEY=your_api_key"
3. Prepare your data:
- Place your raw data files in the `data/raw` directory.
- The system will process the data and store the embeddings in the `data/embeddings` directory.

## Usage

Run the `app.py` file to start the interactive multi-agent RAG system: "python .\MultiAgentRAG\src\app.py"

Enter your queries and the system will retrieve relevant documents, process the query, generate a response, and store the query-response pair in memory for continual learning.

Type 'quit' to exit the program.

## Project Structure

- `data/`: Contains the raw, processed, and embeddings data.
- `notebooks/`: Jupyter notebooks for experimentation and analysis.
- `src/`: Source code for the multi-agent RAG system.
  - `agents/`: Implementations of individual agents (retrieval, processing, response).
  - `memory/`: Memory management for continual learning.
  - `utils/`: Utility functions for data processing.
  - `controller.py`: Central controller for orchestrating the agents and memory.
  - `app.py`: Main application entry point.
- `tests/`: Unit tests for the system (not implemented in this vertical slice).
- `requirements.txt`: Lists the required Python dependencies.
- `README.md`: Project documentation.