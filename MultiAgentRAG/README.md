# Multi-Agent RAG System with Continual Learning

This project implements a vertical slice of a multi-agent RAG (Retrieval-Augmented Generation) system with continual learning capabilities using LangChain, LangGraph, and LLM API calls.

## Setup

### Prerequisites

- Python 3.12
- Docker (for containerization)

### Installation

1. Clone the repository:
   ```sh
   `git clone https://github.com/your-username/your-repo.git`
   `cd your-repo/MultiAgentRAG`

2. Create a virtual environment and activate it:
  `python3.12 -m venv rag-venv`
  `source rag-venv/bin/activate`

3. Install the required dependencies:
  `pip install -r requirements.txt`

4. Set up your OpenAI API key:
  `export OPENAI_API_KEY=your_openai_api_key`

5. Prepare Your Data
  Place your raw data files in the data/raw directory.
  The system will process the data and store the embeddings in the data/embeddings directory.

### Usage
Run the app.py file to start the interactive multi-agent RAG system:
  `python src/app.py`

Enter your queries, and the system will retrieve relevant documents, process the query, generate a response, and store the query-response pair in memory for continual learning.

Type `'quit'` to exit the program.

### Project Structure
data/: Contains the raw, processed, and embeddings data.
json_output/: Stores the JSON output files for the memories.
logs/: Contains log files.
notebooks/: Jupyter notebooks for experimentation and analysis.
scripts/: Contains utility scripts.
generate_tree.sh: Script to generate the directory tree.
logs_to_docs.sh: Script to convert logs to documents.
reset_logs.sh: Script to reset logs.
src/: Source code for the multi-agent RAG system.
agents/: Implementations of individual agents (retrieval, processing, response).
memory/: Memory management for continual learning.
utils/: Utility functions for data processing.
controller.py: Central controller for orchestrating the agents and memory.
app.py: Main application entry point.
requirements.txt: Lists the required Python dependencies.
README.md: Project documentation.

Running with Docker
Build the Docker image:
  `docker build -t multi-agent-rag .`
Run the Docker container:

`docker run -e OPENAI_API_KEY=your_openai_api_key -p 8000:8000 multi-agent-rag`
Access the interactive multi-agent RAG system by connecting to the container.