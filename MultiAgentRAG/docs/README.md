# Multi-Agent RAG System with Continual Learning

## Project Overview

This project implements a vertical slice of a multi-agent RAG (Retrieval-Augmented Generation) system with continual learning capabilities. It uses LangChain, LangGraph, and LLM API calls to create a sophisticated question-answering system that learns from interactions.

## Key Features

- Multi-agent architecture for complex query processing
- RAG (Retrieval-Augmented Generation) for enhanced responses
- Continual learning to improve performance over time
- Distributed memory management with personal and master databases
- CLI and API interfaces for versatile interaction
- Comprehensive logging system for debugging and analysis
- Docker support for easy deployment and scaling

## System Architecture

The system consists of the following main components:

1. **Controller**: Orchestrates the overall flow of query processing.
2. **Agents**: Specialized modules for retrieval, processing, and response generation.
3. **Memory Manager**: Handles storage and retrieval of past interactions.
4. **Logging System**: Provides detailed logs for system operations, including database interactions.

## Directory Structure

```
.
├── Dockerfile
├── README.md
├── config.py
├── data/
├── docker-compose.yml
├── docs/
├── logs/
├── notebooks/
├── scripts/
├── src/
│   ├── agents/
│   ├── memory/
│   ├── utils/
│   ├── api.py
│   ├── app.py
│   └── cli.py
└── tests/
```

## Setup and Installation

### Prerequisites

- Python 3.12+
- Docker and Docker Compose

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/multi-agent-rag.git
   cd multi-agent-rag
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Docker Setup

1. Build the Docker image:
   ```
   docker-compose build
   ```

2. Run the container:
   ```
   docker-compose up
   ```

## Usage

### CLI Interface

Run the CLI interface with:

```
python src/cli.py
```

Available commands:
- `query`: Enter a query to process
- `memories`: Retrieve recent memories
- `consistency`: Run a consistency check on the databases
- `quit`: Exit the program

### API Interface

Start the API server with:

```
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API endpoints:
- POST `/query`: Submit a query for processing
- GET `/consistency_check`: Run a consistency check on the databases
- GET `/health`: Check the health status of the API
- GET `/memory_distribution`: Get the distribution of memories across containers

## Logging

Logs are stored in the `logs/` directory, organized by timestamp and container ID. There are separate log files for:

- Master log (`master.log`)
- Chat log (`chat.log`)
- Memory operations log (`memory.log`)
- Database operations log (`database.log`)

## Testing

Run the test suite with:

```
pytest tests/
```
