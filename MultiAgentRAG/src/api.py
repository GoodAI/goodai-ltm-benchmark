# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging
from config import config

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Initialize logging
master_logger, chat_logger, memory_logger = setup_logging()

app = FastAPI()

controller = Controller()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        response = controller.execute_query(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        master_logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))