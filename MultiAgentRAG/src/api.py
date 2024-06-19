# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.controller import Controller  # Update this line
import os
import logging

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory_db_path = "/app/memory.db" if os.path.exists("/.dockerenv") else "memory.db"
controller = Controller("gpt-3.5-turbo", memory_db_path, os.getenv("GOODAI_OPENAI_API_KEY_LTM01"))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        response = controller.execute_query(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
