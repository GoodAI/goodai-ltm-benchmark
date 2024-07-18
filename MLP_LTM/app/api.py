# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import Agent
from app.db.memory_manager import MemoryManager
from app.config import config
from app.utils.logging import get_logger

app = FastAPI()
logger = get_logger('custom')

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

try:
    memory_manager = MemoryManager(config.DATABASE_URL)
    memory_manager.initialize()
    agent = Agent(config.TOGETHER_API_KEY, memory_manager)
except Exception as e:
    logger.error(f"Error initializing application: {str(e)}", exc_info=True)
    raise

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        response = await agent.process_query(request.query)
        logger.info(f"Query processed successfully")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query")

@app.get("/health")
async def health_check():
    return {"status": "ok"}