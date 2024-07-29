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

class Query(BaseModel):
    query: str

try:
    memory_manager = MemoryManager(config.DATABASE_URL)
    agent = Agent(memory_manager)
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Error initializing application: {str(e)}", exc_info=True)
    raise

@app.post("/query")
async def query_endpoint(request: Query):
    try:
        response = await agent.process_query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}