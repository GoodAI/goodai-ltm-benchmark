from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import Agent
from app.db.memory_manager import MemoryManager
from app.config import config
from app.utils.logging import get_logger

app = FastAPI()
logger = get_logger(__name__)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

memory_manager = MemoryManager(config.DATABASE_URL)
memory_manager.initialize()  # Call this synchronously
agent = Agent(config.TOGETHER_API_KEY, memory_manager)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        response = await agent.process_query(request.query)
        logger.info(f"Query processed successfully with response: {response}")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}