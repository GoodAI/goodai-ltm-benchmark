from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import Agent
from app.db.memory_manager import MemoryManager
from app.config import config
from app.utils.logging import get_logger

app = FastAPI()
logger = get_logger(__name__)

class Query(BaseModel):
    text: str

agent = Agent(config.TOGETHER_API_KEY, MemoryManager(config.DATABASE_URL))

@app.post("/query")
async def query_endpoint(query: Query):
    try:
        logger.info(f"Received query: {query.text}")
        response = await agent.process_query(query.text)
        logger.info(f"Query processed successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
