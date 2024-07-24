from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.root_controller import RootController
from models.nmn_agent import NMNAgent
from models.memory_needed_agent import MemoryNeededAgent
from controllers.spawned_controller import SpawnedController
from config import MAX_TOKENS_PER_AGENT, NMN_MODEL, MEMORY_MODEL, TOGETHER_API_KEY
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

try:
    spawned_controller = SpawnedController(MAX_TOKENS_PER_AGENT)
    nmn_agent = NMNAgent(NMN_MODEL)
    memory_needed_agent = MemoryNeededAgent(MEMORY_MODEL, spawned_controller)
    root_controller = RootController(nmn_agent, memory_needed_agent)
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Error initializing application: {str(e)}", exc_info=True)
    raise

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        response = root_controller.process_query(request.query)
        logger.info(f"Query processed successfully")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query")

@app.get("/health")
async def health_check():
    return {"status": "ok"}