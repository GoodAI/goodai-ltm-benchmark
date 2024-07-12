from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging
from src.utils.tracing import setup_tracing, instrument_fastapi, tracer
from config import config
from src.utils.error_handling import global_exception_handler
from src.utils.structured_logging import get_logger
from src.utils.tracing import setup_tracing, instrument_fastapi, tracer
from src.utils.error_handling import global_exception_handler
import os

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Initialize logging
master_logger, chat_logger, memory_logger, database_logger = setup_logging()
logger = get_logger("api")
app = FastAPI()
app.add_exception_handler(Exception, global_exception_handler)
setup_tracing()
instrument_fastapi(app)
memory_analyzer = None

controller = None

@app.on_event("startup")
async def startup_event():
    global controller
    logger.info("Starting up the API server")
    controller = Controller()
    await controller.initialize()
    logger.info("API server initialization complete")

@app.on_event("shutdown")
async def shutdown_event():
    master_logger.info("Shutting down the API server")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    with tracer.start_as_current_span("query_processing"):
        try:
            logger.info(f"Received query: {request.query}")
            if controller is None or controller.agent is None:
                raise ValueError("Controller or Agent not initialized")
            response = await controller.execute_query(request.query)
            logger.info(f"Query processed successfully")
            return QueryResponse(response=response)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
@app.get("/memory_stats")
async def memory_stats_endpoint():
    try:
        stats = await controller.memory_manager.get_memory_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/link_distribution")
async def link_distribution_endpoint():
    try:
        distribution = await controller.memory_manager.analyze_link_distribution()
        return {"distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize_network")
async def visualize_network_endpoint():
    try:
        await controller.memory_manager.visualize_network()
        return {"message": "Network visualization generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    master_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    master_logger.info("Running the API server directly")
    uvicorn.run(app, host="0.0.0.0", port=8000)