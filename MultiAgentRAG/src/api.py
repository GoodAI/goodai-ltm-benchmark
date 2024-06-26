from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.utils.controller import Controller
from src.utils.logging_setup import setup_logging
from config import config
from src.utils.memory_analysis import MemoryAnalyzer

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Initialize logging
master_logger, chat_logger, memory_logger, database_logger = setup_logging()

app = FastAPI()

controller = Controller()
memory_analyzer = None

@app.on_event("startup")
async def startup_event():
    global memory_analyzer
    master_logger.info("Starting up the API server")
    await controller.initialize()
    memory_analyzer = MemoryAnalyzer(controller.memory_manager)

@app.on_event("shutdown")
async def shutdown_event():
    master_logger.info("Shutting down the API server")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        response = await controller.execute_query(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        master_logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/memory_distribution")
async def memory_distribution_endpoint():
    try:
        distribution = await memory_analyzer.analyze_distribution()
        return {"distribution": distribution}
    except Exception as e:
        master_logger.error(f"Error analyzing memory distribution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing memory distribution: {str(e)}")
    
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    master_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

@app.get("/consistency_check")
async def consistency_check_endpoint():
    try:
        await controller.memory_manager.run_consistency_check_and_fix()
        return {"message": "Consistency check and fix completed."}
    except Exception as e:
        master_logger.error(f"Error during consistency check: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during consistency check: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    master_logger.info("Running the API server directly")
    uvicorn.run(app, host="0.0.0.0", port=8000)