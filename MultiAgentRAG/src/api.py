from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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
        response = await controller.execute_query(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        master_logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    master_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

@app.on_event("startup")
async def startup_event():
    master_logger.info("Starting up the API server")
    # Initialize any resources here, if needed

@app.on_event("shutdown")
async def shutdown_event():
    master_logger.info("Shutting down the API server")
    # Clean up any resources here, if needed

@app.get("/health")
async def health_check():
    return {"status": "ok"}