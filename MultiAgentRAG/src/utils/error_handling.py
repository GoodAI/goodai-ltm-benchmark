from fastapi import Request
from fastapi.responses import JSONResponse
from src.utils.structured_logging import get_logger

error_logger = get_logger("error")

async def global_exception_handler(request: Request, exc: Exception):
    error_logger.error("Unhandled exception", 
                       url=str(request.url),
                       method=request.method,
                       error=str(exc),
                       exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )