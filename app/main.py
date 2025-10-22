"""
Content Moderation ML Microservice - Main Application

A FastAPI-based machine learning microservice for content moderation of music album reviews.
This service uses transformer models to detect toxic content and spam in Spanish text.

Author: ML Team
Version: 1.0.0
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.ml_models import ml_models
from app.api.routes import classification, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    This context manager handles startup and shutdown events:
    - Startup: Load ML models into memory
    - Shutdown: Cleanup resources (if needed)

    Args:
        app: FastAPI application instance

    Yields:
        None: Control flow during application runtime
    """
    # Startup
    logger.info("Starting application...")
    try:
        await ml_models.load_models()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")
    # Add any cleanup code here if needed


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and monitoring endpoints",
        },
        {
            "name": "classification",
            "description": "Content classification endpoints for toxicity and spam detection",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.

    Args:
        request: The request that caused the exception (unused but required by FastAPI)
        exc: The exception that was raised

    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


# Include routers
app.include_router(health.router)
app.include_router(classification.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
