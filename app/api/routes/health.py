"""
Health check endpoints.

This module provides endpoints for monitoring the service health
and readiness status.
"""

import logging
from fastapi import APIRouter, status

from app.schemas.classification import HealthResponse
from app.models.ml_models import ml_models
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root endpoint",
    description="Basic service information and status check",
    response_model=dict,
)
async def root():
    """
    Root endpoint providing basic service information.

    Returns:
        Basic service status and information
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "models_loaded": ml_models.models_loaded(),
        "docs": "/docs",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="""
    Detailed health check endpoint for monitoring.

    This endpoint provides comprehensive information about:
    - Overall service status
    - Model loading status (toxicity and spam models)
    - Computation device being used (CPU or GPU)
    - API version

    Use this endpoint for:
    - Kubernetes/Docker health checks
    - Monitoring and alerting systems
    - Verifying service readiness after deployment
    """,
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "toxicity_model_loaded": True,
                        "spam_model_loaded": True,
                        "device": "cuda:0",
                        "version": "1.0.0",
                    }
                }
            },
        }
    },
)
async def health_check() -> HealthResponse:
    """
    Detailed health check endpoint.

    Returns:
        HealthResponse with detailed service status information
    """
    toxicity_loaded = (
        ml_models.toxicity_model is not None
        and ml_models.toxicity_tokenizer is not None
    )
    spam_loaded = (
        ml_models.spam_model is not None and ml_models.spam_tokenizer is not None
    )

    return HealthResponse(
        status="healthy" if ml_models.models_loaded() else "degraded",
        toxicity_model_loaded=toxicity_loaded,
        spam_model_loaded=spam_loaded,
        device=str(ml_models.device) if ml_models.device else "unknown",
        version=settings.app_version,
    )


@router.get(
    "/readiness",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="""
    Kubernetes-style readiness probe endpoint.

    Returns 200 if the service is ready to accept requests (models loaded),
    503 if the service is not ready.

    Use this for:
    - Kubernetes readiness probes
    - Load balancer health checks
    - Determining if the service can handle traffic
    """,
    responses={
        200: {
            "description": "Service is ready",
            "content": {"application/json": {"example": {"ready": True}}},
        },
        503: {
            "description": "Service is not ready",
            "content": {"application/json": {"example": {"ready": False}}},
        },
    },
)
async def readiness():
    """
    Readiness probe for Kubernetes/load balancers.

    Returns:
        Status indicating if service is ready to accept requests
    """
    ready = ml_models.models_loaded()
    status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready}


@router.get(
    "/liveness",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="""
    Kubernetes-style liveness probe endpoint.

    This endpoint always returns 200 to indicate the service is alive.
    If this endpoint fails, it indicates the service process has crashed
    and should be restarted.

    Use this for:
    - Kubernetes liveness probes
    - Process monitoring
    - Auto-restart triggers
    """,
    responses={
        200: {
            "description": "Service is alive",
            "content": {"application/json": {"example": {"alive": True}}},
        }
    },
)
async def liveness():
    """
    Liveness probe for Kubernetes.

    Returns:
        Status indicating the service is alive
    """
    return {"alive": True}
