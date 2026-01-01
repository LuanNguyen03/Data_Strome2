"""Health check endpoint per contract"""
from fastapi import APIRouter
from backend.services.model_loader import get_model_loader

router = APIRouter()


@router.get("/healthz")
async def health_check():
    """
    GET /api/v1/healthz
    Health check endpoint (versioned)
    """
    model_loader = get_model_loader()
    return {
        "status": "healthy",
        "service": "dry-eye-assessment-api",
        "version": "v1",
        "model_version": model_loader.model_version,
        "use_ml_models": model_loader.use_ml_models,
    }
