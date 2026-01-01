"""
FastAPI backend for Dry Eye Disease risk assessment
Follows medical governance specs from docs/
Strict contract compliance per docs/output_contract.md
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.api.v1 import health, assessments, olap, models
from backend.services.model_loader import get_model_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dry Eye Disease Risk Assessment API",
    description="Medical-standard 2-stage screening and triage system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup - MUST execute before serving requests"""
    logger.info("="*70)
    logger.info("STARTING UP - Loading Models & Services")
    logger.info("="*70)
    
    try:
        # Load ML models
        model_loader = get_model_loader()
        logger.info(f"Model version: {model_loader.model_version}")
        logger.info(f"Using ML models: {model_loader.use_ml_models}")
        logger.info(f"Stage A loaded: {model_loader.model_A is not None}")
        logger.info(f"Stage B loaded: {model_loader.model_B is not None}")
        
        if not model_loader.use_ml_models:
            logger.warning("Using rule-based fallback - no trained models found")
        
        # Check Gemini AI service
        from backend.services.gemini_service import GeminiService
        gemini_service = GeminiService()
        if not gemini_service.enabled:
            logger.info("💡 Tip: Set GEMINI_API_KEY to enable AI treatment recommendations")
            logger.info("   See GEMINI_SETUP.md for instructions")
        
        logger.info("="*70)
        logger.info("Startup complete - API ready")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}", exc_info=True)
        logger.warning("API will use rule-based fallback")


# Include v1 routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(assessments.router, prefix="/api/v1/assessments", tags=["assessments"])
app.include_router(olap.router, prefix="/api/v1/olap", tags=["olap"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])


@app.get("/")
async def root():
    """Root endpoint"""
    model_loader = get_model_loader()
    return {
        "message": "Dry Eye Disease Risk Assessment API",
        "version": "1.0.0",
        "api_version": "v1",
        "model_version": model_loader.model_version,
        "use_ml_models": model_loader.use_ml_models,
        "docs": "/docs",
    }
