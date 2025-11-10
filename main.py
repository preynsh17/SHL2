import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from scripts.recommender import load_models, get_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommendation System",
    description="An API to recommend SHL assessments based on a query or job description.",
    version="1.0.0"
)

# Define request/response models
class RecommendRequest(BaseModel):
    query: str
class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int | None
    remote_support: str
    test_type: list[str]
class RecommendResponse(BaseModel):
    recommended_assessments: list[Assessment]

# Load models at startup
try:
    logger.info("--- Starting model loading... ---")
    app.state.models = load_models()
    logger.info("--- Model loading complete. ---")
except Exception as e:
    logger.error(f"FATAL: Failed to load models on startup: {e}")
    app.state.models = None

# API Endpoints
@app.get("/health")
def health_check():
    if app.state.models:
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: RecommendRequest):
    if not app.state.models:
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    logger.info(f"Received recommendation request for query: {request.query[:50]}...")
    try:
        recommendations = get_recommendations(
            query=request.query,
            models=app.state.models,
            k_retrieval=40,
            k_final=10
        )
        return {"recommended_assessments": recommendations}
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))