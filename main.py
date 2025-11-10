import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Import our model logic from the scripts folder
from scripts.recommender import load_models, get_recommendations

# --- 1. App & Model Loading ---

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation System",
    description="An API to recommend SHL assessments based on a query or job description.",
    version="1.0.0"
)

# Define request/response models (Pydantic)
# This ensures the API input/output matches the assignment spec
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int | None # Allow for null duration
    remote_support: str
    test_type: list[str]

class RecommendResponse(BaseModel):
    recommended_assessments: list[Assessment]


# Load all models ONCE at startup
# The models are stored in the 'state' of the app
try:
    logger.info("Starting model loading...")
    app.state.models = load_models()
    logger.info("Model loading complete.")
except Exception as e:
    logger.error(f"FATAL: Failed to load models on startup: {e}")
    app.state.models = None # Set to None on failure

# --- 2. API Endpoints ---

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    [cite_start][cite: 101]
    """
    if app.state.models:
        return {"status": "healthy"}
    else:
        # If models failed to load, the app is unhealthy
        raise HTTPException(
            status_code=503, 
            detail="Models are not loaded. API is unhealthy."
        )

@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: RecommendRequest):
    """
    Main endpoint to get assessment recommendations.
    Accepts a query and returns 5-10 relevant assessments.
    [cite_start][cite: 111]
    """
    if not app.state.models:
        raise HTTPException(
            status_code=503, 
            detail="Models are not loaded. API is unavailable."
        )
        
    logger.info(f"Received recommendation request for query: {request.query[:50]}...")
    
    try:
        # 3. Call our optimized RAG pipeline
        recommendations = get_recommendations(
            query=request.query,
            models=app.state.models,
            k_retrieval=25, # Our optimized retriever count
            k_final=10        # Our final list count
        )
        
        # 4. Format the response to match the pydantic model
        # This is a crucial validation step
        formatted_response = {"recommended_assessments": recommendations}
        
        return formatted_response
    
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. Run the App ---

if __name__ == "__main__":
    # This block allows us to run the app locally for testing
    # Uvicorn is the server that runs our FastAPI code
    print("--- Starting FastAPI server locally on http://127.0.0.1:8000 ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)