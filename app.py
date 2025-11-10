import streamlit as st
import requests
import pandas as pd
import logging
import subprocess
import os
import atexit
import time

# --- Configuration ---
# On Hugging Face, the API will run on port 8000
API_URL = "http://127.0.0.1:8000/recommend"
HEALTH_CHECK_URL = "http://127.0.0.1:8000/health"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Start the FastAPI Backend Server ---

@st.cache_resource
def start_api_server():
    """
    Start the FastAPI server as a background subprocess.
    This is the key to running both apps on Hugging Face Spaces.
    """
    logger.info("--- Starting FastAPI server in background ---")
    
    # We use os.getenv("GOOGLE_API_KEY") to get the secret from Hugging Face
    env = os.environ.copy()
    
    # Try to get the key from Hugging Face secrets
    hf_api_key = os.getenv("GOOGLE_API_KEY")
    if hf_api_key:
        env["GOOGLE_API_KEY"] = hf_api_key
        logger.info("Loaded GOOGLE_API_KEY from Hugging Face secrets.")
    else:
        logger.warning("GOOGLE_API_KEY secret not found. API calls may fail.")
    
    # Start the server
    process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Ensure the server is shut down when Streamlit exits
    def cleanup():
        logger.info("--- Shutting down FastAPI server ---")
        process.terminate()
        process.wait()
    
    atexit.register(cleanup)
    
    logger.info("FastAPI server process started.")
    return process

start_api_server()

# --- 2. Helper Functions ---

@st.cache_data(ttl=30) # Cache for 30 seconds
def check_api_health():
    """
    Checks if the FastAPI backend is running and healthy.
    """
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        logger.warning("API connection failed. Retrying...")
        return False
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False
    return False

def get_api_recommendations(query_text):
    """
    Calls the FastAPI /recommend endpoint and returns the JSON response.
    """
    payload = {"query": query_text}
    try:
        response = requests.post(API_URL, json=payload, timeout=120) # 120 sec timeout for long queries
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json().get("detail", "Unknown API error")
            except requests.JSONDecodeError:
                error_detail = response.text
            st.error(f"API Error (Status {response.status_code}): {error_detail}")
            return None
            
    except requests.ConnectionError:
        st.error("Connection Error: Could not connect to the FastAPI backend. Is it running?")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- 3. Streamlit UI ---

st.set_page_config(page_title="SHL Recommender", layout="wide")
st.title("🤖 SHL Assessment Recommendation System")
st.markdown("Enter a natural language query or a full job description to get the 5-10 most relevant SHL assessments.")

# --- API Status Check (with retries) ---
with st.spinner("Connecting to backend API... This may take a moment on startup."):
    
    # THIS IS THE LOGIC YOU WERE MISSING
    max_retries = 180 # 3 minutes (180 * 1 second)
    retries = 0
    healthy = False
    
    while retries < max_retries:
        if check_api_health():
            healthy = True
            break
        time.sleep(1) # Wait 1 second for server to boot
        retries += 1
        
    if not healthy:
        st.error("Backend API failed to start or is unhealthy. Please check the logs.")
        st.stop()
    else:
        st.success("Backend API connected and healthy.")

# --- Input Form ---
with st.form(key="query_form"):
    query_input = st.text_area(
        "Enter your query or Job Description:",
        height=250,
        placeholder="e.g., 'I am hiring for a Senior Data Analyst with strong SQL and Python skills'"
    )
    submit_button = st.form_submit_button(label="🚀 Get Recommendations")

# --- Process and Display Results ---
if submit_button and query_input:
    with st.spinner("Processing query... This may take 20-30 seconds."):
        
        results_data = get_api_recommendations(query_input)
        
        if results_data and "recommended_assessments" in results_data:
            assessments = results_data["recommended_assessments"]
            
            if assessments:
                st.subheader(f"Top {len(assessments)} Recommendations")
                
                display_data = []
                for i, rec in enumerate(assessments):
                    display_data.append({
                        "Rank": i + 1,
                        "Assessment Name": rec.get("name"),
                        "URL": rec.get("url"),
                        "Test Type(s)": ", ".join(rec.get("test_type", [])),
                        "Duration (mins)": rec.get("duration"),
                        "Description": rec.get("description")
                    })
                
                st.dataframe(pd.DataFrame(display_data), use_container_width=True)
            else:
                st.warning("The model returned no recommendations for this query.")

elif submit_button:
    st.warning("Please enter a query or job description.") 