import streamlit as st
import requests
import pandas as pd
import logging
import time
import os

# --- Configuration ---
# This app will run on port 8501, the API on 8000
API_BASE_URL = "http://127.0.0.1:8000" 
API_URL = f"{API_BASE_URL}/recommend"
HEALTH_CHECK_URL = f"{API_BASE_URL}/health"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
@st.cache_data(ttl=30)
def check_api_health():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def get_api_recommendations(query_text):
    payload = {"query": query_text}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error (Status {response.status_code}): {response.json().get('detail')}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="SHL Recommender", layout="wide")
st.title("🤖 SHL Assessment Recommendation System")
st.markdown("Enter a natural language query or a full job description to get the 5-10 most relevant SHL assessments.")

with st.spinner("Connecting to backend API... This may take 1-3 minutes on first load."):
    max_retries = 180
    healthy = False
    for _ in range(max_retries):
        if check_api_health():
            healthy = True
            break
        time.sleep(1)
        
    if not healthy:
        st.error("Backend API failed to start. Please check the logs.")
        st.stop()
    else:
        st.success("Backend API connected and healthy.")

with st.form(key="query_form"):
    query_input = st.text_area("Enter your query or Job Description:", height=250)
    submit_button = st.form_submit_button(label="🚀 Get Recommendations")

if submit_button and query_input:
    with st.spinner("Processing query... This may take 20-30 seconds."):
        results_data = get_api_recommendations(query_input)
        if results_data and "recommended_assessments" in results_data:
            assessments = results_data["recommended_assessments"]
            if assessments:
                st.subheader(f"Top {len(assessments)} Recommendations")
                display_data = [{"Rank": i+1, **rec} for i, rec in enumerate(assessments)]
                st.dataframe(pd.DataFrame(display_data), use_container_width=True)
            else:
                st.warning("The model returned no recommendations for this query.")
elif submit_button:
    st.warning("Please enter a query or job description.")