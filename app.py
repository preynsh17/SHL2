import streamlit as st
import requests
import pandas as pd
import logging

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/recommend"
HEALTH_CHECK_URL = "http://127.0.0.1:8000/health"

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---

def check_api_health():
    """
    Checks if the FastAPI backend is running and healthy.
    """
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=2)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        return False
    return False

def get_api_recommendations(query_text):
    """
    Calls the FastAPI /recommend endpoint and returns the JSON response.
    """
    payload = {"query": query_text}
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            # Try to parse the error message from the API
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

# --- Streamlit UI ---

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🤖 SHL Assessment Recommendation System")
st.markdown("Enter a natural language query or a full job description to get the 5-10 most relevant SHL assessments.")

# --- API Status Check ---
with st.spinner("Connecting to backend API..."):
    if not check_api_health():
        st.error("Backend API is not running or unhealthy. Please start the FastAPI server (run `python main.py`) in a separate terminal.")
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
    with st.spinner("Processing query... This may take a moment."):
        
        # 1. Call the API
        results_data = get_api_recommendations(query_input)
        
        if results_data and "recommended_assessments" in results_data:
            assessments = results_data["recommended_assessments"]
            
            if assessments:
                st.subheader(f"Top {len(assessments)} Recommendations")
                
                # 2. Format for display (as required by assignment)
                # We create a list of dicts to pass to st.dataframe
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
                
                # 3. Display as a table
                st.dataframe(pd.DataFrame(display_data), use_container_width=True)
            else:
                st.warning("The model returned no recommendations for this query.")

elif submit_button:
    st.warning("Please enter a query or job description.")