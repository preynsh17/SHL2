import streamlit as st
import pandas as pd
import logging
import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import regex as re
import time

# --- 1. Configuration & Model Loading ---

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Model & Data Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FAISS_INDEX_FILE = os.path.join(DATA_DIR, 'assessments.faiss')
INDEX_MAP_FILE = os.path.join(DATA_DIR, 'assessments_index_map.json')
MODEL_NAME = 'all-mpnet-base-v2' # Our best retriever model
LLM_MODEL_NAME = 'models/gemini-flash-latest'

# --- Load API Key (from .env or HF Secrets) ---
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def load_all_models():
    """
    Loads all models and data into memory.
    This is cached by Streamlit to run only once.
    """
    print("--- Loading all models... (This may take 1-2 minutes) ---")
    
    # 1. Configure Gemini API
    if not API_KEY:
        st.error("GOOGLE_API_KEY not found. Please add it to your Hugging Face secrets.")
        return None
        
    genai.configure(api_key=API_KEY)
    llm = genai.GenerativeModel(LLM_MODEL_NAME)
    
    # 2. Load FAISS Index and Map
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(INDEX_MAP_FILE, 'r', encoding='utf-8') as f:
            index_to_data_map = json.load(f)
            index_to_data_map = {int(k): v for k, v in index_to_data_map.items()}
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Could not find data files. Make sure {FAISS_INDEX_FILE} and {INDEX_MAP_FILE} are pushed to your repo.")
        return None
        
    # 3. Load Sentence Transformer Model
    model = SentenceTransformer(MODEL_NAME)
    
    print("--- Model loading complete. ---")
    
    return {
        "llm": llm,
        "index": index,
        "index_to_data_map": index_to_data_map,
        "model": model
    }

# --- 2. Our Optimized RAG Pipeline Logic (from recommender.py) ---

def preprocess_query(query: str, llm) -> str:
    """
    Uses the LLM to clean and summarize a noisy query
    into a keyword-focused string, *including constraints*.
    """
    print(f"--- Pre-processing (Simple Extraction + Constraints) ---")
    
    prompt = f"""
    You are a talent acquisition expert. Below is a user's query.
    Your task is to extract *only* the most important keywords for finding an assessment.
    Focus on:
    1. Job titles (e.g., "QA Engineer", "Sales Graduate")
    2. Technical skills (e.g., "Java", "SQL", "SEO")
    3. Personality traits (e.g., "collaboration", "cognitive")
    4. **Any constraints** (e.g., "30 minutes", "under 1 hour", "max 60 mins")
    
    Your response MUST be a single, clean, comma-separated string.
    
    Example Input: "I am new looking for new graduates in my sales team, suggest an 30 min long assessment"
    Example Output: "new graduates, sales team, 30 min"

    User Query:
    "{query}"
    
    Clean, comma-separated keyword string:
    """
    
    try:
        response = llm.generate_content(prompt)
        clean_query = response.text.strip()
        print(f"Cleaned Query: {clean_query}")
        return clean_query
    except Exception as e:
        print(f"Error in query pre-processing: {e}. Using original query.")
        return query


def stage_1_retrieve(clean_query: str, models: dict, k_retrieval: int):
    """
    Performs Stage 1 (Retrieval) and returns a simple list of candidates.
    """
    print(f"\nStage 1: Retrieving top {k_retrieval} for query: '{clean_query[:70]}...'")
    
    index = models["index"]
    model = models["model"]
    index_to_data_map = models["index_to_data_map"]
    
    query_vector = model.encode([clean_query], normalize_embeddings=True).astype('float32')
    D, I = index.search(query_vector, k_retrieval)
    
    retrieved_candidates = []
    for idx in I[0]:
        if idx in index_to_data_map:
            retrieved_candidates.append(index_to_data_map[idx])
            
    print(f"Found {len(retrieved_candidates)} candidates.")
    return retrieved_candidates


def stage_2_rerank(original_query: str, retrieved_candidates: list, models: dict, k_final: int):
    """
    Performs Stage 2 (Re-ranking) with our final, hybrid "Keyword + Balance" prompt.
    """
    print(f"Stage 2: Sending {len(retrieved_candidates)} candidates to Gemini for re-ranking...")
    
    llm = models["llm"]
    
    if not retrieved_candidates:
        print("Warning: Stage 2 received no candidates to re-rank.")
        return []

    candidates_str = json.dumps(retrieved_candidates, indent=2)
    
    prompt = f"""
    You are an expert SHL Assessment Recommendation System.
    Your task is to analyze a user's hiring query and a list of "candidate assessments".
    Your goal is to return the **top {k_final} (max)** most relevant assessments.

    **CRITICAL DECISION RULES:**
    1.  **PRIORITIZE KEYWORDS:** Your selection *must* first prioritize direct keyword matches (e.g., for "Java Developer", find "Java" tests).
    2.  **ADD BALANCE (If Needed):** *After* selecting keyword matches, check if the query *also* mentions behavioral or personality traits (e.g., "collaboration", "personality", "cognitive"). If it does, you **must** add a balanced mix of "Knowledge & Skills" AND "Personality & Behavior" / "Ability & Aptitude" assessments.
    3.  **CONSTRAINTS:** Pay *strict* attention to constraints like duration (e.g., "under 40 minutes"). 'duration' is in minutes.
    4.  **FORMAT:** Your *entire* response MUST be *only* the JSON list `[]`. 
        DO NOT write any explanation. Your response must start with `[` and end with `]`.
    5.  **QUANTITY:** You must select *at least 5* assessments and *at most {k_final}*.

    ---
    **USER QUERY (Original Version):**
    "{original_query}"
    ---
    **CANDIDATE ASSESSMENTS:**
    {candidates_str}
    ---

    **YOUR RESPONSE (Must be *only* a JSON list, starting with [):**
    """
    
    try:
        response = llm.generate_content(prompt)
        json_match = re.search(r"\[.*\]", response.text, re.DOTALL)
        
        if not json_match:
            print(f"ERROR: No JSON list found in LLM response: {response.text}")
            return []

        cleaned_response = json_match.group(0)
        final_recommendations = json.loads(cleaned_response)
        
        print(f"Stage 2: Gemini returned {len(final_recommendations)} recommendations.")
        
        if not isinstance(final_recommendations, list):
            print("Warning: LLM did not return a list. Returning empty.")
            return []
            
        return final_recommendations

    except json.JSONDecodeError:
        print(f"ERROR: Could not decode extracted JSON: {cleaned_response}")
        return []
    except Exception as e:
        print(f"ERROR: An error occurred during Gemini API call: {e}")
        return []


def get_recommendations(query: str, models: dict, k_retrieval: int, k_final: int):
    """
    Runs the full 3-step pipeline.
    """
    # 1. Pre-process
    clean_query = preprocess_query(query, models["llm"])
    
    # 2. Retrieve
    retrieved_candidates = stage_1_retrieve(clean_query, models, k_retrieval)
    
    # 3. Re-rank
    final_recommendations = stage_2_rerank(
        query, 
        retrieved_candidates, 
        models, 
        k_final
    )
    
    return final_recommendations

# --- 3. Streamlit UI ---

st.set_page_config(page_title="SHL Recommender", layout="wide")
st.title("🤖 SHL Assessment Recommendation System")
st.markdown("Enter a natural language query or a full job description to get the 5-10 most relevant SHL assessments.")

# --- Load Models ---
with st.spinner("Warming up the AI model... This may take 1-2 minutes on first load."):
    models_dict = load_all_models()
    if models_dict is None:
        st.error("Model loading failed. Please check the logs.")
        st.stop()

st.success("Model is ready! Please enter your query below.")

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
        
        # Call the function *directly*
        assessments = get_recommendations(
            query_input, 
            models=models_dict, 
            k_retrieval=40,
            k_final=10
        )
        
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