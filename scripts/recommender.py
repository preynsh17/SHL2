import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import regex as re
import logging

# --- 1. Configuration ---

logging.basicConfig(level=logging.ERROR)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
FAISS_INDEX_FILE = os.path.join(DATA_DIR, 'assessments.faiss')
INDEX_MAP_FILE = os.path.join(DATA_DIR, 'assessments_index_map.json')
MODEL_NAME = 'all-mpnet-base-v2' # Our best retriever model
LLM_MODEL_NAME = 'models/gemini-flash-latest'

def load_models():
    """
    Loads and returns all necessary models and data.
    """
    print("Loading configuration and models...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(LLM_MODEL_NAME)
    
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(INDEX_MAP_FILE, 'r', encoding='utf-8') as f:
            index_to_data_map = json.load(f)
            index_to_data_map = {int(k): v for k, v in index_to_data_map.items()}
    except FileNotFoundError:
        print("ERROR: Index files not found. Please run 'python scripts/build_index.py' first.")
        raise
        
    model = SentenceTransformer(MODEL_NAME)
    
    print("Models and data loaded successfully.")
    
    return {
        "llm": llm,
        "index": index,
        "index_to_data_map": index_to_data_map,
        "model": model
    }

# --- 2. Query Pre-processing Step (OUR BEST "SIMPLE EXTRACTION" PROMPT) ---

def preprocess_query(query: str, llm) -> str:
    """
    Uses the LLM to clean and summarize a noisy query
    into a keyword-focused string for better retrieval.
    """
    print(f"--- Pre-processing (Simple Extraction) ---")
    
    prompt = f"""
    You are a talent acquisition expert. Below is a long, noisy job description.
    Your task is to extract *only* the most important keywords for finding an assessment.
    Focus on job titles, technical skills, software, and personality traits.
    
    Your response MUST be a single, clean, comma-separated string.
    
    Example Input: "We're looking for a Marketing Manager... About Recro... [long text]... 5+ years of experience in B2B marketing... community building... strong storytelling..."
    Example Output: "Marketing Manager, B2B marketing, community building, storytelling"

    Job Description:
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


# --- 3. Stage 1: Retriever ---

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
    return retrieved_candidates # This is now a simple list of dicts

# --- 4. Stage 2: Re-ranker (OUR BEST "SIMPLE KEYWORD" PROMPT) ---

def stage_2_rerank(original_query: str, retrieved_candidates: list, models: dict, k_final: int):
    """
    Performs Stage 2 (Re-ranking) with our best, simple, keyword-focused prompt.
    """
    print(f"Stage 2: Sending {len(retrieved_candidates)} candidates to Gemini for re-ranking...")
    
    llm = models["llm"]
    
    if not retrieved_candidates:
        print("Warning: Stage 2 received no candidates to re-rank.")
        return []

    candidates_str = json.dumps(retrieved_candidates, indent=2)
    
    # --- OUR PROVEN, BEST-PERFORMING PROMPT ---
    prompt = f"""
    You are an expert SHL Assessment Recommendation System.
    Your task is to analyze a user's hiring query and a list of "candidate assessments".
    Your **ONLY** goal is to find the **top {k_final}** candidates from the list that are the **most direct keyword and semantic match** to the query.

    **CRITICAL RULES:**
    1.  **PRIORITIZE KEYWORDS:** Your selection *must* prioritize direct matches. If the query says "Java" or "SQL", the best results are those with "Java" or "SQL" in their name or description.
    2.  **IGNORE "BALANCE":** Do NOT try to add "balance" (e.g., personality tests) unless the query *explicitly* asks for it. Focus on direct relevance.
    3.  **FORMAT:** Your *entire* response MUST be *only* the JSON list `[]`. 
        DO NOT write any explanation, introduction, or text before or after the JSON list.
        Your response must start with `[` and end with `]`.
    4.  **CONSTRAINTS:** Pay *strict* attention to constraints like duration (e.g., "under 40 minutes"). 'duration' is in minutes.
    5.  **QUANTITY:** You must select *at most {k_final}* assessments.

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

# --- 5. Combined Function (Our Final Pipeline) ---

def get_recommendations(query: str, models: dict, k_retrieval: int, k_final: int):
    """
    Runs the full 3-step pipeline.
    """
    # 1. Pre-process (Simple Extraction)
    clean_query = preprocess_query(query, models["llm"])
    
    # 2. Retrieve candidates (Widen the net)
    retrieved_candidates = stage_1_retrieve(clean_query, models, k_retrieval)
    
    # 3. Re-rank (Simple Keyword prompt)
    final_recommendations = stage_2_rerank(
        query, 
        retrieved_candidates, 
        models, 
        k_final
    )
    
    return final_recommendations

# --- 6. Test Block ---
if __name__ == "__main__":
    models_dict = load_models()
    test_query = "I am hiring for Java developers who can also collaborate effectively."
    
    recommendations = get_recommendations(
        test_query, 
        models=models_dict, 
        k_retrieval=40, # Our wider net
        k_final=10
    )
    
    if recommendations:
        print("\n--- FINAL RECOMMENDATIONS (TEST) ---")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec['name']} (Type: {rec['test_type']})")