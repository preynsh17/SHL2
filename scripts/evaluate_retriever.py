import pandas as pd
import os
import numpy as np
import logging
# Import the functions we need
from recommender import stage_1_retrieve, load_models, preprocess_query

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'Gen_AI Dataset.xlsx')
RETRIEVAL_K = 25 # How many docs we retrieve in stage 1

logging.basicConfig(level=logging.ERROR)

def normalize_url(url: str):
    """
    Standardizes URLs for fair comparison.
    """
    if not isinstance(url, str):
        return ""
    url = url.lower().replace("https://", "").replace("http://", "")
    url = url.replace("www.", "").replace("/solutions/", "/products/") 
    url = url.replace("/products/products/", "/products/") 
    if url.endswith('/'):
        url = url[:-1]
    return url

def load_ground_truth(file_path):
    """
    Loads and normalizes ground truth URLs from the Excel file.
    """
    try:
        df = pd.read_excel(file_path, sheet_name='Train-Set')
    except FileNotFoundError:
        print(f"ERROR: Training file not found at {file_path}")
        return None
    
    df['Normalized_URL'] = df['Assessment_url'].apply(normalize_url)
    ground_truth = df.groupby('Query')['Normalized_URL'].apply(list).to_dict()
    print(f"Loaded {len(ground_truth)} unique queries from training set.")
    return ground_truth

def calculate_retriever_recall(retrieved_urls_norm, true_urls_norm):
    """
    Calculates recall for the retriever stage.
    """
    hits = set(retrieved_urls_norm) & set(true_urls_norm)
    if len(true_urls_norm) == 0:
        return 1.0
    recall = len(hits) / len(true_urls_norm)
    return recall

def main():
    """
    Main function to evaluate the *pre-process + retriever* chain.
    """
    print(f"--- Starting Pre-processed Retriever Evaluation (Recall@{RETRIEVAL_K}) ---")
    
    ground_truth_data = load_ground_truth(TRAIN_FILE)
    if ground_truth_data is None:
        return
        
    try:
        models_dict = load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    all_retriever_recall_scores = []
    
    for i, (query, true_urls_normalized) in enumerate(ground_truth_data.items()):
        
        print(f"\nProcessing Query {i+1}/{len(ground_truth_data)}: {query[:70]}...")
        
        # --- NEW STEP ---
        # 1. Pre-process the query
        clean_query = preprocess_query(query, models_dict["llm"])
        
        # 2. Get Stage 1 retrievals using the clean query
        retrieved_assessments = stage_1_retrieve(clean_query, models=models_dict, k_retrieval=RETRIEVAL_K)
        
        # 3. Normalize the retrieved URLs
        retrieved_urls_normalized = [normalize_url(rec['url']) for rec in retrieved_assessments]
        
        # 4. Calculate retriever's recall
        recall_score = calculate_retriever_recall(retrieved_urls_normalized, true_urls_normalized)
        
        hits = len(set(retrieved_urls_normalized) & set(true_urls_normalized))
        print(f"True URLs: {len(true_urls_normalized)}, Retriever Hits: {hits}")
        print(f"Retriever Recall@{RETRIEVAL_K} for this query: {recall_score:.4f}")
        
        all_retriever_recall_scores.append(recall_score)

    if all_retriever_recall_scores:
        mean_recall = np.mean(all_retriever_recall_scores)
        print("\n" + "="*30)
        print("--- RETRIEVER EVALUATION COMPLETE ---")
        print(f"Total Queries: {len(all_retriever_recall_scores)}")
        print(f"MEAN RETRIEVER RECALL@{RETRIEVAL_K}: {mean_recall:.4f}")
        print("="*30)

if __name__ == "__main__":
    main()