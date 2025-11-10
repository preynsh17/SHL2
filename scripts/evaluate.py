import pandas as pd
import os
import numpy as np
import logging
from recommender import get_recommendations, load_models

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'Gen_AI Dataset.xlsx')

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


def calculate_recall_at_k(predicted_urls_normalized, true_urls_normalized, k=10):
    """
    Calculates Recall@K for a single query using normalized URLs.
    """
    predicted_top_k = predicted_urls_normalized[:k]
    hits = set(predicted_top_k) & set(true_urls_normalized)
    
    if len(true_urls_normalized) == 0:
        return 1.0
        
    recall = len(hits) / len(true_urls_normalized)
    return recall


def main():
    """
    Main function to run the FINAL, ADVANCED, optimized evaluation.
    This tests the (Expand + Retrieve@40 + Score_Re-rank@10) pipeline.
    """
    print("--- Starting FINAL Advanced Pipeline Evaluation ---")
    
    ground_truth_data = load_ground_truth(TRAIN_FILE)
    if ground_truth_data is None:
        return
        
    try:
        models_dict = load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    all_recall_scores = []
    
    for i, (query, true_urls_normalized) in enumerate(ground_truth_data.items()):
        
        print(f"\nProcessing Query {i+1}/{len(ground_truth_data)}:")
        print(f"Query: {query[:70]}...")
        
        # --- OUR FINAL MODEL PIPELINE CALL ---
        predicted_assessments = get_recommendations(
            query, 
            models=models_dict, 
            k_retrieval=40, # Widen the net
            k_final=10      # Re-rank down to 10
        )
        
        if not predicted_assessments:
            print("WARNING: Recommender returned no results for this query.")
            all_recall_scores.append(0.0)
            continue
            
        predicted_urls_normalized = [normalize_url(rec['url']) for rec in predicted_assessments]
        
        recall_score = calculate_recall_at_k(predicted_urls_normalized, true_urls_normalized, k=10)
        
        print(f"True URLs: {len(true_urls_normalized)}, Predicted URLs: {len(predicted_assessments)}")
        print(f"HITS (Normalized): {len(set(predicted_urls_normalized) & set(true_urls_normalized))}")
        print(f"Recall@10 for this query: {recall_score:.4f}")
        
        all_recall_scores.append(recall_score)

    # Calculate and print the final Mean Recall@10
    if all_recall_scores:
        mean_recall = np.mean(all_recall_scores)
        print("\n" + "="*30)
        print("--- FINAL EVALUATION COMPLETE ---")
        print(f"Total Queries: {len(all_recall_scores)}")
        print(f"FINAL MEAN RECALL@10: {mean_recall:.4f}")
        print("="*30)
    else:
        print("No queries were processed.")

if __name__ == "__main__":
    main()