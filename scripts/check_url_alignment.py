import pandas as pd
import json
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# Our database of scraped assessments
OUR_DB_FILE = os.path.join(DATA_DIR, 'assessments_clean.json')

# The ground truth training file
TRAIN_FILE = os.path.join(DATA_DIR, 'Gen_AI Dataset.xlsx')

def normalize_url(url: str):
    """
    A simple function to standardize URLs.
    - Removes http/httpss and www
    - Removes trailing slashes
    - Fixes /solutions/ vs /products/
    - Fixes /products/products/ duplication
    """
    if not isinstance(url, str):
        return ""
        
    url = url.lower()
    url = url.replace("https://", "").replace("http://", "")
    url = url.replace("www.", "")
    
    # --- Alignment Fixes ---
    # Fix 1: Handle /solutions/ vs /products/
    url = url.replace("/solutions/", "/products/") 
    # Fix 2: Handle the /products/products/ duplication
    url = url.replace("/products/products/", "/products/") 
    
    if url.endswith('/'):
        url = url[:-1]
    return url

def main():
    print("--- Starting URL Alignment Check ---")
    
    # 1. Load our database (assessments_clean.json)
    try:
        with open(OUR_DB_FILE, 'r', encoding='utf-8') as f:
            our_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Our database file not found at {OUR_DB_FILE}")
        return

    # Create a set of normalized URLs from our database
    our_normalized_urls = set()
    for item in our_data:
        our_normalized_urls.add(normalize_url(item['url']))
        
    print(f"Loaded {len(our_normalized_urls)} unique, normalized URLs from our database.")

    # 2. Load the ground truth (Train-Set.xlsx)
    try:
        df = pd.read_excel(TRAIN_FILE, sheet_name='Train-Set')
    except FileNotFoundError:
        print(f"ERROR: Training file not found at {TRAIN_FILE}")
        return
        
    ground_truth_urls = df['Assessment_url'].unique()
    
    # Create a set of normalized URLs from the ground truth
    gt_normalized_urls = set()
    for url in ground_truth_urls:
        gt_normalized_urls.add(normalize_url(url))
        
    print(f"Loaded {len(gt_normalized_urls)} unique, normalized URLs from ground truth.")

    # 3. Check the overlap (The "Coverage")
    
    # Find the URLs that are in the ground truth but NOT in our database
    missing_urls = gt_normalized_urls - our_normalized_urls
    
    # Find the URLs that match
    matching_urls = gt_normalized_urls.intersection(our_normalized_urls)
    
    coverage = len(matching_urls) / len(gt_normalized_urls)
    
    print("\n--- Alignment Report ---")
    print(f"Total Unique Ground Truth URLs: {len(gt_normalized_urls)}")
    print(f"Total Unique DB URLs:          {len(our_normalized_urls)}")
    print(f"Matching (Found):              {len(matching_urls)}")
    print(f"Missing (Not Found):           {len(missing_urls)}")
    print("------------------------")
    print(f"COVERAGE: {coverage:.2%}")
    print("------------------------")
    
    if len(missing_urls) > 0:
        print("\n--- MISSING URLS (Sample) ---")
        print("(These URLs are in the Train-Set but not in our database)")
        for i, url in enumerate(list(missing_urls)[:10]):
            print(f" - {url}")
            
    if coverage < 0.9:
        print("\nWARNING: Low coverage. Your Recall score will be low")
        print("because the 'correct' answers are not in your database.")
    else:
        print("\nSUCCESS: High coverage. We can now proceed.")

if __name__ == "__main__":
    main()