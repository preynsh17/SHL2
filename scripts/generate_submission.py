import pandas as pd
import os
import logging
from recommender import get_recommendations, load_models

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
TEST_FILE = os.path.join(DATA_DIR, 'Gen_AI Dataset.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'predictions.csv') # Save to root folder

# Suppress warnings
logging.basicConfig(level=logging.ERROR)

def load_test_queries(file_path):
    """
    Loads the test queries from the Excel file's 'Test-Set' sheet.
    """
    try:
        df = pd.read_excel(file_path, sheet_name='Test-Set')
    except FileNotFoundError:
        print(f"ERROR: Test file not found at {file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read Excel file. {e}")
        return None
    
    # Get the list of queries, ensure they are all strings
    queries = df['Query'].astype(str).tolist()
    print(f"Loaded {len(queries)} unique queries from test set.")
    return queries

def main():
    """
    Main function to generate the final predictions.csv file.
    """
    print("--- Starting Submission File Generation ---")
    
    # 1. Load the test queries
    test_queries = load_test_queries(TEST_FILE)
    if test_queries is None:
        return
        
    # 2. Load our final, optimized model
    try:
        models_dict = load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # This will store our final submission data
    submission_data = [] # List of [query, url] pairs

    # 3. Loop through each test query
    for i, query in enumerate(test_queries):
        
        print(f"\nProcessing Test Query {i+1}/{len(test_queries)}:")
        print(f"Query: {query[:70]}...")
        
        # 4. Get recommendations from our final pipeline
        predicted_assessments = get_recommendations(
            query, 
            models=models_dict, 
            k_retrieval=40, # Our optimized k
            k_final=10
        )
        
        if not predicted_assessments:
            print("WARNING: Recommender returned no results for this query.")
            continue
            
        print(f"Got {len(predicted_assessments)} recommendations.")
        
        # 5. Format for submission (Appendix 3)
        # We need to add one row for *each* recommendation
        for rec in predicted_assessments:
            submission_data.append({
                "Query": query,
                "Assessment_url": rec['url']
            })

    # 6. Create and save the CSV file
    if submission_data:
        submission_df = pd.DataFrame(submission_data)
        
        # Save to the root 'shl2' folder
        submission_df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*30)
        print("--- SUBMISSION FILE GENERATED ---")
        print(f"Successfully saved 'predictions.csv' to your 'shl2' folder.")
        print(f"Total rows: {len(submission_df)}")
        print("="*30)
    else:
        print("No predictions were generated.")

if __name__ == "__main__":
    main()