import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# --- Configuration ---

# NEW: Robust path logic. This finds the script's directory
# and then builds the path to the 'data' folder from there.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, '..', 'data')

CLEAN_DATA_FILE = os.path.join(DATA_FOLDER, 'assessments_clean.json')
FAISS_INDEX_FILE = os.path.join(DATA_FOLDER, 'assessments.faiss')
INDEX_MAP_FILE = os.path.join(DATA_FOLDER, 'assessments_index_map.json')

# NEW: The smarter, larger model we are testing
MODEL_NAME = 'all-mpnet-base-v2'

def create_searchable_text(assessment):
    """
    Combines the most important fields into a single string
    for the model to embed. This improves search relevance.
    """
    test_types = " ".join(assessment.get('test_type', []))

    return f"Name: {assessment.get('name', '')} \n" \
           f"Description: {assessment.get('description', '')} \n" \
           f"Test Types: {test_types}"

def build_index():
    """
    Reads the clean assessment data, generates embeddings,
    and saves the FAISS index and a mapping file.
    """
    print(f"Loading assessment data from {CLEAN_DATA_FILE}...")
    try:
        with open(CLEAN_DATA_FILE, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {CLEAN_DATA_FILE}")
        print("Please make sure 'assessments_clean.json' is in your 'data' folder.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {CLEAN_DATA_FILE}.")
        return

    print(f"Loaded {len(assessments)} assessments.")
    
    print(f"Loading sentence transformer model '{MODEL_NAME}'...")
    # This will download the new 420MB+ model the first time you run it
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")

    corpus = []
    index_to_data_map = {}
    
    for i, assessment in enumerate(assessments):
        corpus.append(create_searchable_text(assessment))
        index_to_data_map[i] = assessment

    print(f"Creating embeddings for {len(corpus)} documents... (This may take a moment)")
    
    embeddings = model.encode(corpus, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    
    print("Embeddings created successfully.")

    dimension = embeddings.shape[1]
    index_flat = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index_flat)
    index.add_with_ids(embeddings, np.arange(len(assessments)))

    print(f"FAISS index built. Total entries: {index.ntotal}")

    print(f"Saving FAISS index to {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    print(f"Saving index-to-data map to {INDEX_MAP_FILE}...")
    with open(INDEX_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_to_data_map, f)

    print("\n--- Indexing Complete! ---")
    print(f"Index file: {FAISS_INDEX_FILE}")
    print(f"Mapping file: {INDEX_MAP_FILE}")

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    build_index()