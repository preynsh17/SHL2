import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- 1. Load Configuration ---
try:
    # Load the .env file (GOOGLE_API_KEY) from the parent directory
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env file.")
        print("Please make sure your .env file is in the root 'shl2' folder.")
        exit()

    genai.configure(api_key=api_key)

except Exception as e:
    print(f"An error occurred during setup: {e}")
    exit()

# --- 2. List Available Models ---
print("Asking Google API for a list of available models...")
print("--------------------------------------------------")

try:
    for model in genai.list_models():
        # We only care about models that can 'generateContent' (the ones we need)
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)

except Exception as e:
    print(f"\nAn error occurred while trying to list models: {e}")
    print("This might be an issue with your API key or network connection.")

print("--------------------------------------------------")
print("Please copy the output above (e.g., 'models/gemini-pro') and paste it in the chat.")
print("We will use one of these names in 'recommender.py'.")