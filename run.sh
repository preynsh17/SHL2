#!/bin/bash

# Start the FastAPI server in the background
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit app in the foreground
streamlit run app.py --server.port 8501 --server.address 0.0.0.0