# Use the official Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install git-lfs for our .faiss file
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Make our run script executable
RUN chmod +x ./run.sh

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Command to run both servers
CMD ["./run.sh"]