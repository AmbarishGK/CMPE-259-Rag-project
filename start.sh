#!/bin/sh
set -e

# 1) Start Ollama server
ollama serve &

# 2) Wait for server to be ready
echo "Waiting for Ollama to start..."
until curl -s http://127.0.0.1:11434/api/version >/dev/null 2>&1; do
  sleep 1
done
echo "Ollama is up."

# 3) Pull models (first run will take longer)
echo "Pulling required Ollama models (llama3, nomic-embed-text)..."
ollama pull llama3
ollama pull nomic-embed-text

# 4) Start FastAPI backend
uvicorn rag_server:app --host 0.0.0.0 --port 8000 &

# 5) Start Streamlit frontend
streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0
