# Base image: official Ollama image (includes ollama server + CLI)
FROM ollama/ollama:latest

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-full \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Work directory
WORKDIR /app

# Create and activate a virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:${PATH}"

# Copy dependency file and install Python packages into venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY rag_server.py streamlit_app.py main.py pyproject.toml uv.lock NASDAQ_AAPL_2020.pdf README.md start.sh ./
COPY chroma_db ./chroma_db

# Make sure start.sh is executable
RUN chmod +x /app/start.sh

# Environment for Ollama client
ENV OLLAMA_HOST=http://127.0.0.1:11434
ENV OLLAMA_LLM_MODEL=llama3
ENV OLLAMA_EMBED_MODEL=nomic-embed-text

# Expose ports:
# - 11434: Ollama server (optional external access)
# - 8000: FastAPI backend
# - 8501: Streamlit frontend
EXPOSE 11434
EXPOSE 8000
EXPOSE 8501

# Override the default entrypoint ("ollama") from the base image
ENTRYPOINT ["/app/start.sh"]
