# HW6 - RAG Over Financial PDF Tables

This project implements a Retrieval-Augmented Generation (RAG) system that extracts text and tables from financial PDFs (such as Apple’s 10-K), stores them in a vector database, and answers questions using a local LLM via Ollama. A Streamlit UI provides an interactive frontend for PDF ingestion, querying, and table browsing.

---

## 1. Features

- PDF ingestion using pdfplumber  
- Text chunking with `RecursiveCharacterTextSplitter`  
- Tables preserved as whole Markdown chunks (no splitting across rows/columns)  
- Metadata stored for each chunk:
  - page  
  - type (text/table)  
  - table_index  
  - heading (page-level section title)  
- Vector store using Chroma + Ollama embeddings  
- Retrieval modes: **all**, **tables-only**, **text-only**  
- Streamlit UI:
  - Upload PDFs  
  - Ask questions  
  - Display sources  
  - Browse/search tables  
  - Explain individual sources using the LLM  

---

## 2. Project Structure

```
rag_server.py        # FastAPI backend
streamlit_app.py     # Streamlit interface
```

Both files live in the same directory.  
There is currently **no Dockerfile** or containerization.

---

## 3. Requirements

### System
- Python 3.10+
- Ollama installed locally

### Models (pull using Ollama)
```
ollama pull llama3
ollama pull nomic-embed-text
```

### Python dependencies
Install via uv (recommended):

```
uv sync
```

Or install manually:

```
pip install fastapi uvicorn streamlit pdfplumber chromadb             langchain-core langchain-community langchain-text-splitters             requests pydantic
```

---

## 4. Running the System

### Step 1 — Start Ollama

```
ollama serve
```

### Step 2 — Start the FastAPI backend

```
uv run uvicorn rag_server:app --reload --port 8000
```

Backend API is available at:

**http://localhost:8000**

### Step 3 — Start the Streamlit UI

In a second terminal:

```
uv run streamlit run streamlit_app.py --server.port 8501
```

UI runs at:

**http://localhost:8501**

---

## 5. Usage Guide

### A. Ingest a PDF
In the Streamlit sidebar, upload a PDF such as:

```
NASDAQ_AAPL_2020.pdf
```

### B. Ask Questions
Use the main UI textbox. Example questions:

- “What were interest and dividend income in 2020 and 2019?”
- “Was interest expense higher in 2020 or 2019?”
- “Summarize net sales by product category from 2020 to 2018.”

Choose retrieval mode:
- **All chunks**
- **Tables only**
- **Text only**

### C. Inspect Sources
Each retrieved source shows:
- Page  
- Table index  
- Heading  
- Preview or rendered table  

### D. Explain Sources
Click **“Explain Source”** to have the LLM summarize and interpret that table/text chunk.

### E. Browse Tables
Use the bottom UI section to search all extracted tables by keyword:
- Example keywords: `iPhone`, `Net sales`, `Services`, `Performance`

Tables appear as clean, interactive DataFrames.

---

## 6. Notes & Limitations

- Retrieval quality depends heavily on similarity scores; similar financial tables may be confused.  
- When insufficient context is retrieved, the system answers **“I don’t know”**, which is correct grounding behavior.  
- Some tables extract imperfectly depending on PDF formatting.

---

## 7. Credits

Created as part of HW6 - Retrieval-Augmented Generation over Financial PDFs.  
Technologies: FastAPI, Chroma, Ollama, Streamlit, LangChain.

