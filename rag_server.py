import os
import tempfile
from typing import List, Optional, Tuple, Literal

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

DB_DIR = "./chroma_db"

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")

vectorstore: Optional[Chroma] = None
all_docs: List[Document] = []

app = FastAPI(title="HW6 RAG PDF/Table Server")


class QueryRequest(BaseModel):
    query: str
    mode: Literal["all", "tables", "text"] = "all"


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class TableSearchRequest(BaseModel):
    keyword: str = ""


class TableInfo(BaseModel):
    page: Optional[int]
    table_index: Optional[int]
    heading: Optional[str]
    preview: str
    content: str


class TableSearchResponse(BaseModel):
    tables: List[TableInfo]


def pdf_to_documents(pdf_path: str) -> List[Document]:
    docs: List[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1

            text = page.extract_text() or ""
            lines = text.split("\n") if text else []

            if text.strip():
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "page": page_num,
                                "chunk_index": i,
                                "type": "text",
                            },
                        )
                    )

            page_heading = ""
            for line in lines:
                if line.strip():
                    page_heading = line.strip()
                    break

            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                if not table:
                    continue

                header = table[0] or []
                rows = table[1:]
                header = [h if h is not None else "" for h in header]
                if not header:
                    continue

                md_lines = []
                md_lines.append("| " + " | ".join(header) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

                for row in rows:
                    row = row or []
                    row = [str(cell) if cell is not None else "" for cell in row]
                    if len(row) < len(header):
                        row += [""] * (len(header) - len(row))
                    else:
                        row = row[: len(header)]
                    md_lines.append("| " + " | ".join(row) + " |")

                table_markdown = "\n".join(md_lines)
                docs.append(
                    Document(
                        page_content=table_markdown,
                        metadata={
                            "page": page_num,
                            "type": "table",
                            "table_index": t_idx,
                            "heading": page_heading,
                        },
                    )
                )

    return docs


def build_vectorstore(docs: List[Document]) -> None:
    global vectorstore, all_docs

    all_docs = docs
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectorstore.persist()


def run_query(query: str, mode: str = "all") -> Optional[Tuple[str, List[dict]]]:
    global vectorstore

    if vectorstore is None:
        return None

    search_kwargs: dict = {"k": 20}
    if mode == "tables":
        search_kwargs["filter"] = {"type": "table"}
    elif mode == "text":
        search_kwargs["filter"] = {"type": "text"}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs: List[Document] = retriever.invoke(query)

    unique_docs: List[Document] = []
    seen = set()
    for d in docs:
        meta = d.metadata or {}
        key = (
            meta.get("page"),
            meta.get("type"),
            meta.get("table_index"),
            meta.get("chunk_index"),
            d.page_content[:100],
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    docs = unique_docs

    if mode == "tables" and docs:
        high_keywords = ["products", "services", "performance", "category"]
        low_keywords = ["net sales", "net", "sales", "2020", "2019", "2018"]

        scored = []
        for d in docs:
            text = d.page_content.lower()
            score = 0
            for kw in high_keywords:
                if kw in text:
                    score += 3
            for kw in low_keywords:
                if kw in text:
                    score += 1
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        docs = [d for _, d in scored]

    context_chunks = [d.page_content for d in docs]
    context = "\n\n---\n\n".join(context_chunks)[:4000]

    prompt = (
        "You are a helpful assistant answering questions based on the given context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    llm = Ollama(model=LLM_MODEL)
    answer = llm.invoke(prompt)

    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append(
            {
                "page": meta.get("page"),
                "type": meta.get("type"),
                "table_index": meta.get("table_index"),
                "chunk_index": meta.get("chunk_index"),
                "heading": meta.get("heading"),
                "preview": d.page_content[:200],
                "content": d.page_content,
            }
        )

    return answer, sources


@app.post("/tables", response_model=TableSearchResponse)
async def search_tables(req: TableSearchRequest):
    kw = (req.keyword or "").lower().strip()
    tables: List[TableInfo] = []

    for d in all_docs:
        meta = d.metadata or {}
        if meta.get("type") != "table":
            continue

        text = d.page_content or ""
        if kw and kw not in text.lower():
            continue

        tables.append(
            TableInfo(
                page=meta.get("page"),
                table_index=meta.get("table_index"),
                heading=meta.get("heading"),
                preview=text[:200],
                content=text,
            )
        )

    return TableSearchResponse(tables=tables[:20])


@app.post("/explain")
async def explain_source(body: dict):
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="No content provided for explanation.")

    prompt = (
        "Explain the following table or text in clear, simple terms. "
        "Do not guess missing information.\n\n"
        f"Content:\n{content}\n\nExplanation:"
    )

    llm = Ollama(model=LLM_MODEL)
    explanation = llm.invoke(prompt)

    return {"explanation": explanation}


@app.get("/")
def root():
    return {
        "message": "RAG PDF/Table server is running",
        "endpoints": {
            "POST /ingest": "Upload a PDF and index it",
            "POST /query": "Ask questions about the last ingested PDF",
            "POST /tables": "Browse/search tables by keyword",
            "POST /explain": "Explain a specific source chunk",
        },
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs = pdf_to_documents(tmp_path)
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No text or tables found in PDF",
            )

        build_vectorstore(docs)
        return {
            "status": "success",
            "message": "PDF ingested and indexed",
            "num_documents": len(docs),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    res = run_query(req.query, req.mode)
    if res is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF indexed yet. Upload via /ingest first.",
        )

    answer, sources = res
    return QueryResponse(answer=answer, sources=sources)
