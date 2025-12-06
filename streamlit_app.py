import requests
import streamlit as st
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="HW6 RAG over Tables", layout="centered")

st.title("📊 HW6: RAG over PDF Tables")
st.write("Backend: FastAPI + Chroma + Ollama")


def markdown_table_to_df(md: str) -> pd.DataFrame | None:
    lines = [l.strip() for l in md.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None

    header_line = lines[0].strip("| ")
    headers = [c.strip() for c in header_line.split("|")]
    if not headers:
        return None

    rows = []
    for line in lines[2:]:
        row_line = line.strip("| ")
        cols = [c.strip() for c in row_line.split("|")]
        if len(cols) != len(headers):
            continue
        rows.append(cols)

    if not rows:
        return None

    return pd.DataFrame(rows, columns=headers)


# keep last QA result across reruns
if "qa_result" not in st.session_state:
    st.session_state["qa_result"] = None


st.sidebar.header("1. Ingest PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload financial PDF", type=["pdf"])

if st.sidebar.button("Ingest PDF") and uploaded_pdf is not None:
    with st.spinner("Uploading and indexing..."):
        files = {"file": (uploaded_pdf.name, uploaded_pdf.read(), "application/pdf")}
        try:
            resp = requests.post(f"{API_BASE}/ingest", files=files)
        except Exception as e:
            st.sidebar.error(f"Request failed: {e}")
            resp = None
    if resp is not None:
        if resp.ok:
            data = resp.json()
            st.sidebar.success(
                f"Ingested! {data.get('num_documents', 0)} chunks indexed."
            )
        else:
            st.sidebar.error(f"Ingest failed: {resp.text}")

st.header("Ask a Question (RAG)")

query = st.text_input("Enter your question about the PDF:")
mode_label = st.selectbox(
    "Restrict retrieval to:",
    ["All chunks", "Tables only", "Text only"],
    index=0,
)

mode_map = {
    "All chunks": "all",
    "Tables only": "tables",
    "Text only": "text",
}
mode = mode_map[mode_label]

# when Ask is clicked, update session_state["qa_result"]
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying RAG backend..."):
            payload = {"query": query, "mode": mode}
            try:
                resp = requests.post(f"{API_BASE}/query", json=payload)
            except Exception as e:
                st.error(f"Request failed: {e}")
                resp = None

        if resp is not None:
            if resp.ok:
                st.session_state["qa_result"] = resp.json()
            else:
                st.error(f"Query failed: {resp.text}")
                st.session_state["qa_result"] = None

# always render last result (if any)
result = st.session_state.get("qa_result")
if result:
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources (from RAG)")
    seen_keys = set()
    for i, src in enumerate(result["sources"]):
        page = src.get("page")
        typ = src.get("type")
        t_idx = src.get("table_index")
        heading = src.get("heading") or "(no heading detected)"
        content = src.get("content") or src.get("preview") or ""

        key = (page, typ, t_idx, content[:50])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        with st.expander(f"Source {i+1} – page {page} ({typ})"):
            st.write(f"**Heading:** {heading}")
            st.write(f"Table index: {t_idx}")

            if typ == "table":
                df = markdown_table_to_df(content)
                if df is not None:
                    st.table(df)
                else:
                    st.code(content, language="markdown")
            else:
                st.write(content)

            # Explain button now works because result is persisted in session_state
            if st.button(f"Explain Source {i+1}", key=f"explain_{i}"):
                with st.spinner("Generating explanation..."):
                    try:
                        exp_resp = requests.post(
                            f"{API_BASE}/explain",
                            json={"content": content},
                        )
                        if exp_resp.ok:
                            exp = exp_resp.json().get("explanation", "")
                            st.markdown("### Explanation")
                            st.write(exp)
                        else:
                            st.error(f"Explain failed: {exp_resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

st.header(" Search Tables")

kw = st.text_input("Filter tables by keyword (e.g., 'iPhone', 'Products and Services'):")

if st.button("Search Tables"):
    with st.spinner("Searching tables..."):
        try:
            resp = requests.post(f"{API_BASE}/tables", json={"keyword": kw})
        except Exception as e:
            st.error(f"Request failed: {e}")
            resp = None

    if resp is not None:
        if resp.ok:
            data = resp.json()
            tables = data.get("tables", [])
            if not tables:
                st.info("No tables matched that keyword.")
            else:
                for i, tbl in enumerate(tables):
                    page = tbl.get("page")
                    t_idx = tbl.get("table_index")
                    heading = tbl.get("heading") or "(no heading detected)"
                    content = tbl.get("content") or ""

                    with st.expander(f"Table {i+1} – page {page}, index {t_idx}"):
                        st.write(f"**Heading:** {heading}")
                        df = markdown_table_to_df(content)
                        if df is not None:
                            st.table(df)
                        else:
                            st.code(content, language="markdown")
        else:
            st.error(f"Table search failed: {resp.text}")
