import streamlit as st
import os

from rag_pipeline.loader import load_documents
from rag_pipeline.chunking import split_documents
from rag_pipeline.embeddings import create_vector_store
from rag_pipeline.retriever import get_retriever
from rag_pipeline.qa_chain import generate_answer
from comparison_engine.comparator import compare_documents
from summary_engine.summarizer import summarize_documents

# Page config
st.set_page_config(page_title="Enterprise Knowledge AI", layout="wide")

# Title
st.title("📊 Enterprise Knowledge AI Assistant")

st.markdown("""
Ask questions from your documents and get **accurate answers with sources**.

- Multi-format support (PDF, TXT, CSV, Excel)  
- Explainable AI (with citations)  
- Comparison engine (advanced analysis)  
- Summary engine (multi-document insights)  
- No hallucination  
""")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload your documents",
    type=["pdf", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)

# -------------------------------
# PROCESS FILES
# -------------------------------
retriever = None

if uploaded_files:

    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_paths.append(file_path)

    docs = load_documents(file_paths)
    chunks = split_documents(docs)
    vectorstore = create_vector_store(chunks)
    retriever = get_retriever(vectorstore)

    st.success("Documents processed successfully ✅")

# -------------------------------
# INPUT UI
# -------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("🔍 Ask a question:")

with col2:
    st.write("")
    st.write("")
    search_clicked = st.button("Search")

# -------------------------------
# QUERY EXECUTION
# -------------------------------
if retriever and query and search_clicked:

    results = retriever.invoke(query)

    query_lower = query.lower()

    # -------------------------------
    # MODE SELECTION
    # -------------------------------
    if "compare" in query_lower:
        answer = compare_documents(query, results)
        sources = []

    elif "summarize" in query_lower or "summary" in query_lower:
        answer = summarize_documents(query, results)
        sources = []

    else:
        answer, sources = generate_answer(query, results)

    # Clean formatting
    formatted_answer = answer.replace(" - ", "\n- ").replace("- ", "\n- ")

    # -------------------------------
    # ANSWER
    # -------------------------------
    st.subheader("📌 Answer")

    st.markdown(f"""
    <div style="
        padding:16px;
        border-radius:10px;
        background-color:#f9fafb;
        border:1px solid #e5e7eb;
        font-size:15px;
        line-height:1.6;
    ">
    {formatted_answer}
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # SOURCES (only QA mode)
    # -------------------------------
    if sources and "I don't know" not in answer:
        st.subheader("📚 Sources")

        for i, src in enumerate(sources):
            file_name = os.path.basename(src['source'])

            st.markdown(f"""
            <div style="
                padding:10px;
                border-radius:8px;
                background-color:#ffffff;
                border:1px solid #e5e7eb;
                margin-bottom:10px;
                font-size:13px;
                color:#374151;
            ">
            <b>Source {i+1}:</b> {file_name} <br>
            <i>{src['content']}</i>
            </div>
            """, unsafe_allow_html=True)

elif search_clicked and not uploaded_files:
    st.warning("Please upload at least one document first ⚠️")