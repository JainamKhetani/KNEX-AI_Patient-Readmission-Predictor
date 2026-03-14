# chatbot_app.py
# RUN WITH: streamlit run chatbot_app.py

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="EHR Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Patient EHR RAG Chatbot")
st.caption("Ask questions about patient data in natural language")

# ── Load models ────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    import chromadb
    from sentence_transformers import SentenceTransformer
    client     = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("patient_ehr")
    model      = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return collection, model

@st.cache_resource
def load_ml_model():
    import joblib
    try:
        model    = joblib.load("models/xgb_readmission.pkl")
        features = joblib.load("models/feature_cols.pkl")
        return model, features
    except:
        return None, None

try:
    collection, embed_model = load_rag()
    rag_ready = True
    st.sidebar.success(f"Vector store: {collection.count():,} records")
except Exception as e:
    rag_ready = False
    st.sidebar.error(f"Run genai_rag.py first: {e}")

ml_model, feature_cols = load_ml_model()
if ml_model:
    st.sidebar.success("ML model: loaded")

# ── Suggested questions ────────────────────────────────────────────────
st.sidebar.subheader("Suggested questions")
suggested = [
    "How many patients were readmitted within 30 days?",
    "Which patients used insulin and were readmitted?",
    "Show patients with more than 15 medications",
    "What diagnoses are most common in readmitted patients?",
    "Find elderly patients with long hospital stays",
    "Which patients had more than 3 prior emergency visits?",
]
for q in suggested:
    if st.sidebar.button(q, use_container_width=True):
        st.session_state.messages = st.session_state.get("messages", [])
        st.session_state.messages.append({"role": "user", "content": q})

# ── Chat interface ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I can answer questions about the patient readmission dataset. "
            "Try asking about readmission patterns, patient demographics, "
            "medications, diagnoses, or specific risk factors."
        )
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about patient data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching patient records..."):
            if rag_ready:
                # Get relevant records from vector store
                query_emb = embed_model.encode([prompt]).tolist()
                results   = collection.query(
                    query_embeddings=query_emb, n_results=5)
                docs = results["documents"][0]

                # Build answer from retrieved records
                answer = f"Based on {len(docs)} relevant patient records:\n\n"
                for i, doc in enumerate(docs, 1):
                    lines = doc.split(". ")
                    key_lines = [l for l in lines if any(
                        kw in l.lower() for kw in
                        ["readmit", "diagnos", "medication", "insulin",
                         "hospital stay", "age", "prior"]
                    )][:4]
                    answer += f"**Record {i}:** {'. '.join(key_lines)}\n\n"

                # Add summary stats if question is about counts
                if any(w in prompt.lower() for w in
                       ["how many", "count", "total", "number of"]):
                    df = pd.read_csv("data/processed/silver_clean.csv")
                    if "readmit" in prompt.lower():
                        n = df["readmitted_30"].sum()
                        rate = df["readmitted_30"].mean() * 100
                        answer += f"\n**Summary:** {n:,} patients "
                        f"({rate:.1f}%) were readmitted within 30 days."
                    elif "medication" in prompt.lower():
                        avg = df["num_medications"].mean()
                        answer += f"\n**Summary:** Average medications per "
                        f"patient: {avg:.1f}"
            else:
                answer = ("RAG system not ready. "
                          "Please run `python genai_rag.py` first.")

        st.write(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})

# ── ICD code lookup tool ───────────────────────────────────────────────
with st.expander("ICD Code Lookup Tool"):
    st.write("Look up diagnosis codes from the dataset")
    df_diag = pd.read_csv("data/processed/silver_clean.csv")
    top_icd = df_diag["diag_1"].value_counts().head(20).reset_index()
    top_icd.columns = ["ICD Code", "Frequency"]
    st.dataframe(top_icd, use_container_width=True, hide_index=True)