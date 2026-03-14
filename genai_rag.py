# genai_rag.py
# PURPOSE: Build a RAG chatbot over patient EHR data using
#          ChromaDB (vector store) + HuggingFace embeddings.
# RUN WITH: python genai_rag.py
# THEN LAUNCH CHAT: streamlit run chatbot_app.py

import pandas as pd
import os
import json

print("=" * 60)
print("  GenAI RAG SETUP — Patient EHR Chatbot")
print("=" * 60)

# ── Install check ──────────────────────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    print("✅ ChromaDB and sentence-transformers available")
except ImportError:
    print("Installing required packages...")
    os.system("pip install chromadb sentence-transformers")
    import chromadb
    from sentence_transformers import SentenceTransformer

# ── Load data ──────────────────────────────────────────────────────────
print("\n[1] Loading patient data...")
df = pd.read_csv("data/processed/silver_clean.csv")
print(f"  Loaded {len(df):,} records")

# ── Convert rows to natural language documents ─────────────────────────
print("\n[2] Converting records to natural language documents...")

def row_to_document(row):
    """Convert a patient row to a readable text document."""
    parts = [
        f"Patient encounter ID: {row.get('encounter_id', 'N/A')}",
        f"Patient number: {row.get('patient_nbr', 'N/A')}",
        f"Demographics: Age {row.get('age', 'N/A')}, "
        f"Gender {row.get('gender', 'N/A')}, "
        f"Race {row.get('race', 'N/A')}",
        f"Hospital stay: {row.get('time_in_hospital', 'N/A')} days",
        f"Admission type ID: {row.get('admission_type_id', 'N/A')}",
        f"Primary diagnosis (ICD): {row.get('diag_1', 'N/A')}",
        f"Secondary diagnoses: {row.get('diag_2', 'N/A')}, {row.get('diag_3', 'N/A')}",
        f"Number of diagnoses: {row.get('number_diagnoses', 'N/A')}",
        f"Lab procedures: {row.get('num_lab_procedures', 'N/A')}",
        f"Medications count: {row.get('num_medications', 'N/A')}",
        f"Insulin treatment: {row.get('insulin', 'N/A')}",
        f"Diabetes medication: {row.get('diabetes_med', 'N/A')}",
        f"Prior outpatient visits: {row.get('number_outpatient', 'N/A')}",
        f"Prior emergency visits: {row.get('number_emergency', 'N/A')}",
        f"Prior inpatient visits: {row.get('number_inpatient', 'N/A')}",
        f"Readmission status: {row.get('readmitted', 'N/A')}",
        f"Readmitted within 30 days: {'Yes' if row.get('readmitted_30') == 1 else 'No'}",
    ]
    return ". ".join(parts)

# Use first 5000 records for speed (enough for demo)
df_sample = df.head(5000)
documents = [row_to_document(row) for _, row in df_sample.iterrows()]
doc_ids   = [str(row["encounter_id"]) for _, row in df_sample.iterrows()]
print(f"  Created {len(documents):,} documents")
print(f"  Sample document:\n  {documents[0][:300]}...")

# ── Build ChromaDB vector store ────────────────────────────────────────
print("\n[3] Building vector store with ChromaDB...")
os.makedirs("data/chroma_db", exist_ok=True)

client = chromadb.PersistentClient(path="data/chroma_db")

# Delete existing collection if it exists (clean rebuild)
try:
    client.delete_collection("patient_ehr")
    print("  Deleted existing collection")
except:
    pass

# Load embedding model (downloads ~80MB once, then cached)
print("  Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
print("  This may take 1-2 minutes on first run...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create collection
collection = client.create_collection(
    name="patient_ehr",
    metadata={"hnsw:space": "cosine"}
)

# Embed and insert in batches of 500
batch_size = 500
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_ids  = doc_ids[i:i+batch_size]
    embeddings = embed_model.encode(batch_docs).tolist()
    collection.add(
        documents=batch_docs,
        embeddings=embeddings,
        ids=batch_ids
    )
    print(f"  Embedded batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")

print(f"  ✅ Vector store built: {collection.count()} documents indexed")

# ── Test retrieval ─────────────────────────────────────────────────────
print("\n[4] Testing retrieval...")

def query_ehr(question: str, n_results: int = 5) -> str:
    """Query the vector store and return relevant patient records."""
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    docs = results["documents"][0]
    return "\n\n---\n\n".join(docs)

test_questions = [
    "Which patients were readmitted within 30 days and had insulin treatment?",
    "Show me elderly patients with more than 15 medications",
    "What are common diagnoses for patients with long hospital stays?"
]

for q in test_questions:
    print(f"\n  Q: {q}")
    result = query_ehr(q, n_results=2)
    print(f"  A (top result): {result[:200]}...")

print("\n✅ RAG vector store is ready!")
print("   Next: streamlit run chatbot_app.py")