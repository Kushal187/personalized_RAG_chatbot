import os
import re
from pathlib import Path

import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

#Config 
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

st.set_page_config(page_title="About Me RAG Chatbot", page_icon="💬", layout="wide")



# Helpers
def normalize_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_from_pdf_file(pdf_file) -> str:
    """pdf_file is an UploadedFile from Streamlit."""
    reader = PdfReader(pdf_file)
    parts = []
    for page in reader.pages:
        parts.append((page.extract_text() or "").strip())
    return "\n\n".join(p for p in parts if p)

def save_txt(filename: str, text: str):
    Path(filename).write_text(text, encoding="utf-8")



# Dynamic chunking (word-based)

def choose_dynamic_chunk_params(total_words: int):
    if total_words <= 400:
        return 200, 40
    if total_words <= 900:
        return 260, 50
    if total_words <= 1500:
        return 320, 60
    return 380, 70

def dynamic_chunk_words(text: str):
    text = normalize_text(text)
    words = text.split()
    total_words = len(words)

    chunk_words, overlap_words = choose_dynamic_chunk_params(total_words)
    step = max(1, chunk_words - overlap_words)

    chunks = []
    start = 0
    while start < total_words:
        end = min(total_words, start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    params = {"total_words": total_words, "chunk_words": chunk_words, "overlap_words": overlap_words}
    return chunks, params



# Embeddings + FAISS (cosine similarity)

def get_embedding(text: str, client: OpenAI):
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

def build_faiss_index(embeddings: np.ndarray):
    # Cosine similarity = normalize + inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query: str, index, chunks_meta, client: OpenAI, k: int = 4):
    q = np.array([get_embedding(query, client)], dtype="float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if 0 <= idx < len(chunks_meta):
            results.append(chunks_meta[idx])
    return results



# Build resources from uploads

def build_from_uploads(resume_file, personal_file, client: OpenAI):
    all_chunks_meta = []
    all_embeddings = []

    docs = [
        {"id": "resume", "file": resume_file, "txt": "resume.txt"},
        {"id": "personal", "file": personal_file, "txt": "personal.txt"},
    ]

    for d in docs:
        raw_text = extract_text_from_pdf_file(d["file"])
        text = normalize_text(raw_text)

        if not text:
            raise RuntimeError(
                f"Extracted empty text from {d['id']} PDF. If scanned/image-only, you need OCR."
            )

        # Save extracted text to .txt (you requested this)
        save_txt(d["txt"], text)

        # Dynamic chunking
        chunks, params = dynamic_chunk_words(text)

        for i, c in enumerate(chunks, start=1):
            all_chunks_meta.append({
                "doc_id": d["id"],
                "source_name": d["file"].name,
                "chunk_id": i,
                "text": c,
                "words": len(c.split()),
                "chunking": params,
            })

    # Embed all chunks
    for c in all_chunks_meta:
        all_embeddings.append(get_embedding(c["text"], client))

    X = np.array(all_embeddings, dtype="float32")
    index = build_faiss_index(X)

    return index, all_chunks_meta


def generate_answer(question: str, hits, client: OpenAI):
    context = "\n\n".join([f"[{h['doc_id']}#{h['chunk_id']}] {h['text']}" for h in hits])

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer using only the provided context. If not present, say you do not have that information."
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return resp.choices[0].message.content



# UI

st.title("💬 About Me RAG Chatbot")
st.markdown("Upload **resume + personal PDF**. The vector store is rebuilt each time you build.")

with st.sidebar:
    st.subheader("🔑 API Key")
    api_key = os.getenv("OPENAI_API_KEY")  # from .env
    if not api_key:
        st.error("Missing OPENAI_API_KEY in .env")
        st.stop()

    st.subheader("📄 Upload PDFs")
    resume_file = st.file_uploader("Upload resume.pdf", type=["pdf"], key="resume_pdf")
    personal_file = st.file_uploader("Upload personal.pdf", type=["pdf"], key="personal_pdf")

    build_btn = st.button("Build / Rebuild Vector Store", type="primary")

    st.divider()
    st.caption("This app rebuilds embeddings + FAISS on demand (no disk persistence).")


# init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks_meta" not in st.session_state:
    st.session_state.chunks_meta = None

client = OpenAI(api_key=api_key)

# Build on button click
if build_btn:
    if not resume_file or not personal_file:
        st.error("Please upload both PDFs before building.")
    else:
        with st.spinner("Extracting text, chunking dynamically, embedding, building FAISS..."):
            try:
                index, chunks_meta = build_from_uploads(resume_file, personal_file, client)
                st.session_state.index = index
                st.session_state.chunks_meta = chunks_meta

                # show chunk stats
                by_doc = {}
                for c in chunks_meta:
                    by_doc.setdefault(c["doc_id"], 0)
                    by_doc[c["doc_id"]] += 1

                st.success("✅ Vector store built successfully!")
                st.info(f"Saved extracted text to `resume.txt` and `personal.txt` in your project folder.")
                st.write("Chunks created:", by_doc)

            except Exception as e:
                st.error(str(e))


# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if q := st.chat_input("Ask about me..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    if st.session_state.index is None or st.session_state.chunks_meta is None:
        with st.chat_message("assistant"):
            st.error("Please upload both PDFs and click **Build / Rebuild Vector Store** first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                hits = retrieve(q, st.session_state.index, st.session_state.chunks_meta, client, k=5)
                ans = generate_answer(q, hits, client)
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

st.divider()
