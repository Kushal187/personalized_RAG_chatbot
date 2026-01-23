import os
import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Try importing PDF libraries - prefer pdfplumber for complex layouts
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from PyPDF2 import PdfReader

load_dotenv()

# Config
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "resumes"
MAX_CONTEXT_TURNS = 5

st.set_page_config(page_title="Multi-Resume RAG Chatbot", page_icon="📄", layout="wide")


# Data structures
@dataclass
class ResumeMetadata:
    person_name: str
    skills: list[str]
    experience_years: Optional[int]
    current_role: Optional[str]
    companies: list[str]
    education: list[str]
    summary: str
    source_file: str


@dataclass
class ChunkMetadata:
    person_name: str
    source_file: str
    chunk_id: int
    chunk_type: str  # "header", "experience", "skills", "education", "projects", "other"
    skills: str  # comma-separated for filtering
    companies: str
    experience_years: int
    text: str


# Text extraction and normalization
def normalize_text(text: str) -> str:
    """Normalize text, handling broken PDF extraction where each word is on a new line."""
    # First, check if text appears to be broken (many short lines)
    lines = text.split('\n')
    if lines:
        avg_line_len = sum(len(line.strip()) for line in lines if line.strip()) / max(len([l for l in lines if l.strip()]), 1)
        
        # If average line length is very short (< 15 chars), text is likely broken
        if avg_line_len < 15:
            # Reconstruct by joining lines intelligently
            reconstructed = []
            current_para = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Empty line might indicate paragraph break
                    if current_para:
                        reconstructed.append(' '.join(current_para))
                        current_para = []
                else:
                    current_para.append(line)
            
            if current_para:
                reconstructed.append(' '.join(current_para))
            
            text = '\n\n'.join(reconstructed)
    
    # Standard normalization
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF, trying multiple methods for best results."""
    text_pypdf2 = ""
    text_pdfplumber = ""
    
    # Method 1: PyPDF2
    try:
        pdf_file.seek(0)  # Reset file pointer
        reader = PdfReader(pdf_file)
        parts = []
        for page in reader.pages:
            parts.append((page.extract_text() or "").strip())
        text_pypdf2 = "\n\n".join(p for p in parts if p)
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {e}")
    
    # Method 2: pdfplumber (better for complex layouts)
    if HAS_PDFPLUMBER:
        try:
            pdf_file.seek(0)  # Reset file pointer
            with pdfplumber.open(pdf_file) as pdf:
                parts = []
                for page in pdf.pages:
                    parts.append((page.extract_text() or "").strip())
                text_pdfplumber = "\n\n".join(p for p in parts if p)
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}")
    
    # Choose the better extraction (longer text usually means better extraction)
    # Also check for "brokenness" - prefer text with longer average line length
    def quality_score(text):
        if not text:
            return 0
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return 0
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        return len(text) * min(avg_line_len / 50, 1)  # Penalize broken text
    
    score_pypdf2 = quality_score(text_pypdf2)
    score_pdfplumber = quality_score(text_pdfplumber)
    
    if score_pdfplumber > score_pypdf2:
        return text_pdfplumber
    return text_pypdf2


# LLM-based metadata extraction (reliable structured extraction)
def extract_resume_metadata(text: str, filename: str, client: OpenAI) -> ResumeMetadata:
    extraction_prompt = """Analyze this resume and extract structured information. 
Return ONLY valid JSON with these exact fields:
{
    "person_name": "Full name of the person",
    "skills": ["skill1", "skill2", ...],
    "experience_years": <number or null>,
    "current_role": "Most recent job title or null",
    "companies": ["company1", "company2", ...],
    "education": ["degree1 from school1", ...],
    "summary": "2-3 sentence professional summary"
}

Be precise. If information is not clearly present, use null for single values or empty arrays for lists.
Do not invent or hallucinate information not present in the resume."""

    resp = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": f"Resume text:\n\n{text[:8000]}"}  # Limit to avoid token issues
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(resp.choices[0].message.content)
        return ResumeMetadata(
            person_name=data.get("person_name", "Unknown"),
            skills=data.get("skills", []),
            experience_years=data.get("experience_years"),
            current_role=data.get("current_role"),
            companies=data.get("companies", []),
            education=data.get("education", []),
            summary=data.get("summary", ""),
            source_file=filename
        )
    except json.JSONDecodeError:
        return ResumeMetadata(
            person_name="Unknown",
            skills=[],
            experience_years=None,
            current_role=None,
            companies=[],
            education=[],
            summary="",
            source_file=filename
        )


# Semantic chunking based on resume sections
def identify_section_type(text: str) -> str:
    text_lower = text.lower()[:500]  # Check beginning of chunk
    if any(kw in text_lower for kw in ["experience", "work history", "employment", "professional background"]):
        return "experience"
    if any(kw in text_lower for kw in ["skill", "technologies", "tools", "proficiencies", "competencies", "technical"]):
        return "skills"
    if any(kw in text_lower for kw in ["education", "degree", "university", "college", "certification"]):
        return "education"
    if any(kw in text_lower for kw in ["project", "portfolio"]):
        return "projects"
    if any(kw in text_lower for kw in ["summary", "objective", "profile", "about"]):
        return "header"
    return "other"


def chunk_by_character_limit(text: str, metadata: ResumeMetadata, target_size: int = 800, overlap: int = 100) -> list[ChunkMetadata]:
    """Fallback chunking by character count with sentence boundary awareness."""
    chunks = []
    chunk_id = 0
    
    # Split into sentences (roughly)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < target_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk and len(current_chunk.strip()) > 50:  # Min chunk size
                chunk_id += 1
                chunks.append(ChunkMetadata(
                    person_name=metadata.person_name,
                    source_file=metadata.source_file,
                    chunk_id=chunk_id,
                    chunk_type=identify_section_type(current_chunk),
                    skills=",".join(metadata.skills[:10]),
                    companies=",".join(metadata.companies[:5]),
                    experience_years=metadata.experience_years or 0,
                    text=current_chunk.strip()
                ))
            # Start new chunk with overlap from end of previous
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text + " " + sentence
    
    # Don't forget the last chunk
    if current_chunk and len(current_chunk.strip()) > 50:
        chunk_id += 1
        chunks.append(ChunkMetadata(
            person_name=metadata.person_name,
            source_file=metadata.source_file,
            chunk_id=chunk_id,
            chunk_type=identify_section_type(current_chunk),
            skills=",".join(metadata.skills[:10]),
            companies=",".join(metadata.companies[:5]),
            experience_years=metadata.experience_years or 0,
            text=current_chunk.strip()
        ))
    
    return chunks


def semantic_chunk_resume(text: str, metadata: ResumeMetadata) -> list[ChunkMetadata]:
    """Chunk resume by sections, with consistent sizing."""
    text = normalize_text(text)
    
    # More specific section header pattern - requires header to be on its own line
    # Matches: "EXPERIENCE", "Work Experience:", "SKILLS & TECHNOLOGIES", etc.
    section_headers = [
        r'^\s*(EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT)',
        r'^\s*(EDUCATION|ACADEMIC)',
        r'^\s*(SKILLS|TECHNICAL SKILLS|TECHNOLOGIES|COMPETENCIES)',
        r'^\s*(PROJECTS|PERSONAL PROJECTS|ACADEMIC PROJECTS)',
        r'^\s*(SUMMARY|PROFESSIONAL SUMMARY|OBJECTIVE|PROFILE|ABOUT)',
        r'^\s*(CERTIFICATIONS?|LICENSES?|AWARDS?|HONORS?)',
        r'^\s*(PUBLICATIONS?|RESEARCH)',
    ]
    combined_pattern = '|'.join(section_headers)
    
    # Split by lines first, then identify section boundaries
    lines = text.split('\n')
    sections = []
    current_section = []
    current_header = "header"
    
    for line in lines:
        # Check if this line is a section header
        is_header = False
        if line.strip():
            # Check against known patterns
            if re.match(combined_pattern, line.strip(), re.IGNORECASE):
                is_header = True
            # Also check for standalone ALL CAPS short lines (likely headers)
            elif line.strip().isupper() and len(line.strip().split()) <= 4 and len(line.strip()) > 3:
                is_header = True
        
        if is_header and current_section:
            # Save previous section
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append((current_header, section_text))
            current_section = [line]
            current_header = identify_section_type(line)
        else:
            current_section.append(line)
    
    # Don't forget last section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            sections.append((current_header, section_text))
    
    # Now create chunks from sections
    chunks = []
    chunk_id = 0
    
    TARGET_CHUNK_SIZE = 600  # characters
    MAX_CHUNK_SIZE = 1000
    MIN_CHUNK_SIZE = 150
    
    for section_type, section_text in sections:
        # If section is small enough, keep as one chunk
        if len(section_text) <= MAX_CHUNK_SIZE:
            if len(section_text) >= MIN_CHUNK_SIZE:
                chunk_id += 1
                chunks.append(ChunkMetadata(
                    person_name=metadata.person_name,
                    source_file=metadata.source_file,
                    chunk_id=chunk_id,
                    chunk_type=section_type,
                    skills=",".join(metadata.skills[:10]),
                    companies=",".join(metadata.companies[:5]),
                    experience_years=metadata.experience_years or 0,
                    text=section_text
                ))
        else:
            # Split large sections by paragraphs or bullet points
            # Try to split on double newlines or bullet patterns
            subsections = re.split(r'\n\n+|\n(?=[\•\-\*\▪])', section_text)
            
            current_chunk = ""
            for subsection in subsections:
                subsection = subsection.strip()
                if not subsection:
                    continue
                    
                if len(current_chunk) + len(subsection) < TARGET_CHUNK_SIZE:
                    current_chunk += "\n\n" + subsection if current_chunk else subsection
                else:
                    # Save current chunk if it's substantial
                    if len(current_chunk) >= MIN_CHUNK_SIZE:
                        chunk_id += 1
                        chunks.append(ChunkMetadata(
                            person_name=metadata.person_name,
                            source_file=metadata.source_file,
                            chunk_id=chunk_id,
                            chunk_type=section_type,
                            skills=",".join(metadata.skills[:10]),
                            companies=",".join(metadata.companies[:5]),
                            experience_years=metadata.experience_years or 0,
                            text=current_chunk
                        ))
                    current_chunk = subsection
            
            # Last chunk from this section
            if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                chunk_id += 1
                chunks.append(ChunkMetadata(
                    person_name=metadata.person_name,
                    source_file=metadata.source_file,
                    chunk_id=chunk_id,
                    chunk_type=section_type,
                    skills=",".join(metadata.skills[:10]),
                    companies=",".join(metadata.companies[:5]),
                    experience_years=metadata.experience_years or 0,
                    text=current_chunk
                ))
    
    # If no chunks created (weird formatting), fall back to character-based chunking
    if not chunks:
        return chunk_by_character_limit(text, metadata)
    
    # If only 1 chunk but text is long, also fall back
    if len(chunks) == 1 and len(text) > MAX_CHUNK_SIZE * 2:
        return chunk_by_character_limit(text, metadata)
    
    return chunks


# ChromaDB setup with OpenAI embeddings
def get_chroma_client():
    return chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))


def get_openai_embedding(texts: list[str], client: OpenAI) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_collection(chunks: list[ChunkMetadata], chroma_client, openai_client: OpenAI):
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Batch embed and add
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        embeddings = get_openai_embedding(texts, openai_client)
        
        collection.add(
            ids=[f"{c.person_name}_{c.chunk_id}" for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "person_name": c.person_name,
                "source_file": c.source_file,
                "chunk_id": c.chunk_id,
                "chunk_type": c.chunk_type,
                "skills": c.skills,
                "companies": c.companies,
                "experience_years": c.experience_years
            } for c in batch]
        )
    
    return collection


# Query rewriting with context
def rewrite_query(query: str, conversation_history: list[dict], client: OpenAI) -> str:
    if not conversation_history:
        return query
    
    # Format recent conversation
    recent = conversation_history[-MAX_CONTEXT_TURNS * 2:]  # Last 5 Q&A pairs
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in recent
    ])
    
    rewrite_prompt = """You are a query rewriter for a resume search system. 
Given the conversation history and the new query, rewrite the query to be self-contained and specific.

Rules:
1. If the query references "them", "that person", "he/she" etc., replace with the actual name from context
2. If the query asks for "more details" or "what else", specify what information to look for
3. If the query is already self-contained, return it unchanged
4. Keep the rewritten query concise
5. Return ONLY the rewritten query, nothing else

Conversation history:
{history}

New query: {query}

Rewritten query:"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": rewrite_prompt.format(history=history_text, query=query)}
        ],
        temperature=0,
        max_tokens=200
    )
    
    return resp.choices[0].message.content.strip()


# Retrieval with filtering
def retrieve_with_filters(
    query: str,
    collection,
    openai_client: OpenAI,
    k: int = 8,
    person_filter: Optional[str] = None,
    skill_filter: Optional[str] = None,
    min_experience: Optional[int] = None
) -> list[dict]:
    
    query_embedding = get_openai_embedding([query], openai_client)[0]
    
    # Build where clause for filtering
    where_clauses = []
    if person_filter:
        where_clauses.append({"person_name": {"$eq": person_filter}})
    if min_experience is not None:
        where_clauses.append({"experience_years": {"$gte": min_experience}})
    
    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    
    hits = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Post-filter by skill if specified (ChromaDB doesn't support contains well)
            if skill_filter:
                if skill_filter.lower() not in meta.get("skills", "").lower():
                    continue
            
            hits.append({
                "text": doc,
                "person_name": meta.get("person_name"),
                "chunk_type": meta.get("chunk_type"),
                "source_file": meta.get("source_file"),
                "skills": meta.get("skills"),
                "distance": dist
            })
    
    return hits


# Answer generation with anti-hallucination measures
def generate_answer(
    query: str,
    hits: list[dict],
    conversation_history: list[dict],
    client: OpenAI
) -> str:
    
    if not hits:
        return "I don't have any relevant information in the uploaded resumes to answer this question."
    
    # Format context with clear source attribution
    context_parts = []
    for i, h in enumerate(hits, 1):
        context_parts.append(
            f"[Source {i}: {h['person_name']} - {h['chunk_type']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    # Format conversation history
    recent_history = conversation_history[-MAX_CONTEXT_TURNS * 2:]
    history_text = ""
    if recent_history:
        history_text = "Previous conversation:\n" + "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent_history
        ]) + "\n\n"
    
    system_prompt = """You are a helpful assistant answering questions about job candidates based on their resumes.

STRICT RULES:
1. ONLY use information explicitly stated in the provided context
2. If information is not in the context, say "I don't have that information in the resumes"
3. Always attribute information to the specific person (e.g., "John Smith has experience in...")
4. Do not infer or assume information not explicitly stated
5. If asked to compare candidates, only compare based on information present for all of them
6. Use the conversation history to maintain context but don't contradict the resume data
7. If the context is insufficient to fully answer, explain what you can answer and what's missing"""

    user_message = f"""{history_text}Resume excerpts:
{context}

Question: {query}

Answer based ONLY on the information above:"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=800
    )
    
    return resp.choices[0].message.content


# Streamlit UI
st.title("📄 Multi-Resume RAG Chatbot")
st.markdown("Upload multiple resumes to search and query candidate information.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_chroma_client()
if "collection" not in st.session_state:
    st.session_state.collection = None
if "resume_metadata" not in st.session_state:
    st.session_state.resume_metadata = {}

# Sidebar
with st.sidebar:
    st.subheader("🔑 API Key")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key", type="password")
    if not api_key:
        st.error("Please provide an OpenAI API key")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    
    st.divider()
    
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF resumes (up to 20)",
        type=["pdf"],
        accept_multiple_files=True,
        key="resume_uploader"
    )
    
    if uploaded_files and len(uploaded_files) > 20:
        st.warning("Maximum 20 files allowed. Only first 20 will be processed.")
        uploaded_files = uploaded_files[:20]
    
    build_btn = st.button("🔨 Build Vector Store", type="primary")
    
    st.divider()
    
    # Filtering options
    st.subheader("🔍 Search Filters")
    
    person_names = list(st.session_state.resume_metadata.keys())
    person_filter = st.selectbox(
        "Filter by person",
        ["All"] + person_names,
        key="person_filter"
    )
    
    skill_filter = st.text_input("Filter by skill (contains)", key="skill_filter")
    
    min_exp = st.number_input(
        "Minimum years experience",
        min_value=0,
        max_value=50,
        value=0,
        key="min_exp"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # DEBUG: Show all chunks button
    if st.button("🔍 Debug: Show All Chunks"):
        if "all_chunks" in st.session_state:
            st.write(f"Total chunks: {len(st.session_state.all_chunks)}")
            for chunk in st.session_state.all_chunks:
                st.write(f"**{chunk.person_name}** - {chunk.chunk_type}")
                st.caption(chunk.text[:300])
                st.divider()
    
    st.divider()
    
    # Show loaded resumes
    if st.session_state.resume_metadata:
        st.subheader("📋 Loaded Resumes")
        for name, meta in st.session_state.resume_metadata.items():
            with st.expander(name):
                st.write(f"**Role:** {meta.current_role or 'N/A'}")
                st.write(f"**Experience:** {meta.experience_years or 'N/A'} years")
                st.write(f"**Skills:** {', '.join(meta.skills[:5])}...")
                st.write(f"**File:** {meta.source_file}")


# Build vector store
if build_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    else:
        all_chunks = []
        progress = st.progress(0)
        status = st.status("Processing resumes...", expanded=True)
        
        for i, pdf_file in enumerate(uploaded_files):
            status.write(f"Processing: {pdf_file.name}")
            
            try:
                # Extract text
                raw_text = extract_text_from_pdf(pdf_file)
                if not raw_text.strip():
                    status.write(f"⚠️ Empty text extracted from {pdf_file.name}")
                    continue
                
                text = normalize_text(raw_text)
                
                # DEBUG: Show extracted text length
                status.write(f"   📝 Extracted {len(text)} chars, {len(text.split())} words")
                
                # Extract metadata using LLM
                metadata = extract_resume_metadata(text, pdf_file.name, client)
                st.session_state.resume_metadata[metadata.person_name] = metadata
                
                # DEBUG: Show extracted metadata
                status.write(f"   👤 Name: {metadata.person_name}")
                status.write(f"   🛠️ Skills: {metadata.skills[:5]}...")
                
                # Chunk with metadata
                chunks = semantic_chunk_resume(text, metadata)
                all_chunks.extend(chunks)
                
                # DEBUG: Show chunk details
                status.write(f"✅ {pdf_file.name}: {metadata.person_name} ({len(chunks)} chunks)")
                for j, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                    status.write(f"   Chunk {j+1} [{chunk.chunk_type}]: {len(chunk.text)} chars - '{chunk.text[:80]}...'")
                
            except Exception as e:
                status.write(f"❌ Error processing {pdf_file.name}: {str(e)}")
                import traceback
                status.write(f"   {traceback.format_exc()}")
            
            progress.progress((i + 1) / len(uploaded_files))
        
        if all_chunks:
            status.write("Building ChromaDB collection...")
            collection = build_collection(
                all_chunks,
                st.session_state.chroma_client,
                client
            )
            st.session_state.collection = collection
            
            # Store chunks for debugging
            st.session_state.all_chunks = all_chunks
            
            status.update(label="✅ Vector store built!", state="complete")
            st.success(f"Processed {len(st.session_state.resume_metadata)} resumes with {len(all_chunks)} chunks")
        else:
            st.error("No valid chunks created. Check your PDF files.")


# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about the candidates..."):
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    if st.session_state.collection is None:
        with st.chat_message("assistant"):
            st.error("Please upload resumes and build the vector store first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # Rewrite query with context
                rewritten_query = rewrite_query(
                    query,
                    st.session_state.messages[:-1],  # Exclude current query
                    client
                )
                
                # Show rewritten query if different
                if rewritten_query.lower() != query.lower():
                    st.caption(f"🔄 Searching for: {rewritten_query}")
                
                # Apply filters
                p_filter = None if person_filter == "All" else person_filter
                s_filter = skill_filter if skill_filter else None
                m_exp = min_exp if min_exp > 0 else None
                
                # Retrieve
                hits = retrieve_with_filters(
                    rewritten_query,
                    st.session_state.collection,
                    client,
                    k=8,
                    person_filter=p_filter,
                    skill_filter=s_filter,
                    min_experience=m_exp
                )
                
                # DEBUG: Show what was retrieved
                st.caption(f"🔍 Retrieved {len(hits)} chunks")
                if not hits:
                    st.warning("No chunks retrieved! Check if collection has data.")
                    # Try to debug collection
                    try:
                        count = st.session_state.collection.count()
                        st.caption(f"Collection has {count} documents")
                    except Exception as e:
                        st.caption(f"Error checking collection: {e}")
                
                # Generate answer
                answer = generate_answer(
                    query,
                    hits,
                    st.session_state.messages[:-1],
                    client
                )
                
                st.markdown(answer)
                
                # Show sources in expander
                if hits:
                    with st.expander("📚 Sources"):
                        for h in hits:
                            st.markdown(f"**{h['person_name']}** ({h['chunk_type']})")
                            st.caption(h['text'][:200] + "...")
                            st.divider()
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("💡 Tip: Use filters in the sidebar to narrow down your search to specific candidates or skills.")