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
    text_lower = text.lower()[:500]
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


def extract_roles_from_experience(section_text: str) -> list[dict]:
    """Extract individual roles/jobs from an experience section."""
    roles = []
    
    # Pattern to match role entries - looks for:
    # Title | Company | Location | Date patterns
    # Common formats:
    # "Software Engineer May 2025 - Present Meta Menlo Park, CA"
    # "Software Engineer at Meta | May 2025 - Present"
    # "Software Engineer, Meta (May 2025 - Present)"
    
    # Date patterns
    date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}'
    date_range_pattern = rf'({date_pattern})\s*[-–—to]+\s*({date_pattern}|Present|Current)'
    
    # Split by date ranges (each role typically has a date range)
    # Find all date range positions
    matches = list(re.finditer(date_range_pattern, section_text, re.IGNORECASE))
    
    if not matches:
        # No date ranges found, return whole section as one chunk
        return [{"text": section_text, "company": None, "title": None}]
    
    # Extract roles based on date range positions
    for i, match in enumerate(matches):
        # Find the start of this role (either start of text or end of previous role's bullets)
        if i == 0:
            # Look backwards from date to find role title
            role_start = 0
            # Check if there's a section header before first role
            lines_before = section_text[:match.start()].split('\n')
            for j, line in enumerate(lines_before):
                if line.strip().upper() in ['EXPERIENCE', 'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT']:
                    role_start = section_text.find(lines_before[j]) + len(lines_before[j])
                    break
        else:
            # Start after previous role's content
            role_start = prev_role_end
        
        # Find end of this role (start of next role or end of section)
        if i + 1 < len(matches):
            # Find where next role's title likely starts (look for line before next date)
            next_match = matches[i + 1]
            # Search backwards from next date to find the role title line
            search_area = section_text[match.end():next_match.start()]
            
            # Look for a line that looks like a job title (not a bullet point)
            lines = search_area.split('\n')
            role_end = match.end()
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith(('•', '-', '*', '–', '▪')):
                    # This might be the next role's title
                    title_pos = section_text.find(stripped, match.end())
                    if title_pos > 0:
                        role_end = title_pos
                        break
                role_end += len(line) + 1
            
            prev_role_end = role_end
        else:
            role_end = len(section_text)
            prev_role_end = role_end
        
        role_text = section_text[role_start:role_end].strip()
        
        # Try to extract company name from the role text
        company = None
        # Look for company names in metadata.companies or extract from text
        first_line = role_text.split('\n')[0] if role_text else ""
        
        roles.append({
            "text": role_text,
            "company": company,
            "title": first_line[:100] if first_line else None,
            "date_range": match.group(0)
        })
    
    return roles


def extract_entries_from_section(section_text: str, section_type: str) -> list[str]:
    """Extract individual entries (roles, projects, education) from a section."""
    
    if section_type == "experience":
        roles = extract_roles_from_experience(section_text)
        return [r["text"] for r in roles if r["text"].strip()]
    
    elif section_type == "projects":
        # Split projects by project headers (usually Name | Tech Stack | Date)
        # Look for lines that start a new project (not bullet points)
        entries = []
        current_entry = []
        lines = section_text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            # Detect project header: not a bullet, contains | or tech keywords
            is_project_header = (
                stripped and 
                not stripped.startswith(('•', '-', '*', '–', '▪')) and
                (
                    '|' in stripped or 
                    re.search(r'(?:Python|Java|React|Node|Flask|Django|AWS|Docker)', stripped, re.IGNORECASE)
                ) and
                len(stripped.split()) >= 2
            )
            
            if is_project_header and current_entry:
                entries.append('\n'.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        # Filter out section header if it's the only "entry"
        entries = [e.strip() for e in entries if e.strip() and len(e.strip()) > 20]
        return entries if entries else [section_text]
    
    elif section_type == "education":
        # Split by university/school names or degree entries
        entries = []
        current_entry = []
        lines = section_text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            # Detect education header: contains university/college keywords or degree
            is_edu_header = (
                stripped and
                not stripped.startswith(('•', '-', '*', '–', '▪')) and
                (
                    re.search(r'(?:University|College|Institute|School|Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.)', stripped, re.IGNORECASE)
                )
            )
            
            if is_edu_header and current_entry and any(c.strip() for c in current_entry):
                entries.append('\n'.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        entries = [e.strip() for e in entries if e.strip() and len(e.strip()) > 20]
        return entries if entries else [section_text]
    
    # For other sections (skills, summary, etc.), keep as single chunk
    return [section_text]


def semantic_chunk_resume(text: str, metadata: ResumeMetadata) -> list[ChunkMetadata]:
    """Chunk resume by sections and individual entries (roles, projects, etc.)."""
    text = normalize_text(text)
    
    # Section header patterns
    section_headers = [
        r'^\s*(EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY)',
        r'^\s*(EDUCATION|ACADEMIC BACKGROUND)',
        r'^\s*(SKILLS|TECHNICAL SKILLS|TECHNOLOGIES|CORE COMPETENCIES)',
        r'^\s*(PROJECTS|PERSONAL PROJECTS|ACADEMIC PROJECTS|SIDE PROJECTS)',
        r'^\s*(SUMMARY|PROFESSIONAL SUMMARY|OBJECTIVE|PROFILE|ABOUT ME)',
        r'^\s*(CERTIFICATIONS?|LICENSES?|AWARDS?|HONORS?|ACHIEVEMENTS?)',
        r'^\s*(PUBLICATIONS?|RESEARCH|PAPERS)',
        r'^\s*(LEADERSHIP|ACTIVITIES|EXTRACURRICULAR)',
    ]
    combined_pattern = '|'.join(section_headers)
    
    # Parse into sections
    lines = text.split('\n')
    sections = []
    current_section_lines = []
    current_section_type = "header"
    
    for line in lines:
        stripped = line.strip()
        is_section_header = False
        
        if stripped:
            if re.match(combined_pattern, stripped, re.IGNORECASE):
                is_section_header = True
            elif stripped.isupper() and 2 <= len(stripped.split()) <= 5 and len(stripped) > 3:
                # Likely a section header
                is_section_header = True
        
        if is_section_header:
            # Save previous section
            if current_section_lines:
                section_text = '\n'.join(current_section_lines).strip()
                if section_text:
                    sections.append((current_section_type, section_text))
            current_section_lines = [line]
            current_section_type = identify_section_type(stripped)
        else:
            current_section_lines.append(line)
    
    # Save last section
    if current_section_lines:
        section_text = '\n'.join(current_section_lines).strip()
        if section_text:
            sections.append((current_section_type, section_text))
    
    # Create chunks from sections
    chunks = []
    chunk_id = 0
    
    for section_type, section_text in sections:
        # Extract individual entries from the section
        entries = extract_entries_from_section(section_text, section_type)
        
        for entry in entries:
            entry = entry.strip()
            if not entry or len(entry) < 50:
                continue
            
            # If entry is still too long (>1500 chars), split by sentences
            if len(entry) > 1500:
                sentences = re.split(r'(?<=[.!?])\s+', entry)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 1200:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunk_id += 1
                            chunks.append(ChunkMetadata(
                                person_name=metadata.person_name,
                                source_file=metadata.source_file,
                                chunk_id=chunk_id,
                                chunk_type=section_type,
                                skills=",".join(metadata.skills[:10]),
                                companies=",".join(metadata.companies[:5]),
                                experience_years=metadata.experience_years or 0,
                                text=current_chunk.strip()
                            ))
                        current_chunk = sentence
                if current_chunk:
                    chunk_id += 1
                    chunks.append(ChunkMetadata(
                        person_name=metadata.person_name,
                        source_file=metadata.source_file,
                        chunk_id=chunk_id,
                        chunk_type=section_type,
                        skills=",".join(metadata.skills[:10]),
                        companies=",".join(metadata.companies[:5]),
                        experience_years=metadata.experience_years or 0,
                        text=current_chunk.strip()
                    ))
            else:
                chunk_id += 1
                chunks.append(ChunkMetadata(
                    person_name=metadata.person_name,
                    source_file=metadata.source_file,
                    chunk_id=chunk_id,
                    chunk_type=section_type,
                    skills=",".join(metadata.skills[:10]),
                    companies=",".join(metadata.companies[:5]),
                    experience_years=metadata.experience_years or 0,
                    text=entry
                ))
    
    # Fallback if no chunks created
    if not chunks:
        chunk_id += 1
        chunks.append(ChunkMetadata(
            person_name=metadata.person_name,
            source_file=metadata.source_file,
            chunk_id=chunk_id,
            chunk_type="other",
            skills=",".join(metadata.skills[:10]),
            companies=",".join(metadata.companies[:5]),
            experience_years=metadata.experience_years or 0,
            text=text[:2000]  # First 2000 chars as fallback
        ))
    
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
def rewrite_query(query: str, conversation_history: list[dict], known_names: list[str], client: OpenAI) -> tuple[str, Optional[str], bool]:
    """
    Rewrite query with context and extract person name if mentioned.
    Returns: (rewritten_query, detected_person_name or None, is_multi_person_query)
    """
    
    # First, check if the query explicitly mentions a known person
    # If so, DON'T let the rewriter change it
    explicit_person, is_multi = detect_person_in_query(query, known_names)
    
    if explicit_person:
        # Query already has an explicit name - don't rewrite the person
        return query, explicit_person, False
    
    if is_multi:
        # Multiple people mentioned or multi-person keywords
        return query, None, True
    
    # No explicit person in query - check if we need to resolve pronouns/references
    pronouns_and_refs = ['he', 'she', 'his', 'her', 'him', 'they', 'their', 'them', 
                         'that person', 'that candidate', 'there', 'that company', 
                         'that role', 'that job', 'this person', 'this candidate']
    query_lower = query.lower()
    has_reference = any(f' {p} ' in f' {query_lower} ' or query_lower.startswith(f'{p} ') or query_lower.endswith(f' {p}') for p in pronouns_and_refs)
    
    if not has_reference and not conversation_history:
        # No pronouns and no history - return as is
        return query, None, False
    
    if not conversation_history:
        return query, None, False
    
    # Format recent conversation
    recent = conversation_history[-MAX_CONTEXT_TURNS * 2:]  # Last 5 Q&A pairs
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in recent
    ])
    
    rewrite_prompt = """You are a query rewriter for a resume search system. 
Given the conversation history and the new query, rewrite the query to be FULLY self-contained.

CRITICAL RULES:
1. Replace ALL pronouns (he/she/his/her/they/their/them) with actual names from conversation history
2. Replace ALL contextual references like "there", "that company", "that role", "it" with the actual entity from context
3. If the query already contains a person's name, DO NOT change it to a different person
4. If the query asks about "their", "them", "all", "everyone", or "candidates", it's asking about multiple people
5. The rewritten query must be understandable WITHOUT any conversation history

Known candidates: {known_names}

Return JSON with exactly these fields:
{{
  "rewritten_query": "the fully self-contained rewritten query",
  "person_name": "full name if query is about a SINGLE specific person, otherwise null",
  "is_multi_person": true/false (true if query asks about multiple people)
}}

Examples:
- "what did he do there" after discussing John at Google → "what did John do at Google"
- "tell me more about his skills" after discussing Jane → "tell me more about Jane's skills"
- "what is their experience" → is_multi_person: true

Conversation history:
{history}

New query: {query}

JSON response:"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": rewrite_prompt.format(
                history=history_text, 
                query=query,
                known_names=", ".join(known_names)
            )}
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(resp.choices[0].message.content)
        rewritten = result.get("rewritten_query", query)
        person = result.get("person_name")
        is_multi = result.get("is_multi_person", False)
        
        # Safety check: if original query mentioned a name, don't let rewriter change it
        original_person, _ = detect_person_in_query(query, known_names)
        if original_person and person != original_person:
            # Rewriter tried to change the person - override
            return query, original_person, False
        
        return rewritten, person, is_multi
    except json.JSONDecodeError:
        return query, None, False


def detect_person_in_query(query: str, known_names: list[str]) -> tuple[Optional[str], bool]:
    """
    Check if query mentions known person names.
    Returns: (single_person_name or None, is_multi_person)
    """
    query_lower = query.lower()
    
    # Check for multi-person indicators
    multi_person_keywords = ['their', 'them', 'all', 'everyone', 'candidates', 'compare', 'both', 'each']
    is_multi = any(f' {kw} ' in f' {query_lower} ' or query_lower.startswith(f'{kw} ') for kw in multi_person_keywords)
    
    # Count how many people are mentioned
    mentioned_people = []
    for name in known_names:
        name_lower = name.lower()
        # Check full name
        if name_lower in query_lower:
            mentioned_people.append(name)
            continue
        # Check individual parts of name (first name, last name)
        name_parts = name.split()
        for part in name_parts:
            # Need at least 3 chars and must be a whole word
            if len(part) > 2:
                # Check as whole word
                pattern = rf'\b{re.escape(part.lower())}\b'
                if re.search(pattern, query_lower):
                    if name not in mentioned_people:
                        mentioned_people.append(name)
                    break
    
    # If multiple people mentioned, it's a multi-person query
    if len(mentioned_people) > 1:
        return None, True
    
    # If multi-person keywords present, don't filter even if one name mentioned
    if is_multi:
        return None, True
    
    # Single person mentioned
    if len(mentioned_people) == 1:
        return mentioned_people[0], False
    
    return None, False


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
    
    # Retrieve more results initially to allow for post-filtering
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k * 2, 20),  # Get more to filter from
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
            # Post-filter by skill if specified
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
            
            if len(hits) >= k:
                break
    
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
                st.write(f"**{chunk.person_name}** - {chunk.chunk_type} ({len(chunk.text)} chars)")
                st.caption(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
                st.divider()
    
    # DEBUG: Search specific text in chunks
    debug_search = st.text_input("🔎 Debug: Search in chunks", key="debug_search")
    if debug_search and "all_chunks" in st.session_state:
        matches = [c for c in st.session_state.all_chunks if debug_search.lower() in c.text.lower()]
        st.write(f"Found {len(matches)} chunks containing '{debug_search}':")
        for chunk in matches:
            st.write(f"**{chunk.person_name}** - {chunk.chunk_type}")
            # Highlight the search term
            st.caption(chunk.text[:500])
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
                # Get known names first
                known_names = list(st.session_state.resume_metadata.keys())
                
                # Rewrite query with context
                rewritten_query, detected_person, is_multi_person = rewrite_query(
                    query,
                    st.session_state.messages[:-1],  # Exclude current query
                    known_names,
                    client
                )
                
                # Show rewritten query if different
                if rewritten_query.lower() != query.lower():
                    st.caption(f"🔄 Searching for: {rewritten_query}")
                
                if is_multi_person:
                    # Multi-person query - don't filter
                    p_filter = None
                    st.caption(f"👥 Searching across all {len(known_names)} candidates")
                elif detected_person and detected_person in known_names:
                    # Single person detected
                    p_filter = detected_person
                    st.caption(f"👤 Filtering for: {detected_person}")
                elif person_filter != "All":
                    p_filter = person_filter
                else:
                    p_filter = None
                
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
                            st.caption(h['text'][:300] + "..." if len(h['text']) > 300 else h['text'])
                            st.divider()
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("💡 Tip: Use filters in the sidebar to narrow down your search to specific candidates or skills.")