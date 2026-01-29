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


import re
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_SIZE_MB = 2
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
GUARDRAIL_MODEL = "gpt-4o-mini"  # Cheap model for guardrails

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    severity: str = "info"  # "info", "warning", "error"


# ============================================================
# FILE-LEVEL GUARDRAILS
# ============================================================

def check_file_size(file) -> GuardrailResult:
    """Check if file is within size limits."""
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        return GuardrailResult(
            passed=False,
            reason=f"File size ({size_mb:.1f}MB) exceeds maximum allowed ({MAX_FILE_SIZE_MB}MB)",
            severity="error"
        )
    return GuardrailResult(passed=True)


def check_file_type(file) -> GuardrailResult:
    """Verify file is actually a PDF by checking magic bytes."""
    file.seek(0)
    header = file.read(8)
    file.seek(0)
    
    # PDF magic bytes: %PDF
    if not header.startswith(b'%PDF'):
        return GuardrailResult(
            passed=False,
            reason="File does not appear to be a valid PDF (invalid header)",
            severity="error"
        )
    return GuardrailResult(passed=True)


# ============================================================
# CONTENT-LEVEL GUARDRAILS
# ============================================================

def detect_prompt_injection(text: str) -> GuardrailResult:
    """Detect potential prompt injection attempts in document text."""
    
    # Patterns that indicate prompt injection attempts
    injection_patterns = [
        # Direct instruction overrides
        r'ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)',
        r'disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)',
        r'forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)',
        r'override\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)',
        
        # Role-play injections
        r'you\s+are\s+now\s+(a|an|the)',
        r'act\s+as\s+(a|an|if)',
        r'pretend\s+(to\s+be|you\s+are)',
        r'roleplay\s+as',
        r'from\s+now\s+on\s+you',
        
        # System prompt extraction
        r'(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system\s+)?prompt',
        r'what\s+(are|is)\s+your\s+(system\s+)?(instructions?|prompts?|rules?)',
        r'repeat\s+(your|the)\s+(system\s+)?(instructions?|prompts?)',
        
        # Jailbreak attempts
        r'(DAN|STAN|DUDE)\s*mode',
        r'developer\s+mode',
        r'jailbreak',
        r'bypass\s+(your\s+)?(restrictions?|filters?|rules?)',
        
        # Command injection
        r'\[INST\]|\[\/INST\]',
        r'<\|im_start\|>|<\|im_end\|>',
        r'###\s*(Human|Assistant|System):',
        r'<\|system\|>|<\|user\|>|<\|assistant\|>',
        
        # Hidden instructions (often in white text)
        r'hidden\s+instructions?',
        r'secret\s+instructions?',
        r'do\s+not\s+tell\s+(the\s+)?user',
        r'when\s+(answering|responding)',
        
        # Behavioral override attempts
        r'no\s+matter\s+what\s+(question|query|prompt)',
        r'always\s+(say|respond|answer|reply)\s+that',
        r'for\s+(any|every|all)\s+(question|query|prompt)',
        r'regardless\s+of\s+(the\s+)?(question|query|input)',
        r'whatever\s+(the\s+)?(user|question|query)\s+(asks?|says?)',
    ]
    
    text_lower = text.lower()
    
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return GuardrailResult(
                passed=False,
                reason=f"Potential prompt injection detected in document",
                severity="error"
            )
    
    return GuardrailResult(passed=True)


def detect_suspicious_formatting(text: str) -> GuardrailResult:
    """Detect suspicious formatting that might indicate hidden content."""
    
    issues = []
    
    # Check for excessive whitespace (might hide text)
    whitespace_ratio = text.count(' ') / max(len(text), 1)
    if whitespace_ratio > 0.5:
        issues.append("Excessive whitespace detected")
    
    # Check for unusual Unicode characters (often used to hide text)
    suspicious_unicode = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\ufeff',  # Zero-width no-break space
        '\u00a0',  # Non-breaking space (excessive use)
    ]
    
    unicode_count = sum(text.count(char) for char in suspicious_unicode)
    if unicode_count > 10:
        issues.append(f"Suspicious Unicode characters detected ({unicode_count} instances)")
    
    # Check for very long lines without breaks (might be obfuscated)
    lines = text.split('\n')
    for line in lines:
        if len(line) > 5000:
            issues.append("Unusually long text lines detected")
            break
    
    if issues:
        return GuardrailResult(
            passed=False,
            reason="; ".join(issues),
            severity="warning"
        )
    
    return GuardrailResult(passed=True)


def validate_resume_content(text: str, client: OpenAI) -> GuardrailResult:
    """Use LLM to verify the document appears to be a legitimate resume."""
    
    validation_prompt = """Analyze this document text and determine if it appears to be a legitimate resume/CV.

A legitimate resume typically contains:
- A person's name
- Contact information (email, phone, location)
- Work experience or education
- Skills or qualifications

Respond with JSON:
{
    "is_resume": true/false,
    "confidence": "high"/"medium"/"low",
    "reason": "brief explanation",
    "detected_content_type": "resume"/"job_posting"/"article"/"spam"/"other"
}

Document text (first 3000 chars):
"""
    
    try:
        resp = client.chat.completions.create(
            model=GUARDRAIL_MODEL,
            messages=[
                {"role": "user", "content": validation_prompt + text[:3000]}
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(resp.choices[0].message.content)
        
        if not result.get("is_resume", False):
            content_type = result.get("detected_content_type", "unknown")
            reason = result.get("reason", "Document does not appear to be a resume")
            return GuardrailResult(
                passed=False,
                reason=f"Document appears to be '{content_type}': {reason}",
                severity="error"  # CHANGED FROM "warning" TO "error"
            )
        
        # Low confidence warning (still allow, but warn)
        if result.get("confidence") == "low":
            return GuardrailResult(
                passed=True,
                reason="Document may not be a standard resume format",
                severity="warning"
            )
        
        return GuardrailResult(passed=True)
        
    except Exception as e:
        # If validation fails, allow but warn
        return GuardrailResult(
            passed=True,
            reason=f"Could not validate document type: {str(e)}",
            severity="warning"
        )


# ============================================================
# QUERY-LEVEL GUARDRAILS
# ============================================================

def check_query_injection(query: str) -> GuardrailResult:
    """Check if user query contains prompt injection attempts."""
    
    injection_patterns = [
        r'ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)',
        r'system\s*prompt',
        r'you\s+are\s+now',
        r'act\s+as\s+if',
        r'pretend\s+(to\s+be|you)',
        r'\[INST\]',
        r'<\|',
        r'###\s*(Human|System|Assistant)',
    ]
    
    query_lower = query.lower()
    
    for pattern in injection_patterns:
        if re.search(pattern, query_lower):
            return GuardrailResult(
                passed=False,
                reason="Query contains suspicious patterns",
                severity="warning"
            )
    
    return GuardrailResult(passed=True)


def check_query_relevance(query: str, client: OpenAI) -> GuardrailResult:
    """Check if query is relevant to resume/candidate information."""
    
    # Quick keyword check first (avoid LLM call for obvious cases)
    resume_keywords = [
        'experience', 'skill', 'education', 'work', 'job', 'role', 'company',
        'project', 'resume', 'candidate', 'qualification', 'degree', 'university',
        'intern', 'position', 'team', 'manage', 'develop', 'build', 'create',
        'year', 'month', 'doing', 'did', 'where', 'what', 'who', 'which',
        'compare', 'list', 'show', 'tell', 'summary', 'background', 'history'
    ]
    
    query_lower = query.lower()
    if any(kw in query_lower for kw in resume_keywords):
        return GuardrailResult(passed=True)
    
    # For ambiguous queries, use LLM
    relevance_prompt = """Is this query asking about resume/CV information, job candidates, or professional backgrounds?

Valid queries include:
- Questions about work experience, skills, education
- Questions about specific candidates
- Comparisons between candidates
- Questions about projects, roles, companies
- Meta questions about loaded resumes

Invalid queries include:
- General knowledge questions unrelated to resumes
- Requests to write code, stories, or other content
- Questions about current events, news, etc.
- Personal advice unrelated to job candidates

Query: "{query}"

Respond with JSON: {{"is_relevant": true/false, "reason": "brief explanation"}}"""

    try:
        resp = client.chat.completions.create(
            model=GUARDRAIL_MODEL,
            messages=[{"role": "user", "content": relevance_prompt.format(query=query)}],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(resp.choices[0].message.content)
        
        if not result.get("is_relevant", True):
            return GuardrailResult(
                passed=False,
                reason=result.get("reason", "Query does not appear to be about resume information"),
                severity="info"
            )
        
        return GuardrailResult(passed=True)
        
    except Exception:
        # If check fails, allow the query
        return GuardrailResult(passed=True)


# ============================================================
# RESPONSE-LEVEL GUARDRAILS
# ============================================================

def check_response_safety(response: str) -> GuardrailResult:
    """Check if the generated response is safe and appropriate."""
    
    # Check for potential data leakage patterns
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
        r'\b\d{16}\b',  # Credit card number
        r'password\s*[:=]\s*\S+',  # Password exposure
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                reason="Response may contain sensitive information",
                severity="error"
            )
    
    return GuardrailResult(passed=True)


# ============================================================
# MAIN GUARDRAIL FUNCTIONS (USE THESE IN YOUR APP)
# ============================================================

def run_file_guardrails(file, client: OpenAI) -> tuple[bool, list[str], list[str]]:
    """
    Run all file-level guardrails.
    Returns: (passed, errors, warnings)
    """
    errors = []
    warnings = []
    
    # 1. File size check
    result = check_file_size(file)
    if not result.passed:
        errors.append(result.reason)
    
    # 2. File type check
    result = check_file_type(file)
    if not result.passed:
        errors.append(result.reason)
    
    return len(errors) == 0, errors, warnings


def run_content_guardrails(text: str, client: OpenAI) -> tuple[bool, list[str], list[str]]:
    """
    Run all content-level guardrails after text extraction.
    Returns: (passed, errors, warnings)
    """
    errors = []
    warnings = []
    
    # 1. Prompt injection detection
    result = detect_prompt_injection(text)
    if not result.passed:
        if result.severity == "error":
            errors.append(result.reason)
        else:
            warnings.append(result.reason)
    
    # 2. Suspicious formatting detection
    result = detect_suspicious_formatting(text)
    if not result.passed:
        if result.severity == "error":
            errors.append(result.reason)
        else:
            warnings.append(result.reason)
    
    # 3. Resume content validation
    result = validate_resume_content(text, client)
    if not result.passed:
        if result.severity == "error":
            errors.append(result.reason)
        else:
            warnings.append(result.reason)
    elif result.reason:  # Passed but with warning
        warnings.append(result.reason)
    
    return len(errors) == 0, errors, warnings


def run_query_guardrails(query: str, client: OpenAI) -> tuple[bool, Optional[str]]:
    """
    Run all query-level guardrails.
    Returns: (passed, error_message)
    """
    # 1. Injection check
    result = check_query_injection(query)
    if not result.passed:
        return False, "I can't process that query. Please ask a question about the candidates."
    
    # 2. Relevance check
    result = check_query_relevance(query, client)
    if not result.passed:
        return False, "I'm designed to answer questions about job candidates and their resumes. Please ask something related to the uploaded resumes."
    
    return True, None


def sanitize_text_for_llm(text: str) -> str:
    """
    Sanitize extracted text before sending to LLM.
    Removes or neutralizes potential injection attempts.
    """
    # Remove zero-width characters
    zero_width_chars = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff']
    for char in zero_width_chars:
        text = text.replace(char, '')
    
    # Normalize excessive whitespace
    text = re.sub(r'[ \t]{10,}', ' ', text)
    
    # Remove potential instruction markers
    instruction_markers = [
        r'\[INST\].*?\[/INST\]',
        r'<\|im_start\|>.*?<\|im_end\|>',
        r'###\s*(Human|Assistant|System):.*?(?=###|$)',
    ]
    
    for pattern in instruction_markers:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text

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
ANSWER_MODEL = "gpt-4o"
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
        
        # Handle None or empty person_name
        person_name = data.get("person_name")
        if person_name is None or person_name == "null" or str(person_name).strip() == "":
            person_name = "Unknown"
        
        return ResumeMetadata(
            person_name=person_name,
            skills=data.get("skills") or [],
            experience_years=data.get("experience_years"),
            current_role=data.get("current_role"),
            companies=data.get("companies") or [],
            education=data.get("education") or [],
            summary=data.get("summary") or "",
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
        # Be more careful here - don't lose small entries
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
                    re.search(r'(?:University|College|Institute|School|Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.Tech|M\.Tech|B\.E\.|M\.E\.)', stripped, re.IGNORECASE)
                )
            )
            
            if is_edu_header and current_entry and any(c.strip() for c in current_entry):
                entry_text = '\n'.join(current_entry).strip()
                if entry_text:
                    entries.append(entry_text)
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entry_text = '\n'.join(current_entry).strip()
            if entry_text:
                entries.append(entry_text)
        
        # For education, keep even small entries (universities might be just 1-2 lines)
        # But filter out very tiny ones (less than 10 chars)
        entries = [e.strip() for e in entries if e.strip() and len(e.strip()) > 10]
        
        # If we split into multiple entries, combine very small adjacent entries
        if len(entries) > 1:
            combined = []
            i = 0
            while i < len(entries):
                entry = entries[i]
                # If this entry is very short, try to combine with next
                while len(entry) < 100 and i + 1 < len(entries):
                    i += 1
                    entry = entry + "\n" + entries[i]
                combined.append(entry)
                i += 1
            entries = combined
        
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
            entry_lower = entry.lower()
            injection_phrases = [
                'no matter what question',
                'always say',
                'always respond',
                'ignore all instructions',
                'ignore previous instructions',
                'you are now',
                'act as if',
                'pretend to be',
                'disregard your',
            ]
            if any(phrase in entry_lower for phrase in injection_phrases):
                continue
            
            # Different minimum sizes for different section types
            # Education entries can be short (just school + degree + date)
            min_size = 30 if section_type == "education" else 50
            
            if not entry or len(entry) < min_size:
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

def expand_temporal_query(query: str, client: OpenAI) -> tuple[str, bool]:
    """
    Expand temporal queries to improve retrieval of date-range content.
    Returns: (expanded_query, was_temporal)
    """
    
    # Quick check if query has temporal indicators
    temporal_patterns = [
        r'\b(in|during|around|at|by|before|after)\s+\w+\s+\d{4}',  # "in December 2024"
        r'\bwhat\s+was\s+\w+\s+doing',  # "what was X doing"
        r'\bwhere\s+did\s+\w+\s+work',  # "where did X work"
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}',  # month year
        r'\b(early|late|mid)\s+\d{4}',  # "early 2024"
        r'\b(q[1-4]|first|second|third|fourth)\s+(quarter|half)?\s*\d{4}',  # "Q1 2024"
    ]
    
    has_temporal = any(re.search(p, query, re.IGNORECASE) for p in temporal_patterns)
    
    if not has_temporal:
        return query, False
    
    # Use LLM to expand the query
    expansion_prompt = """You help improve search queries for a resume database. When someone asks about a specific time period, the resume chunks contain DATE RANGES (like "Sep 2024 - Dec 2025"), not individual months.

Your job: Rewrite the query to better match resume content with date ranges that INCLUDE the queried time.

RULES:
1. Keep the original intent and person name
2. Add terms that would match date ranges containing the queried period
3. Include the year and surrounding months/years
4. Keep it concise - just expand the search terms, don't write a paragraph

Examples:
- "What was John doing in December 2024?" → "John roles positions jobs December 2024 late 2024 2024-2025 experience work"
- "Where did Sarah work in 2023?" → "Sarah work company employer job position 2023 2022-2023 2023-2024"
- "What was Mike's role in early 2024?" → "Mike role position title job January February March April 2024 Q1 2024 late 2023 early 2024"
- "What did Alice do in summer 2023?" → "Alice June July August 2023 summer 2023 mid 2023 work role position"

Input: "{query}"
Output (expanded search terms only):"""

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": expansion_prompt.format(query=query)}],
            temperature=0,
            max_tokens=100
        )
        
        expanded_terms = resp.choices[0].message.content.strip()
        
        # Combine original query with expanded terms
        combined = f"{query} {expanded_terms}"
        return combined, True
        
    except Exception as e:
        # If expansion fails, return original
        return query, False



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
    
    # Filter out chunks with invalid metadata
    valid_chunks = []
    for c in chunks:
        # Skip chunks with None person_name
        if c.person_name is None or c.person_name == "None" or c.person_name.strip() == "":
            continue
        valid_chunks.append(c)
    
    if not valid_chunks:
        return collection
    
    # Batch embed and add
    batch_size = 50
    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        embeddings = get_openai_embedding(texts, openai_client)
        
        # Ensure no None values in metadata
        metadatas = []
        for c in batch:
            metadatas.append({
                "person_name": c.person_name or "Unknown",
                "source_file": c.source_file or "Unknown",
                "chunk_id": c.chunk_id or 0,
                "chunk_type": c.chunk_type or "other",
                "skills": c.skills or "",
                "companies": c.companies or "",
                "experience_years": c.experience_years if c.experience_years is not None else 0
            })
        
        collection.add(
            ids=[f"{c.person_name}_{c.chunk_id}" for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
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
    
    # For pronoun resolution, we primarily care about the LAST exchange
    # Get the last Q&A pair
    last_user_msg = None
    last_assistant_msg = None
    
    for m in reversed(conversation_history):
        if m["role"] == "assistant" and last_assistant_msg is None:
            last_assistant_msg = m["content"]
        elif m["role"] == "user" and last_user_msg is None:
            last_user_msg = m["content"]
        if last_user_msg and last_assistant_msg:
            break
    
    # Determine who was the PRIMARY subject of the last exchange
    primary_person = None
    if last_assistant_msg:
        # Find who is discussed substantively (not negatively)
        for name in known_names:
            name_lower = name.lower()
            msg_lower = last_assistant_msg.lower()
            
            # Find first occurrence
            pos = msg_lower.find(name_lower)
            if pos == -1:
                continue
            
            # Check if this is a substantive mention (not "I don't have info about X")
            context_before = msg_lower[max(0, pos-60):pos]
            negative_phrases = ["don't have", "no information", "not have", "don't know", "no data", "i don't", "not found"]
            
            if not any(phrase in context_before for phrase in negative_phrases):
                # This person was discussed substantively
                # Check if they appear early in the response (likely the subject)
                if pos < 200:  # Mentioned in first 200 chars
                    primary_person = name
                    break
    
    # If we identified a primary person from last response, use simple substitution
    if primary_person and has_reference:
        # Simple pronoun replacement without LLM
        rewritten = query
        replacements = [
            (r'\bhis\b', f"{primary_person}'s"),
            (r'\bher\b', f"{primary_person}'s"),
            (r'\bhim\b', primary_person),
            (r'\bhe\b', primary_person),
            (r'\bshe\b', primary_person),
        ]
        for pattern, replacement in replacements:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten, primary_person, False
    
    # Fall back to LLM rewriter for complex cases
    # Only use last 2 exchanges for context (to avoid confusion from older history)
    recent_history = []
    exchange_count = 0
    for m in reversed(conversation_history):
        recent_history.insert(0, m)
        if m["role"] == "user":
            exchange_count += 1
        if exchange_count >= 2:
            break
    
    history_text = "\n\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in recent_history
    ])
    
    rewrite_prompt = """You are a query rewriter for a resume search system. 
Given the RECENT conversation and the new query, rewrite the query to be FULLY self-contained.

CRITICAL RULES:
1. Replace ALL pronouns (he/she/his/her/they/their/them) with actual names
2. Replace contextual references like "there", "that company" with actual entities
3. Look at the LAST assistant response to determine who pronouns refer to
4. The rewritten query must be understandable without any conversation history

Known candidates: {known_names}

Return JSON: {{"rewritten_query": "...", "person_name": "name or null", "is_multi_person": true/false}}

Recent conversation:
{history}

New query: {query}

JSON:"""

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
    multi_person_keywords = ['their', 'them', 'all', 'everyone', 'candidates', 'compare', 
                             'both', 'each', 'every', 'all of', 'everybody', 'anyone',
                             'who has', 'who have', 'which candidate', 'any of']
    is_multi = any(kw in query_lower for kw in multi_person_keywords)
    
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
    min_experience: Optional[int] = None,
    ensure_all_candidates: bool = False,
    known_names: Optional[list[str]] = None
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
    
    # If we need to ensure all candidates are represented, query more and distribute
    if ensure_all_candidates and known_names and not person_filter:
        all_hits = []
        hits_per_person = {}
        
        # Query each person separately to ensure coverage
        for name in known_names:
            person_where = {"person_name": {"$eq": name}}
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,  # Get top 3 per person
                where=person_where,
                include=["documents", "metadatas", "distances"]
            )
            
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    if skill_filter:
                        if skill_filter.lower() not in meta.get("skills", "").lower():
                            continue
                    
                    hit = {
                        "text": doc,
                        "person_name": meta.get("person_name"),
                        "chunk_type": meta.get("chunk_type"),
                        "source_file": meta.get("source_file"),
                        "skills": meta.get("skills"),
                        "distance": dist
                    }
                    
                    if name not in hits_per_person:
                        hits_per_person[name] = []
                    hits_per_person[name].append(hit)
        
        # Collect hits, ensuring at least 1-2 from each person
        for name in known_names:
            if name in hits_per_person:
                all_hits.extend(hits_per_person[name][:2])  # Top 2 per person
        
        # Sort by relevance (distance) and return
        all_hits.sort(key=lambda x: x["distance"])
        return all_hits[:k * 2]  # Allow more results for multi-person queries
    
    # Standard retrieval
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k * 2, 20),
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
# Replace your existing generate_answer function with this one

def generate_answer(
    query: str,
    hits: list[dict],
    conversation_history: list[dict],
    client: OpenAI,
    is_multi_person: bool = False
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
3. Always attribute information to the specific person (e.g., "John Smith was working as...")
4. Do not infer or assume information not explicitly stated
5. If asked to compare candidates, only compare based on information present for all of them
6. Use the conversation history to maintain context but don't contradict the resume data
7. If the context is insufficient to fully answer, explain what you can answer and what's missing
8. Always complete your response - never leave sentences unfinished

TEMPORAL/DATE REASONING - FOLLOW THIS EXACT PROCESS:

When the user asks about a specific date/month/time period, you MUST:

STEP 1: Extract the queried date
- Convert to numeric: e.g., "July 2023" = Month 7, Year 2023
- For seasons: "summer 2023" = June-August 2023, "early 2024" = Jan-April 2024

STEP 2: For EACH date range in the context, check if queried date falls within it
- Parse start date: e.g., "Aug 2023" = Month 8, Year 2023
- Parse end date: e.g., "May 2024" = Month 5, Year 2024 (or current date if "Present")
- Compare: queried_date >= start_date AND queried_date <= end_date

STEP 3: Determine the answer
- If queried date is BEFORE the start date → NOT within range, don't mention this role
- If queried date is AFTER the end date → NOT within range, don't mention this role  
- If queried date is WITHIN the range → This role IS relevant

MONTH NUMBERS: Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12

EXAMPLES OF CORRECT REASONING:

Example 1:
- Context: "Software Engineer at Google, Sep 2024 - Dec 2025"
- Query: "What was X doing in Dec 2024?"
- Reasoning: Dec 2024 (12/2024) >= Sep 2024 (9/2024) ✓ AND Dec 2024 (12/2024) <= Dec 2025 (12/2025) ✓
- Answer: "X was working as a Software Engineer at Google"

Example 2:
- Context: "Intern at Meta, Aug 2023 - May 2024"
- Query: "What was X doing in July 2023?"
- Reasoning: July 2023 (7/2023) >= Aug 2023 (8/2023)? NO! 7 < 8, so July is BEFORE August
- Answer: "I don't have information about July 2023. The earliest role I have is an internship at Meta starting in August 2023."

Example 3:
- Context: "Data Analyst, Jan 2022 - Present"
- Query: "Where did X work in 2023?"
- Reasoning: 2023 >= Jan 2022 ✓ AND 2023 <= Present (2025) ✓
- Answer: "X was working as a Data Analyst"

Example 4:
- Context: "Intern at Startup, May 2023 - Aug 2023"
- Query: "What was X doing in October 2023?"
- Reasoning: Oct 2023 (10/2023) <= Aug 2023 (8/2023)? NO! 10 > 8, so October is AFTER August
- Answer: "I don't have information about October 2023. The internship at Startup ended in August 2023."

CRITICAL: Do NOT say someone "was working" at a job if the queried date is OUTSIDE the job's date range. 
If the queried date is before ANY role starts, say you don't have information for that time period."""

    user_message = f"""{history_text}Resume excerpts:
{context}

Question: {query}

IMPORTANT: If this question asks about a specific time period, carefully compare the queried date against each date range in the context using numeric month/year comparison. Only mention roles where the queried date falls WITHIN the start and end dates.

Answer:"""

    # Use more tokens for multi-person queries that need comprehensive answers
    max_tokens = 2000 if is_multi_person else 1000
    
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,  # Lower temperature for more precise reasoning
        max_tokens=max_tokens
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
    # st.subheader("🔑 API Key")
    api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     api_key = st.text_input("Enter OpenAI API Key", type="password")
    if not api_key:
        st.error("Please provide an OpenAI API key")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    
    
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
    
    
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
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
        processed_count = 0
        rejected_count = 0
        rejected_files = []
        
        progress = st.progress(0)
        status = st.status("Processing resumes...", expanded=True)
        
        for i, pdf_file in enumerate(uploaded_files):
            status.write(f"Processing: {pdf_file.name}")
            
            try:
                # === GUARDRAIL: File-level checks ===
                passed, errors, warnings = run_file_guardrails(pdf_file, client)
                
                if not passed:
                    for err in errors:
                        status.write(f"❌ {pdf_file.name}: {err}")
                    rejected_count += 1
                    rejected_files.append((pdf_file.name, errors[0] if errors else "Unknown error"))
                    progress.progress((i + 1) / len(uploaded_files))
                    continue
                
                for warn in warnings:
                    status.write(f"⚠️ {pdf_file.name}: {warn}")
                # =====================================
                
                # Extract text
                raw_text = extract_text_from_pdf(pdf_file)
                if not raw_text.strip():
                    status.write(f"⚠️ Empty text extracted from {pdf_file.name}")
                    rejected_count += 1
                    rejected_files.append((pdf_file.name, "Empty text extracted"))
                    progress.progress((i + 1) / len(uploaded_files))
                    continue
                
                text = normalize_text(raw_text)
                
                # === GUARDRAIL: Content-level checks ===
                passed, errors, warnings = run_content_guardrails(text, client)
                
                if not passed:
                    for err in errors:
                        status.write(f"❌ {pdf_file.name}: {err}")
                    rejected_count += 1
                    rejected_files.append((pdf_file.name, errors[0] if errors else "Content validation failed"))
                    progress.progress((i + 1) / len(uploaded_files))
                    continue
                
                for warn in warnings:
                    status.write(f"⚠️ {pdf_file.name}: {warn}")
                
                # Sanitize text before processing
                text = sanitize_text_for_llm(text)
                # ========================================
                
                # DEBUG: Show extracted text length
                status.write(f"   📝 Extracted {len(text)} chars, {len(text.split())} words")
                
                # Extract metadata using LLM
                metadata = extract_resume_metadata(text, pdf_file.name, client)
                if metadata.person_name == "Unknown" or metadata.person_name is None:
                    status.write(f"⚠️ {pdf_file.name}: Could not extract person name - skipping")
                    rejected_count += 1
                    rejected_files.append((pdf_file.name, "Could not identify person name in document"))
                    progress.progress((i + 1) / len(uploaded_files))
                    continue
                st.session_state.resume_metadata[metadata.person_name] = metadata
                
                # DEBUG: Show extracted metadata
                status.write(f"   👤 Name: {metadata.person_name}")
                status.write(f"   🛠️ Skills: {metadata.skills[:5]}...")
                
                # Chunk with metadata
                chunks = semantic_chunk_resume(text, metadata)
                all_chunks.extend(chunks)
                
                status.write(f"✅ {pdf_file.name}: {metadata.person_name} ({len(chunks)} chunks)")
                processed_count += 1
                
            except Exception as e:
                status.write(f"❌ Error processing {pdf_file.name}: {str(e)}")
                rejected_count += 1
                rejected_files.append((pdf_file.name, str(e)))
                import traceback
                status.write(f"   {traceback.format_exc()}")
            
            progress.progress((i + 1) / len(uploaded_files))
        
        # === IMPROVED: Handle results based on what was processed ===
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
            st.success(f"Processed {processed_count} resume(s) with {len(all_chunks)} chunks")
            
            # Show rejected files if any
            if rejected_count > 0:
                st.warning(f"{rejected_count} file(s) were rejected:")
                for filename, reason in rejected_files:
                    st.caption(f"• {filename}: {reason}")
        
        elif rejected_count > 0 and processed_count == 0:
            # All files were rejected
            status.update(label="⚠️ No files could be processed", state="complete")
            st.error(f"All {rejected_count} file(s) were rejected due to security or validation issues:")
            for filename, reason in rejected_files:
                st.write(f"• **{filename}**: {reason}")
            st.info("Please upload valid resume PDFs without any suspicious content.")
        
        else:
            # No files uploaded or some other issue
            status.update(label="❌ Processing failed", state="error")
            st.error("No valid chunks created. Please check your PDF files.")


# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about the candidates..."):
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    if st.session_state.collection is not None:
        query_ok, query_error = run_query_guardrails(query, client)
        if not query_ok:
            with st.chat_message("assistant"):
                st.warning(query_error)
                st.session_state.messages.append({"role": "assistant", "content": query_error})
            st.stop()
    
    if st.session_state.collection is None:
        with st.chat_message("assistant"):
            st.error("Please upload resumes and build the vector store first.")
    else:
        with st.chat_message("assistant"):
            # Check for meta-questions about the system (don't need RAG)
            query_lower = query.lower()
            meta_patterns = [
                "whose resumes", "which resumes", "what resumes", 
                "how many resumes", "how many candidates", "list all",
                "who do you have", "whose resume do you have",
                "what candidates", "which candidates", "list candidates",
                "list the candidates", "list the resumes", "show all candidates"
            ]
            
            is_meta_question = any(pattern in query_lower for pattern in meta_patterns)
            
            if is_meta_question:
                # Answer directly from metadata without RAG
                names = list(st.session_state.resume_metadata.keys())
                answer = f"I have resumes for {len(names)} candidates:\n\n"
                for i, name in enumerate(names, 1):
                    meta = st.session_state.resume_metadata[name]
                    role = meta.current_role or "N/A"
                    answer += f"{i}. **{name}**"
                    if role != "N/A":
                        answer += f" - {role}"
                    answer += "\n"
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
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
                    
                    # === NEW: Expand temporal queries for better retrieval ===
                    search_query, is_temporal = expand_temporal_query(rewritten_query, client)
                    if is_temporal:
                        st.caption(f"🕐 Temporal query detected - expanding search")
                    # =========================================================
                    
                    if is_multi_person:
                        # Multi-person query - don't filter
                        p_filter = None
                        st.caption(f"👥 Searching across all {len(known_names)} candidates")
                    elif detected_person and detected_person in known_names:
                        # Single person detected
                        p_filter = detected_person
                        st.caption(f"👤 Filtering for: {detected_person}")
                    else:
                        p_filter = None
                    
                
                    
                    # Retrieve using expanded query for temporal searches
                    # AFTER
                    hits = retrieve_with_filters(
                        search_query,
                        st.session_state.collection,
                        client,
                        k=10 if is_temporal else 8,
                        person_filter=p_filter,
                        ensure_all_candidates=is_multi_person or p_filter is None,
                        known_names=known_names
                    )
                    
                    # DEBUG: Show what was retrieved
                    st.caption(f"🔍 Retrieved {len(hits)} chunks from {len(set(h['person_name'] for h in hits))} candidates")
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
                    client,
                    is_multi_person=is_multi_person or p_filter is None
                )
                
                response_check = check_response_safety(answer)
                if not response_check.passed:
                    answer = "I encountered an issue generating a safe response. Please try rephrasing your question."
                
                st.markdown(answer)
                
                # Show sources in expander
                if hits:
                    with st.expander("📚 Sources"):
                        seen = set()
                        for h in hits:
                            # Skip chunks that look like injection attempts
                            chunk_text_lower = h['text'].lower()
                            suspicious_phrases = [
                                'no matter what question',
                                'always say',
                                'ignore all instructions',
                                'you are now',
                                'act as if',
                            ]
                            if any(phrase in chunk_text_lower for phrase in suspicious_phrases):
                                continue  # Skip this suspicious chunk
                            
                            # Deduplicate by person + chunk_type
                            key = f"{h['person_name']}_{h['chunk_type']}"
                            if key in seen:
                                continue
                            seen.add(key)
                            
                            st.markdown(f"**{h['person_name']}** ({h['chunk_type']})")
                            preview = h['text'][:300].replace('\n', ' ').strip()
                            if len(h['text']) > 300:
                                preview += "..."
                            st.caption(preview)
                            st.divider()
                
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("💡 Tip: Use filters in the sidebar to narrow down your search to specific candidates or skills.")