import os
import re
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any
from dataclasses import dataclass

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

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
            severity="error",
        )
    return GuardrailResult(passed=True)


def check_file_type(file) -> GuardrailResult:
    """Verify file is actually a PDF by checking magic bytes."""
    file.seek(0)
    header = file.read(8)
    file.seek(0)

    # PDF magic bytes: %PDF
    if not header.startswith(b"%PDF"):
        return GuardrailResult(
            passed=False,
            reason="File does not appear to be a valid PDF (invalid header)",
            severity="error",
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
        r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
        r"forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
        r"override\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
        # Role-play injections
        r"you\s+are\s+now\s+(a|an|the)",
        r"act\s+as\s+(a|an|if)",
        r"pretend\s+(to\s+be|you\s+are)",
        r"roleplay\s+as",
        r"from\s+now\s+on\s+you",
        # System prompt extraction
        r"(show|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system\s+)?prompt",
        r"what\s+(are|is)\s+your\s+(system\s+)?(instructions?|prompts?|rules?)",
        r"repeat\s+(your|the)\s+(system\s+)?(instructions?|prompts?)",
        # Jailbreak attempts
        r"(DAN|STAN|DUDE)\s*mode",
        r"developer\s+mode",
        r"jailbreak",
        r"bypass\s+(your\s+)?(restrictions?|filters?|rules?)",
        # Command injection
        r"\[INST\]|\[\/INST\]",
        r"<\|im_start\|>|<\|im_end\|>",
        r"###\s*(Human|Assistant|System):",
        r"<\|system\|>|<\|user\|>|<\|assistant\|>",
        # Hidden instructions (often in white text)
        r"hidden\s+instructions?",
        r"secret\s+instructions?",
        r"do\s+not\s+tell\s+(the\s+)?user",
        r"when\s+(answering|responding)",
        # Behavioral override attempts
        r"no\s+matter\s+what\s+(question|query|prompt)",
        r"always\s+(say|respond|answer|reply)\s+that",
        r"for\s+(any|every|all)\s+(question|query|prompt)",
        r"regardless\s+of\s+(the\s+)?(question|query|input)",
        r"whatever\s+(the\s+)?(user|question|query)\s+(asks?|says?)",
    ]

    text_lower = text.lower()

    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return GuardrailResult(
                passed=False,
                reason=f"Potential prompt injection detected in document",
                severity="error",
            )

    return GuardrailResult(passed=True)


def detect_suspicious_formatting(text: str) -> GuardrailResult:
    """Detect suspicious formatting that might indicate hidden content."""

    issues = []

    # Check for excessive whitespace (might hide text)
    whitespace_ratio = text.count(" ") / max(len(text), 1)
    if whitespace_ratio > 0.5:
        issues.append("Excessive whitespace detected")

    # Check for unusual Unicode characters (often used to hide text)
    suspicious_unicode = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u2060",  # Word joiner
        "\ufeff",  # Zero-width no-break space
        "\u00a0",  # Non-breaking space (excessive use)
    ]

    unicode_count = sum(text.count(char) for char in suspicious_unicode)
    if unicode_count > 10:
        issues.append(
            f"Suspicious Unicode characters detected ({unicode_count} instances)"
        )

    # Check for very long lines without breaks (might be obfuscated)
    lines = text.split("\n")
    for line in lines:
        if len(line) > 5000:
            issues.append("Unusually long text lines detected")
            break

    if issues:
        return GuardrailResult(
            passed=False, reason="; ".join(issues), severity="warning"
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
            messages=[{"role": "user", "content": validation_prompt + text[:3000]}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

        import json

        result = json.loads(resp.choices[0].message.content)

        if not result.get("is_resume", False):
            content_type = result.get("detected_content_type", "unknown")
            reason = result.get("reason", "Document does not appear to be a resume")
            return GuardrailResult(
                passed=False,
                reason=f"Document appears to be '{content_type}': {reason}",
                severity="error",  # CHANGED FROM "warning" TO "error"
            )

        # Low confidence warning (still allow, but warn)
        if result.get("confidence") == "low":
            return GuardrailResult(
                passed=True,
                reason="Document may not be a standard resume format",
                severity="warning",
            )

        return GuardrailResult(passed=True)

    except Exception as e:
        # If validation fails, allow but warn
        return GuardrailResult(
            passed=True,
            reason=f"Could not validate document type: {str(e)}",
            severity="warning",
        )


# ============================================================
# QUERY-LEVEL GUARDRAILS
# ============================================================


def check_query_injection(query: str) -> GuardrailResult:
    """Check if user query contains prompt injection attempts."""

    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)",
        r"system\s*prompt",
        r"you\s+are\s+now",
        r"act\s+as\s+if",
        r"pretend\s+(to\s+be|you)",
        r"\[INST\]",
        r"<\|",
        r"###\s*(Human|System|Assistant)",
    ]

    query_lower = query.lower()

    for pattern in injection_patterns:
        if re.search(pattern, query_lower):
            return GuardrailResult(
                passed=False,
                reason="Query contains suspicious patterns",
                severity="warning",
            )

    return GuardrailResult(passed=True)


def check_query_relevance(query: str, client: OpenAI) -> GuardrailResult:
    """Check if query is relevant to resume/candidate information."""

    # Quick keyword check first (avoid LLM call for obvious cases)
    resume_keywords = [
        "experience",
        "skill",
        "education",
        "work",
        "job",
        "role",
        "company",
        "project",
        "resume",
        "candidate",
        "qualification",
        "degree",
        "university",
        "intern",
        "position",
        "team",
        "manage",
        "develop",
        "build",
        "create",
        "year",
        "month",
        "doing",
        "did",
        "where",
        "what",
        "who",
        "which",
        "compare",
        "list",
        "show",
        "tell",
        "summary",
        "background",
        "history",
        "github",
        "gihtub",
        "git hub",
        "repo",
        "repos",
        "repository",
        "repositories",
        "portfolio",
        "profile",
        "link",
        "links",
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
- Questions about GitHub, portfolio links, or repository work from the candidate
- Weather questions (supported in this app via weather tool)

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
            messages=[
                {"role": "user", "content": relevance_prompt.format(query=query)}
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )

        import json

        result = json.loads(resp.choices[0].message.content)

        if not result.get("is_relevant", True):
            return GuardrailResult(
                passed=False,
                reason=result.get(
                    "reason", "Query does not appear to be about resume information"
                ),
                severity="info",
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
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
        r"\b\d{16}\b",  # Credit card number
        r"password\s*[:=]\s*\S+",  # Password exposure
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                reason="Response may contain sensitive information",
                severity="error",
            )

    # Agent guardrail: don't reveal AI/language model identity
    # Avoid matching job titles like "assistant professor", "research assistant"
    # Allow "AI" in job titles: "AI Engineer", "Generative AI Engineer", "AI researcher", etc.
    # Allow general references to LLM technology (e.g. "built LLM tools", "LLM-based")
    ai_job_title_lookahead = (
        r"(?!\s+engineer)(?!\s+researcher)(?!\s+developer)(?!\s+scientist)"
        r"(?!\s+specialist)(?!\s+co-?op)(?!\s+intern)(?!\s+tools)"
        r"(?!\s+model)(?!\s+application)(?!\s+platform)"
    )
    ai_reveal_patterns = [
        r"\bas (a|an) (ai|language model)\b" + ai_job_title_lookahead,
        r"\bi('m| am) (a|an) (ai|language model)\b" + ai_job_title_lookahead,
        r"\b(i'm|i am) (an? )?assistant\b(?!\s+professor)(?!\s+at\s)",
        r"\bopenai\b",
        r"\bi('m| am) (a|an) llm\b",
    ]
    for pattern in ai_reveal_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return GuardrailResult(
                passed=False, reason="Response may reveal AI identity", severity="error"
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


def run_content_guardrails(
    text: str, client: OpenAI
) -> tuple[bool, list[str], list[str]]:
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


def _is_contextual_followup_query(
    query: str, conversation_history: Optional[list[dict]] = None
) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    followup_markers = [
        "that",
        "this",
        "it",
        "those",
        "these",
        "more details",
        "expand on that",
        "tell me more",
        "more about that",
        "go deeper",
        "elaborate",
    ]
    if not any(marker in q for marker in followup_markers):
        return False
    if not conversation_history:
        return False
    recent = conversation_history[-6:]
    return any(
        m.get("role") == "assistant" and (m.get("content") or "").strip()
        for m in recent
    )


def _augment_followup_query_with_recent_subject(
    query: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Expand referential follow-up queries with the most recent recruiter question context."""
    if not _is_contextual_followup_query(query, conversation_history):
        return query
    if not conversation_history:
        return query
    for m in reversed(conversation_history):
        if m.get("role") != "user":
            continue
        prev = (m.get("content") or "").strip()
        if prev and prev.strip().lower() != query.strip().lower():
            return f"{query}\nRelated prior recruiter question: {prev}"
    return query


def _extract_followup_details(text: str) -> dict[str, Optional[str]]:
    src = text or ""
    lower = src.lower()
    day_match = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|next\s+week)\b",
        lower,
        flags=re.IGNORECASE,
    )
    time_match = re.search(
        r"\b(?:at\s*)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
        src,
        flags=re.IGNORECASE,
    )
    tz_match = re.search(
        r"\b(utc|gmt|est|edt|cst|cdt|mst|mdt|pst|pdt)\b",
        lower,
        flags=re.IGNORECASE,
    )
    return {
        "day": day_match.group(1).strip() if day_match else None,
        "time": (
            re.sub(
                r"(?i)\s*(am|pm)\b",
                lambda m: f" {m.group(1).upper()}",
                re.sub(r"\s+", " ", time_match.group(1)).strip(),
            )
            if time_match
            else None
        ),
        "timezone": tz_match.group(1).upper() if tz_match else None,
    }


def _extract_recruiter_identity(text: str) -> dict[str, Optional[str]]:
    src = (text or "").strip()
    name = None
    company = None

    name_patterns = [
        r"\b(?:i['’]m|i am|my name is)\s+([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2})(?=\s+(?:from|at|with)\b|[.,;!]|$)",
        r"\bthis is\s+([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2})(?=\s+(?:from|at|with)\b|[.,;!]|$)",
    ]
    for p in name_patterns:
        m = re.search(p, src, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip(".,")
            break

    company_patterns = [
        r"\bfrom\s+([A-Z][A-Za-z0-9& \-]{1,50}?)(?=[\.,;]|$)",
        r"\bwith\s+([A-Z][A-Za-z0-9& \-]{1,50}?)(?=[\.,;]|$)",
        r"\bat\s+([A-Z][A-Za-z0-9& \-]{1,50}?)(?=[\.,;]|$)",
    ]
    for p in company_patterns:
        m = re.search(p, src, flags=re.IGNORECASE)
        if not m:
            continue
        cand = m.group(1).strip().rstrip(".,")
        if any(
            tok in cand.lower()
            for tok in [
                "follow up",
                "everything looks",
                "good for now",
                "friday",
                "pm",
                "am",
            ]
        ):
            continue
        company = cand
        break

    return {"name": name, "company": company}


def _cleanup_recruiter_identity_fields(
    recruiter_name: Optional[str],
    recruiter_company: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    name = (recruiter_name or "").strip() or None
    company = (recruiter_company or "").strip() or None

    if name and company:
        lowered = name.lower()
        for prep in [" from ", " at ", " with "]:
            marker = f"{prep}{company.lower()}"
            if marker in lowered:
                name = name[: lowered.find(marker)].strip()
                break

    if name:
        tokens = [t for t in re.split(r"\s+", name) if t]
        if tokens and all(t.islower() for t in tokens):
            name = " ".join(t.capitalize() for t in tokens)

    return name, company


def _extract_recruiter_context_fields(
    query: str,
    client: OpenAI,
) -> dict[str, Optional[str]]:
    """
    Deterministic parser with LLM fallback for recruiter identity/scheduling context.
    """
    query = query or ""
    identity = _extract_recruiter_identity(query)
    followup = _extract_followup_details(query)
    out = {
        "recruiter_name": identity.get("name"),
        "recruiter_company": identity.get("company"),
        "followup_day": followup.get("day"),
        "followup_time": followup.get("time"),
        "followup_timezone": followup.get("timezone"),
    }
    if any(v for v in out.values()):
        return out

    # LLM fallback for tricky phrasing.
    try:
        prompt = """Extract recruiter context fields from this message.
Return JSON with keys:
- recruiter_name (string or null)
- recruiter_company (string or null)
- followup_day (string or null)
- followup_time (string or null)
- followup_timezone (string or null)
If none are present, return nulls.

Message: {text}
"""
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt.format(text=query[:500])}],
            temperature=0,
            max_tokens=180,
            response_format={"type": "json_object"},
        )
        data = json.loads((resp.choices[0].message.content or "{}").strip())
        return {
            "recruiter_name": data.get("recruiter_name"),
            "recruiter_company": data.get("recruiter_company"),
            "followup_day": data.get("followup_day"),
            "followup_time": data.get("followup_time"),
            "followup_timezone": data.get("followup_timezone"),
        }
    except Exception:
        return out


def _normalize_followup_display_parts(
    day: Optional[str],
    time_str: Optional[str],
    timezone_str: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    day_map = {
        "monday": "Monday",
        "tuesday": "Tuesday",
        "wednesday": "Wednesday",
        "thursday": "Thursday",
        "friday": "Friday",
        "saturday": "Saturday",
        "sunday": "Sunday",
        "today": "today",
        "tomorrow": "tomorrow",
        "next week": "next week",
    }
    d = (day or "").strip()
    t = (time_str or "").strip()
    z = (timezone_str or "").strip()

    d_norm = day_map.get(d.lower(), d) if d else None
    t_norm = re.sub(r"(?i)\b(am|pm)\b", lambda m: m.group(1).upper(), t) if t else None
    z_norm = z.upper() if z else None
    return d_norm, t_norm, z_norm


def _is_recruiter_context_query(query: str, client: OpenAI) -> bool:
    q = (query or "").lower()
    explicit_markers = [
        "my name is",
        "i'm ",
        "i am ",
        "this is ",
        "let's follow up",
        "lets follow up",
        "follow up",
        "follow-up",
        "schedule",
        "call me",
        "everything looks good",
    ]
    if any(m in q for m in explicit_markers):
        fields = _extract_recruiter_context_fields(query, client)
        return any(v for v in fields.values())
    return False


def _detect_memory_recall_intent(query: str) -> Optional[str]:
    q = (query or "").lower()
    if any(
        p in q
        for p in [
            "what time did we agree",
            "follow up time",
            "follow-up time",
            "when did we agree to follow up",
        ]
    ):
        return "followup_time"
    if any(
        p in q
        for p in [
            "what did we agree to follow up",
            "when are we following up",
            "what follow up did we agree",
        ]
    ):
        return "followup"
    if any(
        p in q
        for p in [
            "what's my name",
            "what is my name",
            "who am i",
            "who did you say i am",
        ]
    ):
        return "recruiter_name"
    if any(
        p in q for p in ["which company", "what company am i from", "where am i from"]
    ):
        return "recruiter_company"
    return None


def run_query_guardrails(
    query: str,
    client: OpenAI,
    conversation_history: Optional[list[dict]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Run all query-level guardrails.
    Returns: (passed, error_message)
    """
    # 1. Injection check
    result = check_query_injection(query)
    if not result.passed:
        return (
            False,
            "I can't process that query. Please ask a question about the candidates.",
        )

    # Weather queries are explicitly supported via agent web_search tool.
    if _is_weather_query(query):
        return True, None

    # 2. Relevance check
    result = check_query_relevance(query, client)
    if not result.passed:
        if _is_contextual_followup_query(query, conversation_history):
            return True, None
        return (
            False,
            "I'm designed to answer questions about job candidates and their resumes. Please ask something related to the uploaded resumes.",
        )

    return True, None


def sanitize_text_for_llm(text: str) -> str:
    """
    Sanitize extracted text before sending to LLM.
    Removes or neutralizes potential injection attempts.
    """
    # Remove zero-width characters
    zero_width_chars = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
    for char in zero_width_chars:
        text = text.replace(char, "")

    # Normalize excessive whitespace
    text = re.sub(r"[ \t]{10,}", " ", text)

    # Remove potential instruction markers
    instruction_markers = [
        r"\[INST\].*?\[/INST\]",
        r"<\|im_start\|>.*?<\|im_end\|>",
        r"###\s*(Human|Assistant|System):.*?(?=###|$)",
    ]

    for pattern in instruction_markers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text


# Try importing PDF libraries - prefer pdfplumber for complex layouts
try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from PyPDF2 import PdfReader

# Load .env from the same directory as this file (so it works regardless of cwd when running streamlit)
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# Config
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o"
EXTRACTION_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "resumes"
MAX_CONTEXT_TURNS = 5
AGENT_MEMORY_FILE = Path(__file__).resolve().parent / "agent_memory.json"
TARGET_PERSON = "Sanjeev Kushal Pendekanti"
TARGET_RESUME_MISSING_ERROR = "Sanjeev Kushal's resume is not uploaded please upload."
MAX_MEMORIES_PER_PERSON = 50
MAX_TOTAL_MEMORIES = 300


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _name_tokens(name: str) -> set[str]:
    return {tok for tok in _normalize_name(name).split() if tok}


def _make_candidate_id(person_name: str, source_file: str) -> str:
    slug = (
        re.sub(r"[^a-z0-9]+", "-", _normalize_name(person_name)).strip("-") or "unknown"
    )
    source_hash = hashlib.sha1((source_file or "").encode("utf-8")).hexdigest()[:10]
    return f"{slug}__{source_hash}"


def resolve_target_person(loaded_names: list[str]) -> Optional[str]:
    """
    Resolve which loaded candidate should be treated as the canonical target person.
    Allows minor extraction variance while requiring strong token overlap with target name.
    """
    if not loaded_names:
        return None

    target_norm = _normalize_name(TARGET_PERSON)
    target_tokens = _name_tokens(TARGET_PERSON)
    best_name = None
    best_score = -1

    for name in loaded_names:
        if _normalize_name(name) == target_norm:
            return name
        tokens = _name_tokens(name)
        score = len(tokens & target_tokens)
        if score > best_score:
            best_score = score
            best_name = name

    # Require high confidence overlap (at least first + last name tokens)
    if best_name and best_score >= 2:
        return best_name
    return None


def get_current_datetime() -> datetime:
    """Return current local datetime with timezone."""
    return datetime.now().astimezone()


def get_current_datetime_context(now: Optional[datetime] = None) -> str:
    """Return local current date/time for temporal grounding in agent responses."""
    now = now or get_current_datetime()
    return (
        f"Current date/time: {now.strftime('%A')}, {now.strftime('%B')} {now.day}, "
        f"{now.year} {now.strftime('%I:%M %p %Z')}"
    )


st.set_page_config(
    page_title="ResumeAI | Smart Candidate Search",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        flex: 1;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Candidate cards in sidebar */
    .candidate-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.2s ease;
    }
    
    .candidate-card:hover {
        background: rgba(255,255,255,0.15);
        border-color: rgba(255,255,255,0.2);
    }
    
    .candidate-name {
        font-weight: 600;
        font-size: 1rem;
        color: white;
        margin-bottom: 0.25rem;
    }
    
    .candidate-role {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        margin-bottom: 0.5rem;
    }
    
    .candidate-skills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    
    .skill-tag {
        background: rgba(102, 126, 234, 0.3);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.7rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #9ca3af;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar divider */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


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
    candidate_id: str = ""


@dataclass
class ChunkMetadata:
    person_name: str
    candidate_id: str
    source_file: str
    chunk_id: int
    chunk_type: (
        str  # "header", "experience", "skills", "education", "projects", "other"
    )
    skills: str  # comma-separated for filtering
    companies: str
    experience_years: int
    text: str


# Text extraction and normalization
def normalize_text(text: str) -> str:
    """Normalize text, handling broken PDF extraction where each word is on a new line."""
    # First, check if text appears to be broken (many short lines)
    lines = text.split("\n")
    if lines:
        avg_line_len = sum(len(line.strip()) for line in lines if line.strip()) / max(
            len([l for l in lines if l.strip()]), 1
        )

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
                        reconstructed.append(" ".join(current_para))
                        current_para = []
                else:
                    current_para.append(line)

            if current_para:
                reconstructed.append(" ".join(current_para))

            text = "\n\n".join(reconstructed)

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
        lines = [l for l in text.split("\n") if l.strip()]
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
Do not invent or hallucinate information not present in the resume.
For companies, list every employer mentioned (internships, co-ops, full-time, research)."""

    resp = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": extraction_prompt},
            {
                "role": "user",
                "content": f"Resume text:\n\n{text[:8000]}",
            },  # Limit to avoid token issues
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(resp.choices[0].message.content)

        # Handle None or empty person_name
        person_name = data.get("person_name")
        if (
            person_name is None
            or person_name == "null"
            or str(person_name).strip() == ""
        ):
            person_name = "Unknown"

        return ResumeMetadata(
            person_name=person_name,
            skills=data.get("skills") or [],
            experience_years=data.get("experience_years"),
            current_role=data.get("current_role"),
            companies=data.get("companies") or [],
            education=data.get("education") or [],
            summary=data.get("summary") or "",
            source_file=filename,
            candidate_id="",
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
            source_file=filename,
            candidate_id="",
        )


# Semantic chunking based on resume sections
def _extract_company_from_role_text(role_text: str) -> Optional[str]:
    """Heuristic: extract company from first line (e.g. 'Title at Company', 'Company | Title')."""
    if not role_text or not role_text.strip():
        return None
    first_line = role_text.strip().split("\n")[0].strip()
    if not first_line:
        return None
    # "Software Engineer at Boeing" or "Boeing | Software Engineer" or "Boeing, Inc. – Software Engineer"
    for sep in [" at ", " | ", " – ", " - ", " — ", ", "]:
        if sep in first_line:
            parts = first_line.split(sep, 1)
            if len(parts) == 2:
                # Prefer the part that looks like a company (often shorter or has Inc/LLC)
                a, b = parts[0].strip(), parts[1].strip()
                if 2 <= len(a) <= 80 and not a.lower().startswith(
                    ("software", "engineer", "intern", "developer", "associate")
                ):
                    return a
                if 2 <= len(b) <= 80 and not b.lower().startswith(
                    ("software", "engineer", "intern", "developer", "associate")
                ):
                    return b
                return a if len(a) <= len(b) else b
    return None


def identify_section_type(text: str) -> str:
    text_lower = text.lower()[:500]
    if any(
        kw in text_lower
        for kw in [
            "experience",
            "work history",
            "employment",
            "professional background",
            "internships",
            "co-op",
            "coop",
            "research experience",
            "positions",
        ]
    ):
        return "experience"
    if any(
        kw in text_lower
        for kw in [
            "skill",
            "technologies",
            "tools",
            "proficiencies",
            "competencies",
            "technical",
        ]
    ):
        return "skills"
    if any(
        kw in text_lower
        for kw in ["education", "degree", "university", "college", "certification"]
    ):
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

    # Date patterns: month-year and year-only (e.g. "Jun 2022" or "2022 - 2023")
    date_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}"
    year_only = r"\d{4}"
    date_range_pattern = rf"({date_pattern}|{year_only})\s*[-–—to]+\s*({date_pattern}|{year_only}|Present|Current)"

    # Split by date ranges (each role typically has a date range)
    matches = list(re.finditer(date_range_pattern, section_text, re.IGNORECASE))

    if not matches:
        # No date ranges found, return whole section as one chunk; try to extract company from first line
        company = _extract_company_from_role_text(section_text)
        return [{"text": section_text, "company": company, "title": None}]

    # Extract roles based on date range positions
    for i, match in enumerate(matches):
        # Find the start of this role (either start of text or end of previous role's bullets)
        if i == 0:
            # Look backwards from date to find role title
            role_start = 0
            # Check if there's a section header before first role
            lines_before = section_text[: match.start()].split("\n")
            for j, line in enumerate(lines_before):
                if line.strip().upper() in [
                    "EXPERIENCE",
                    "WORK EXPERIENCE",
                    "PROFESSIONAL EXPERIENCE",
                    "EMPLOYMENT",
                ]:
                    role_start = section_text.find(lines_before[j]) + len(
                        lines_before[j]
                    )
                    break
        else:
            # Start after previous role's content
            role_start = prev_role_end

        # Find end of this role (start of next role or end of section)
        if i + 1 < len(matches):
            # Find where next role's title likely starts (look for line before next date)
            next_match = matches[i + 1]
            # Search backwards from next date to find the role title line
            search_area = section_text[match.end() : next_match.start()]

            # Look for a line that looks like a job title (not a bullet point)
            lines = search_area.split("\n")
            role_end = match.end()
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith(("•", "-", "*", "–", "▪")):
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
        company = _extract_company_from_role_text(role_text)
        first_line = role_text.split("\n")[0] if role_text else ""

        roles.append(
            {
                "text": role_text,
                "company": company,
                "title": first_line[:100] if first_line else None,
                "date_range": match.group(0),
            }
        )

    return roles


def _split_experience_entries_fallback(section_text: str) -> list[dict]:
    """
    Fallback splitter for experience sections when date-range parsing collapses roles.
    Uses date-anchor lines first, then paragraph blocks.
    """
    if not section_text or not section_text.strip():
        return []

    date_pattern = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}"
    year_only = r"\d{4}"
    date_range_pattern = rf"({date_pattern}|{year_only})\s*[-–—to]+\s*({date_pattern}|{year_only}|Present|Current)"

    lines = section_text.split("\n")
    anchors: list[int] = []
    for i, line in enumerate(lines):
        if re.search(date_range_pattern, line, re.IGNORECASE):
            start = i
            if i > 0:
                prev = lines[i - 1].strip()
                if prev and not prev.startswith(("•", "-", "*", "–", "▪")):
                    start = i - 1
            anchors.append(start)

    anchors = sorted(set(anchors))
    entries: list[str] = []
    if anchors:
        for idx, start in enumerate(anchors):
            end = anchors[idx + 1] if idx + 1 < len(anchors) else len(lines)
            chunk = "\n".join(lines[start:end]).strip()
            if len(chunk) > 25:
                entries.append(chunk)

    if len(entries) < 2:
        blocks = [b.strip() for b in re.split(r"\n\s*\n+", section_text) if b.strip()]
        block_entries = [b for b in blocks if len(b) > 25]
        if len(block_entries) > len(entries):
            entries = block_entries

    out = []
    for e in entries:
        out.append({"text": e, "company": _extract_company_from_role_text(e)})
    return out


def extract_entries_from_section(section_text: str, section_type: str) -> list[dict]:
    """Extract individual entries (roles, projects, education) from a section."""

    if section_type == "experience":
        roles = extract_roles_from_experience(section_text)
        entries = [
            {"text": r["text"], "company": r.get("company")}
            for r in roles
            if r["text"].strip()
        ]
        if len(entries) <= 1:
            fallback_entries = _split_experience_entries_fallback(section_text)
            if len(fallback_entries) > len(entries):
                entries = fallback_entries
        return entries

    elif section_type == "projects":
        # Split projects by project headers (usually Name | Tech Stack | Date)
        # Look for lines that start a new project (not bullet points)
        entries = []
        current_entry = []
        lines = section_text.split("\n")

        for line in lines:
            stripped = line.strip()
            # Detect project header: not a bullet, contains | or tech keywords
            is_project_header = (
                stripped
                and not stripped.startswith(("•", "-", "*", "–", "▪"))
                and (
                    "|" in stripped
                    or re.search(
                        r"(?:Python|Java|React|Node|Flask|Django|AWS|Docker)",
                        stripped,
                        re.IGNORECASE,
                    )
                )
                and len(stripped.split()) >= 2
            )

            if is_project_header and current_entry:
                entries.append("\n".join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)

        if current_entry:
            entries.append("\n".join(current_entry))

        # Filter out section header if it's the only "entry"
        entries = [e.strip() for e in entries if e.strip() and len(e.strip()) > 20]
        return [
            {"text": e, "company": None}
            for e in (entries if entries else [section_text])
        ]

    elif section_type == "education":
        # Split by university/school names or degree entries
        # Be more careful here - don't lose small entries
        entries = []
        current_entry = []
        lines = section_text.split("\n")

        for line in lines:
            stripped = line.strip()
            # Detect education header: contains university/college keywords or degree
            is_edu_header = (
                stripped
                and not stripped.startswith(("•", "-", "*", "–", "▪"))
                and (
                    re.search(
                        r"(?:University|College|Institute|School|Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.Tech|M\.Tech|B\.E\.|M\.E\.)",
                        stripped,
                        re.IGNORECASE,
                    )
                )
            )

            if (
                is_edu_header
                and current_entry
                and any(c.strip() for c in current_entry)
            ):
                entry_text = "\n".join(current_entry).strip()
                if entry_text:
                    entries.append(entry_text)
                current_entry = [line]
            else:
                current_entry.append(line)

        if current_entry:
            entry_text = "\n".join(current_entry).strip()
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

        return [
            {"text": e, "company": None}
            for e in (entries if entries else [section_text])
        ]

    # For other sections (skills, summary, etc.), keep as single chunk (same shape: dict with text, company)
    return [{"text": section_text, "company": None}]


def semantic_chunk_resume(text: str, metadata: ResumeMetadata) -> list[ChunkMetadata]:
    """Chunk resume by sections and individual entries (roles, projects, etc.)."""
    text = normalize_text(text)

    # Section header patterns (include Internships, Co-op, Research so they chunk like experience)
    section_headers = [
        r"^\s*(EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY|INTERNSHIPS?|CO-OP|COOP|RESEARCH EXPERIENCE|POSITIONS?)",
        r"^\s*(EDUCATION|ACADEMIC BACKGROUND)",
        r"^\s*(SKILLS|TECHNICAL SKILLS|TECHNOLOGIES|CORE COMPETENCIES)",
        r"^\s*(PROJECTS|PERSONAL PROJECTS|ACADEMIC PROJECTS|SIDE PROJECTS)",
        r"^\s*(SUMMARY|PROFESSIONAL SUMMARY|OBJECTIVE|PROFILE|ABOUT ME)",
        r"^\s*(CERTIFICATIONS?|LICENSES?|AWARDS?|HONORS?|ACHIEVEMENTS?)",
        r"^\s*(PUBLICATIONS?|RESEARCH|PAPERS)",
        r"^\s*(LEADERSHIP|ACTIVITIES|EXTRACURRICULAR)",
    ]
    combined_pattern = "|".join(section_headers)

    # Parse into sections
    lines = text.split("\n")
    sections = []
    current_section_lines = []
    current_section_type = "header"

    for line in lines:
        stripped = line.strip()
        is_section_header = False

        if stripped:
            if re.match(combined_pattern, stripped, re.IGNORECASE):
                is_section_header = True
            elif (
                stripped.isupper()
                and 2 <= len(stripped.split()) <= 5
                and len(stripped) > 3
            ):
                # Likely a section header
                is_section_header = True

        if is_section_header:
            # Save previous section
            if current_section_lines:
                section_text = "\n".join(current_section_lines).strip()
                if section_text:
                    sections.append((current_section_type, section_text))
            current_section_lines = [line]
            current_section_type = identify_section_type(stripped)
        else:
            current_section_lines.append(line)

    # Save last section
    if current_section_lines:
        section_text = "\n".join(current_section_lines).strip()
        if section_text:
            sections.append((current_section_type, section_text))

    # Create chunks from sections
    chunks = []
    chunk_id = 0

    for section_type, section_text in sections:
        # Extract individual entries from the section (each entry: {"text": ..., "company": ...})
        raw_entries = extract_entries_from_section(section_text, section_type)

        for entry_obj in raw_entries:
            entry = (
                entry_obj["text"] if isinstance(entry_obj, dict) else entry_obj
            ).strip()
            chunk_company = (
                entry_obj.get("company") if isinstance(entry_obj, dict) else None
            )
            entry_lower = entry.lower()
            injection_phrases = [
                "no matter what question",
                "always say",
                "always respond",
                "ignore all instructions",
                "ignore previous instructions",
                "you are now",
                "act as if",
                "pretend to be",
                "disregard your",
            ]
            if any(phrase in entry_lower for phrase in injection_phrases):
                continue

            # Different minimum sizes: education and experience can have short entries (e.g. one role header)
            min_size = 30 if section_type in ("education", "experience") else 50

            if not entry or len(entry) < min_size:
                continue

            # Companies for this chunk: per-chunk company (e.g. Boeing) + global from metadata
            companies_list = list(metadata.companies[:5])
            if chunk_company and chunk_company not in companies_list:
                companies_list = [chunk_company] + companies_list[:4]
            companies_str = ",".join(companies_list[:5])

            # If entry is still too long (>1500 chars), split by sentences
            if len(entry) > 1500:
                sentences = re.split(r"(?<=[.!?])\s+", entry)
                current_sentences: list[str] = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    draft = " ".join(current_sentences + [sentence]).strip()
                    if len(draft) <= 1200:
                        current_sentences.append(sentence)
                    else:
                        if current_sentences:
                            chunk_text = " ".join(current_sentences).strip()
                            chunk_id += 1
                            chunks.append(
                                ChunkMetadata(
                                    person_name=metadata.person_name,
                                    candidate_id=metadata.candidate_id,
                                    source_file=metadata.source_file,
                                    chunk_id=chunk_id,
                                    chunk_type=section_type,
                                    skills=",".join(metadata.skills[:10]),
                                    companies=companies_str,
                                    experience_years=metadata.experience_years or 0,
                                    text=chunk_text,
                                )
                            )
                        # 1-sentence overlap preserves context across adjacent chunks
                        overlap = current_sentences[-1:] if current_sentences else []
                        current_sentences = overlap + [sentence]
                if current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    chunk_id += 1
                    chunks.append(
                        ChunkMetadata(
                            person_name=metadata.person_name,
                            candidate_id=metadata.candidate_id,
                            source_file=metadata.source_file,
                            chunk_id=chunk_id,
                            chunk_type=section_type,
                            skills=",".join(metadata.skills[:10]),
                            companies=companies_str,
                            experience_years=metadata.experience_years or 0,
                            text=chunk_text,
                        )
                    )
            else:
                chunk_id += 1
                chunks.append(
                    ChunkMetadata(
                        person_name=metadata.person_name,
                        candidate_id=metadata.candidate_id,
                        source_file=metadata.source_file,
                        chunk_id=chunk_id,
                        chunk_type=section_type,
                        skills=",".join(metadata.skills[:10]),
                        companies=companies_str,
                        experience_years=metadata.experience_years or 0,
                        text=entry,
                    )
                )

    # Fallback if no chunks created
    if not chunks:
        chunk_id += 1
        chunks.append(
            ChunkMetadata(
                person_name=metadata.person_name,
                candidate_id=metadata.candidate_id,
                source_file=metadata.source_file,
                chunk_id=chunk_id,
                chunk_type="other",
                skills=",".join(metadata.skills[:10]),
                companies=(
                    ",".join(metadata.companies[:5]) if metadata.companies else ""
                ),
                experience_years=metadata.experience_years or 0,
                text=text[:2000],  # First 2000 chars as fallback
            )
        )

    return chunks


def expand_temporal_query(query: str, client: OpenAI) -> tuple[str, bool]:
    """
    Expand temporal queries to improve retrieval of date-range content.
    Returns: (expanded_query, was_temporal)
    """

    # Quick check if query has temporal indicators
    temporal_patterns = [
        r"\b(in|during|around|at|by|before|after)\s+\w+\s+\d{4}",  # "in December 2024"
        r"\bwhat\s+was\s+\w+\s+doing",  # "what was X doing"
        r"\bwhere\s+did\s+\w+\s+work",  # "where did X work"
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}",  # month year
        r"\b(early|late|mid)\s+\d{4}",  # "early 2024"
        r"\b(q[1-4]|first|second|third|fourth)\s+(quarter|half)?\s*\d{4}",  # "Q1 2024"
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
            messages=[
                {"role": "user", "content": expansion_prompt.format(query=query)}
            ],
            temperature=0,
            max_tokens=100,
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
    return chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))


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
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # Filter out chunks with invalid metadata
    valid_chunks = []
    for c in chunks:
        # Skip chunks with None person_name
        if (
            c.person_name is None
            or c.person_name == "None"
            or c.person_name.strip() == ""
        ):
            continue
        valid_chunks.append(c)

    if not valid_chunks:
        return collection

    # Batch embed and add
    batch_size = 50
    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = get_openai_embedding(texts, openai_client)

        # Ensure no None values in metadata
        metadatas = []
        for c in batch:
            metadatas.append(
                {
                    "person_name": c.person_name or "Unknown",
                    "candidate_id": c.candidate_id
                    or _make_candidate_id(
                        c.person_name or "Unknown", c.source_file or ""
                    ),
                    "source_file": c.source_file or "Unknown",
                    "chunk_id": c.chunk_id or 0,
                    "chunk_type": c.chunk_type or "other",
                    "skills": c.skills or "",
                    "companies": c.companies or "",
                    "experience_years": (
                        c.experience_years if c.experience_years is not None else 0
                    ),
                }
            )

        collection.add(
            ids=[
                f"{(c.candidate_id or _make_candidate_id(c.person_name or 'Unknown', c.source_file or 'unknown'))}_{c.chunk_id}"
                for c in batch
            ],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    return collection


# Query rewriting with context
def rewrite_query(
    query: str, conversation_history: list[dict], known_names: list[str], client: OpenAI
) -> tuple[str, Optional[str], bool]:
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
    pronouns_and_refs = [
        "he",
        "she",
        "his",
        "her",
        "him",
        "they",
        "their",
        "them",
        "that person",
        "that candidate",
        "there",
        "that company",
        "that role",
        "that job",
        "this person",
        "this candidate",
    ]
    query_lower = query.lower()
    has_reference = any(
        f" {p} " in f" {query_lower} "
        or query_lower.startswith(f"{p} ")
        or query_lower.endswith(f" {p}")
        for p in pronouns_and_refs
    )

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
            context_before = msg_lower[max(0, pos - 60) : pos]
            negative_phrases = [
                "don't have",
                "no information",
                "not have",
                "don't know",
                "no data",
                "i don't",
                "not found",
            ]

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
            (r"\bhis\b", f"{primary_person}'s"),
            (r"\bher\b", f"{primary_person}'s"),
            (r"\bhim\b", primary_person),
            (r"\bhe\b", primary_person),
            (r"\bshe\b", primary_person),
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

    history_text = "\n\n".join(
        [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
            for m in recent_history
        ]
    )

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
            {
                "role": "user",
                "content": rewrite_prompt.format(
                    history=history_text,
                    query=query,
                    known_names=", ".join(known_names),
                ),
            }
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
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


def detect_person_in_query(
    query: str, known_names: list[str]
) -> tuple[Optional[str], bool]:
    """
    Check if query mentions known person names.
    Returns: (single_person_name or None, is_multi_person)
    """
    query_lower = query.lower()

    # Check for multi-person indicators
    multi_person_keywords = [
        "their",
        "them",
        "all",
        "everyone",
        "candidates",
        "compare",
        "both",
        "each",
        "every",
        "all of",
        "everybody",
        "anyone",
        "who has",
        "who have",
        "which candidate",
        "any of",
    ]
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
                pattern = rf"\b{re.escape(part.lower())}\b"
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
def _query_tokens(text: str) -> list[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "to",
        "in",
        "on",
        "of",
        "at",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "as",
        "it",
        "this",
        "that",
        "about",
        "me",
        "my",
        "your",
        "you",
        "i",
        "we",
        "our",
        "do",
        "did",
        "what",
        "where",
        "when",
    }
    tokens = re.findall(r"[a-zA-Z0-9+#.-]+", (text or "").lower())
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def _lexical_score(query_tokens: list[str], doc_text: str) -> float:
    if not query_tokens or not doc_text:
        return 0.0
    text_lower = doc_text.lower()
    hit_count = sum(1 for tok in query_tokens if tok in text_lower)
    if hit_count == 0:
        return 0.0
    return hit_count / max(len(query_tokens), 1)


def retrieve_with_filters(
    query: str,
    collection,
    openai_client: OpenAI,
    k: int = 8,
    person_filter: Optional[str] = None,
    skill_filter: Optional[str] = None,
    min_experience: Optional[int] = None,
    ensure_all_candidates: bool = False,
    known_names: Optional[list[str]] = None,
) -> list[dict]:

    query_embedding = get_openai_embedding([query], openai_client)[0]
    query_tokens = _query_tokens(query)

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
                include=["documents", "metadatas", "distances"],
            )

            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    if skill_filter:
                        if skill_filter.lower() not in meta.get("skills", "").lower():
                            continue

                    hit = {
                        "text": doc,
                        "person_name": meta.get("person_name"),
                        "candidate_id": meta.get("candidate_id"),
                        "chunk_type": meta.get("chunk_type"),
                        "source_file": meta.get("source_file"),
                        "skills": meta.get("skills"),
                        "chunk_id": meta.get("chunk_id"),
                        "distance": dist,
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
        return all_hits[: k * 2]  # Allow more results for multi-person queries

    # Vector retrieval
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(k * 3, 20), 60),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    vector_hits = []
    if vector_results["documents"] and vector_results["documents"][0]:
        for doc, meta, dist in zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0],
        ):
            if (
                skill_filter
                and skill_filter.lower() not in meta.get("skills", "").lower()
            ):
                continue
            vector_hits.append(
                {
                    "text": doc,
                    "person_name": meta.get("person_name"),
                    "candidate_id": meta.get("candidate_id"),
                    "chunk_type": meta.get("chunk_type"),
                    "source_file": meta.get("source_file"),
                    "skills": meta.get("skills"),
                    "chunk_id": meta.get("chunk_id"),
                    "distance": dist,
                    "lexical_score": _lexical_score(query_tokens, doc),
                }
            )

    # Lexical retrieval pass for term-heavy queries (company names, exact skills)
    lexical_hits = []
    try:
        docs_result = collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        docs = docs_result.get("documents") or []
        metas = docs_result.get("metadatas") or [{}] * len(docs)
        for idx, doc in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            if (
                skill_filter
                and skill_filter.lower() not in meta.get("skills", "").lower()
            ):
                continue
            score = _lexical_score(query_tokens, doc or "")
            if score <= 0:
                continue
            lexical_hits.append(
                {
                    "text": doc,
                    "person_name": meta.get("person_name"),
                    "candidate_id": meta.get("candidate_id"),
                    "chunk_type": meta.get("chunk_type"),
                    "source_file": meta.get("source_file"),
                    "skills": meta.get("skills"),
                    "chunk_id": meta.get("chunk_id"),
                    "distance": 1.5,
                    "lexical_score": score,
                }
            )
        lexical_hits.sort(key=lambda h: h["lexical_score"], reverse=True)
        lexical_hits = lexical_hits[: min(k * 4, 50)]
    except Exception:
        lexical_hits = []

    # Low-confidence fallback: vector says weak match and lexical pass found nothing
    if vector_hits and not lexical_hits and vector_hits[0].get("distance", 2.0) > 1.1:
        return []

    # Weighted merge/rerank
    merged: dict[str, dict] = {}
    for hit in vector_hits:
        key = f"{hit.get('source_file','')}::{hit.get('chunk_id','')}::{(hit.get('text') or '')[:120]}"
        vector_component = max(
            0.0, 1.0 - min(float(hit.get("distance", 1.5)), 1.5) / 1.5
        )
        lexical_component = min(float(hit.get("lexical_score", 0.0)), 1.0)
        hit["score"] = 0.75 * vector_component + 0.25 * lexical_component
        merged[key] = hit

    for hit in lexical_hits:
        key = f"{hit.get('source_file','')}::{hit.get('chunk_id','')}::{(hit.get('text') or '')[:120]}"
        lexical_component = min(float(hit.get("lexical_score", 0.0)), 1.0)
        vector_component = 0.0
        if key in merged:
            vector_component = max(
                0.0, 1.0 - min(float(merged[key].get("distance", 1.5)), 1.5) / 1.5
            )
        hit["score"] = 0.65 * vector_component + 0.35 * lexical_component
        if key not in merged or hit["score"] > merged[key].get("score", 0.0):
            merged[key] = hit

    merged_hits = list(merged.values())
    merged_hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)

    if merged_hits and merged_hits[0].get("score", 0.0) < 0.15:
        return []
    return merged_hits[:k]


# Answer generation with anti-hallucination measures
# Replace your existing generate_answer function with this one


def generate_answer(
    query: str,
    hits: list[dict],
    conversation_history: list[dict],
    client: OpenAI,
    is_multi_person: bool = False,
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
    recent_history = conversation_history[-MAX_CONTEXT_TURNS * 2 :]
    history_text = ""
    if recent_history:
        history_text = (
            "Previous conversation:\n"
            + "\n".join(
                [
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
                    for m in recent_history
                ]
            )
            + "\n\n"
        )

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
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,  # Lower temperature for more precise reasoning
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content


# ============================================================
# AGENT: Tools + run_agent (recruiter-impersonation)
# ============================================================
# Where agents fit in the flow:
# 1. User uploads PDFs, clicks Build → text extracted, chunked (semantic_chunk_resume),
#    embedded and stored in Chroma (build_collection). resume_metadata and collection live in session.
# 2. User sets "Set as me" on a profile → me_person is set.
# 3. When the user sends a chat message and me_person is set, we run the AGENT (run_agent)
#    instead of plain RAG. The agent is an LLM that can call tools:
#    - semantic_search: queries Chroma (retrieve_with_filters) over the user's resume chunks.
#    - web_search: Tavily for weather, news, etc.
#    - github_search: GitHub API for repos.
# 4. So "What did you do at Boeing?" → agent calls semantic_search("Boeing"); retrieval
#    returns matching chunks (now with hybrid: vector + keyword match); agent answers from those chunks.
# ============================================================

AGENT_SYSTEM_PROMPT_TEMPLATE = """You are [NAME]. You are in a live chat with a recruiter who is asking you questions as if in an informal screening or networking conversation. Your goal is to answer exactly as you would in real life: as yourself, based on your real resume, experience, and public presence.

RULES:
1. Identity: You are [NAME] and ONLY [NAME]. Never break character. Do not say you are an AI, assistant, or language model. Answer in first person as [NAME]. Never mention or use information about any other person (e.g. other candidates or names that are not you). If the search results or memory mention someone else, ignore that—only use information that is clearly about you, [NAME].
2. Ground answers in facts: Use the semantic_search tool to look up your resume whenever the recruiter asks about your background, skills, experience, education, or projects. The search returns only YOUR resume chunks (for [NAME]). Base your answer only on those results. For questions about a specific company or role (e.g. "What did you do at Boeing?"), call semantic_search with the company or employer name (e.g. "Boeing" or "experience at Boeing") so resume chunks about that employer are found. Only state things supported by the search results or by web/GitHub search when relevant. If semantic_search returns no matching excerpts, say honestly that you don't have that in your resume or the information you have.
3. Real-time and public info: Use web_search ONLY when the recruiter asks about the weather (e.g. "what's the weather in Boston?"). For weather questions, call web_search with a query like "weather in [city]". Do NOT use web_search for company info, news, or any other topic—for those, say you don't have that information or it's not available. For GitHub: first use semantic_search with queries like "GitHub" or "github username" or "profile link" to see if your resume lists your GitHub. Only if you find a GitHub username or profile URL in the search results, call github_search with that username (e.g. search_type "users" with query "username", or "repositories" with query "user:username" to get that user's repos). If your resume does not contain your GitHub username or link, do not guess—tell the recruiter honestly that you don't have your GitHub on your resume and they can ask you for it. If web_search returns that it is "not configured" or "not available", tell the recruiter: "Weather search isn't set up on this app yet. Add TAVILY_API_KEY to the .env file to enable weather (get a free key at https://tavily.com)."
4. Tone: Professional but personable. Be concise. Match the recruiter's level of formality. If you don't have information, say so honestly (e.g. "I don't have that on my resume" or "I'd need to check").
5. Consistency: Stay consistent with what you've already said in this conversation. If the recruiter refers back to something (e.g. "that project you mentioned"), align with your earlier answers.
6. Temporal grounding: Use the current date/time context below when using words like "recent", "currently", or "latest". Avoid calling old experiences "recent" when they are clearly not recent relative to today's date. Prefer explicit month/year phrasing when possible.
7. No hallucinations: Do not invent technologies, metrics, tools, architecture names, or project details. If a detail is not explicitly in resume context, omit it.

[CURRENT_DATETIME_CONTEXT]

[MEMORY_CONTEXT]"""


def _is_weather_query(query: str) -> bool:
    """Return True only if the query is clearly about weather (web search is restricted to weather only)."""
    if not query or not query.strip():
        return False
    q = query.lower().strip()
    weather_terms = [
        "weather",
        "temperature",
        "forecast",
        "temp ",
        " temp",
        "degrees",
        "rain",
        "snow",
        "sunny",
        "cloudy",
        "humid",
        "how hot",
        "how cold",
        "will it rain",
        "will it snow",
    ]
    return any(term in q for term in weather_terms)


def tool_web_search(query: str) -> str:
    """Web search via Tavily. Allowed only for weather queries; all other uses are blocked."""
    if not _is_weather_query(query):
        return "Web search is only allowed for weather queries. For other topics (news, company info, etc.) tell the recruiter you don't have that information or it's not available in this app."
    if not TAVILY_AVAILABLE:
        return "Web search is not available (tavily-python not installed or import failed). Tell the user: to enable weather and live info, install tavily-python and set TAVILY_API_KEY in .env (get a key at https://tavily.com)."
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip().strip("\"'")
    if not api_key:
        return "Web search is not configured (TAVILY_API_KEY not set). Tell the user: to enable weather and other live information, add TAVILY_API_KEY to the .env file. Get a free API key at https://tavily.com ."
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic", max_results=5)
        results = response.get("results", [])
        if not results:
            return "No web results found for that query."
        parts = []
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "")
            content = r.get("content", "")[:400]
            url = r.get("url", "")
            parts.append(f"{i}. {title}\n{content}\nSource: {url}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Web search error: {str(e)}"


def tool_semantic_search(
    query: str,
    collection,
    openai_client: OpenAI,
    me_person: Optional[str],
    k: int = 8,
) -> str:
    """Search over the user's resume (ChromaDB). Returns top-k chunks. Uses hybrid: vector + keyword match for company names."""
    if collection is None:
        return "Resume search is not available (no collection built)."
    if not me_person or resolve_target_person([me_person]) is None:
        return TARGET_RESUME_MISSING_ERROR
    if not query or not query.strip():
        return "Please provide a resume-related query."
    person_filter = me_person
    query_clean = query.strip()
    is_tech_query = _is_technology_query(query_clean)
    known_companies = _get_known_companies_for_person(me_person)
    mentioned_companies = _extract_company_mentions(query_clean, known_companies)

    # Broad profile/experience queries should include fuller role coverage, not just top-k semantic hits.
    if _is_broad_experience_query(query_clean):
        try:
            result = collection.get(
                where={"person_name": {"$eq": me_person}},
                include=["documents", "metadatas"],
            )
            docs = result.get("documents") or []
            metas = result.get("metadatas") or [{}] * len(docs)
            if docs:
                priority = {
                    "header": 0,
                    "experience": 1,
                    "education": 2,
                    "projects": 3,
                    "skills": 4,
                    "other": 5,
                }
                rows = []
                for idx, doc in enumerate(docs):
                    meta = metas[idx] if idx < len(metas) else {}
                    chunk_type = (meta.get("chunk_type") or "other").lower()
                    rows.append(
                        {
                            "text": doc or "",
                            "chunk_type": chunk_type,
                            "chunk_id": int(meta.get("chunk_id") or 0),
                            "priority": priority.get(chunk_type, 6),
                        }
                    )
                rows.sort(key=lambda r: (r["priority"], r["chunk_id"]))
                selected = rows[:40]
                parts = []
                for i, row in enumerate(selected, 1):
                    parts.append(f"[{i}] ({row['chunk_type']})\n{row['text'][:1200]}")
                return "\n\n---\n\n".join(parts)
        except Exception:
            pass

    # Vector retrieval
    hits = retrieve_with_filters(
        query_clean,
        collection,
        openai_client,
        k=k,
        person_filter=person_filter,
        ensure_all_candidates=False,
        known_names=(
            list(st.session_state.resume_metadata.keys())
            if hasattr(st, "session_state")
            and getattr(st.session_state, "resume_metadata", None)
            else None
        ),
    )
    # Company-aware keyword boost for targeted follow-ups (works for long queries too).
    if me_person and query_clean:
        try:
            # Get all chunks for this person and find any where key terms appear in text.
            result = collection.get(
                where={"person_name": {"$eq": me_person}},
                include=["documents", "metadatas"],
            )
            if result and result.get("documents") and result["documents"]:
                docs = result["documents"]
                metas = result.get("metadatas") or [{}] * len(docs)
                keyword_hits = []
                seen_texts = set()
                q_lower = query_clean.lower()
                query_tokens = [
                    t for t in re.findall(r"[a-zA-Z0-9+#.-]+", q_lower) if len(t) > 2
                ]
                text_needles = set()
                if len(query_clean.split()) <= 4:
                    text_needles.add(q_lower)
                for comp in mentioned_companies:
                    text_needles.add(comp.lower())
                # For technology questions, anchor on company + implementation hints.
                tech_needles = {
                    "technology",
                    "technologies",
                    "tools",
                    "framework",
                    "stack",
                    "implemented",
                    "built",
                    "using",
                }

                for i, doc in enumerate(docs):
                    if not doc:
                        continue
                    doc_lower = doc.lower()
                    meta = metas[i] if i < len(metas) else {}

                    company_match = (
                        any(comp.lower() in doc_lower for comp in mentioned_companies)
                        if mentioned_companies
                        else False
                    )
                    token_overlap = sum(1 for t in query_tokens if t in doc_lower)
                    direct_match = (
                        any(needle in doc_lower for needle in text_needles)
                        if text_needles
                        else False
                    )

                    keep = direct_match or token_overlap >= 2
                    if mentioned_companies:
                        keep = keep and company_match
                    if is_tech_query and keep:
                        keep = _doc_has_tech_signal(doc) or any(
                            t in doc_lower for t in tech_needles
                        )
                    if not keep:
                        continue

                    key = (doc[:200], meta.get("chunk_id"))
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    keyword_hits.append(
                        {
                            "text": doc,
                            "person_name": meta.get("person_name", me_person),
                            "chunk_type": meta.get("chunk_type", "other"),
                            "source_file": meta.get("source_file", ""),
                            "skills": meta.get("skills", ""),
                            "distance": 0.0,
                        }
                    )
                # Prepend keyword matches so they appear first; dedupe vector hits that are already in keyword_hits
                if keyword_hits:
                    # Prefer highly aligned chunks first.
                    if mentioned_companies:
                        keyword_hits.sort(
                            key=lambda h: (
                                (
                                    0
                                    if any(
                                        c.lower() in h["text"].lower()
                                        for c in mentioned_companies
                                    )
                                    else 1
                                ),
                                (
                                    0
                                    if (
                                        _doc_has_tech_signal(h["text"])
                                        if is_tech_query
                                        else False
                                    )
                                    else 1
                                ),
                                len(h["text"]),
                            ),
                            reverse=False,
                        )
                    vector_texts = {h["text"][:200] for h in hits}
                    keyword_only = [
                        h for h in keyword_hits if h["text"][:200] not in vector_texts
                    ]
                    hits = keyword_only + [
                        h
                        for h in hits
                        if h["text"][:200]
                        not in {x["text"][:200] for x in keyword_only}
                    ]
                    hits = hits[: max(k + len(keyword_only), k)]
        except Exception:
            pass
    if not hits:
        return "No matching resume excerpts found for that query."
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] ({h.get('chunk_type', 'other')})\n{h['text'][:1200]}")
    return "\n\n---\n\n".join(parts)


def _extract_github_handles_from_text(text: str) -> set[str]:
    if not text:
        return set()
    handles = set()
    patterns = [
        r"github\.com/([A-Za-z0-9-]{1,39})",
        r"github\s*(?:\.|dot)?\s*com\s*/\s*([A-Za-z0-9-]{1,39})",
        r"github\s*[:\-]\s*([A-Za-z0-9-]{1,39})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            handle = (match or "").strip().strip("/").lower()
            if handle and handle not in {
                "features",
                "topics",
                "marketplace",
                "orgs",
                "pricing",
            }:
                handles.add(handle)
    return handles


def _select_allowed_github_handle(
    query: str, allowed_handles: list[str]
) -> Optional[str]:
    if not allowed_handles:
        return None
    q = (query or "").lower()
    normalized = sorted({h.lower() for h in allowed_handles if h})
    for handle in normalized:
        if handle in q:
            return handle
    return normalized[0] if normalized else None


def _is_github_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        term in q
        for term in ["github", "git hub", "gihtub", "repo", "repos", "repository"]
    )


def _wants_repo_examples(query: str) -> bool:
    q = (query or "").lower()
    return any(
        term in q
        for term in [
            "repo",
            "repos",
            "repository",
            "repositories",
            "project",
            "projects",
            "examples",
        ]
    )


def _is_github_error_result(text: str) -> bool:
    t = (text or "").lower()
    return (
        t.startswith("github api error")
        or t.startswith("github search error")
        or t.startswith("no github results found")
    )


def _is_broad_experience_query(query: str) -> bool:
    q = (query or "").lower().strip()
    patterns = [
        "tell me about yourself",
        "introduce yourself",
        "work experience",
        "more details into your work experience",
        "details into your work experience",
        "walk me through your background",
        "background",
        "professional experience",
        "career summary",
    ]
    return any(p in q for p in patterns)


def _is_detailed_experience_query(query: str) -> bool:
    q = (query or "").lower()
    markers = [
        "work experience",
        "details into your work experience",
        "give me details",
        "more details",
        "tell me more about your experience",
    ]
    return any(m in q for m in markers)


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _extract_experience_detail_lines(text: str, max_lines: int = 2) -> list[str]:
    out: list[str] = []
    lines = (text or "").splitlines()
    skip_first = True
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if skip_first:
            skip_first = False
            continue
        clean = re.sub(r"^[•\-\*\u2022\u25aa\u25cf]+\s*", "", line).strip()
        if not clean:
            continue
        # Skip standalone date/location lines to keep detail lines substantive.
        if re.fullmatch(
            r"[A-Za-z]{3,}\.?\s+\d{4}\s*[-–—to]+\s*(?:[A-Za-z]{3,}\.?\s+\d{4}|Present|Current|\d{4})",
            clean,
        ):
            continue
        if clean not in out:
            out.append(clean)
        if len(out) >= max_lines:
            break
    return out


def _collect_experience_entries(collection, me_person: str) -> list[dict]:
    """
    Deterministically gather all distinct experience-like entries from indexed chunks.
    """
    if collection is None or not me_person:
        return []
    try:
        result = collection.get(
            where={"person_name": {"$eq": me_person}},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(docs)
        date_range_pattern = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to]+\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})|\b\d{4}\s*[-–—to]+\s*(?:Present|Current|\d{4})"
        role_terms = [
            "intern",
            "engineer",
            "assistant",
            "co-op",
            "teaching assistant",
            "research assistant",
            "developer",
            "analyst",
            "specialist",
        ]
        non_experience_heading_terms = [
            "gpa",
            "master of",
            "bachelor of",
            "education",
            "university",
            "project",
            "patent",
            "skills",
            "certification",
            "coursework",
            "technical skills",
            "publications",
            "achievements",
        ]
        generic_section_headers = {
            "experience",
            "work experience",
            "professional experience",
            "employment history",
        }

        rows: list[dict] = []
        for idx, doc in enumerate(docs):
            text = (doc or "").strip()
            if not text:
                continue
            meta = metas[idx] if idx < len(metas) else {}
            chunk_type = (meta.get("chunk_type") or "other").lower()
            text_lower = text.lower()
            heading = _first_nonempty_line(text)
            if not heading:
                continue
            heading_lower = heading.lower()
            if heading_lower in generic_section_headers:
                continue
            if any(term in heading_lower for term in non_experience_heading_terms):
                continue

            has_date = bool(re.search(date_range_pattern, text, flags=re.IGNORECASE))
            has_role_term = any(term in heading_lower for term in role_terms) or any(
                term in text_lower for term in role_terms
            )
            looks_like_experience = (
                chunk_type == "experience" and has_date
            ) or has_role_term
            if not looks_like_experience:
                continue
            date_match = re.search(date_range_pattern, text, flags=re.IGNORECASE)
            date_range = date_match.group(0) if date_match else ""
            sort_year = 0
            year_matches = re.findall(r"\b(19|20)\d{2}\b", date_range)
            if year_matches:
                try:
                    sort_year = max(
                        int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", date_range)
                    )
                except Exception:
                    sort_year = 0

            rows.append(
                {
                    "heading": heading,
                    "date_range": date_range,
                    "details": _extract_experience_detail_lines(text, max_lines=2),
                    "chunk_type": chunk_type,
                    "chunk_id": int(meta.get("chunk_id") or 0),
                    "sort_year": sort_year,
                }
            )

        dedup: dict[str, dict] = {}
        for row in rows:
            key = _normalize_name(row["heading"])
            if key not in dedup:
                dedup[key] = row
            else:
                # Prefer richer entries (more details, has date).
                prev = dedup[key]
                prev_score = (1 if prev["date_range"] else 0) + len(prev["details"])
                cur_score = (1 if row["date_range"] else 0) + len(row["details"])
                if cur_score > prev_score:
                    dedup[key] = row

        entries = list(dedup.values())
        entries.sort(
            key=lambda r: (
                r["sort_year"],
                1 if r["chunk_type"] == "experience" else 0,
                -r["chunk_id"],
            ),
            reverse=True,
        )
        return entries[:12]
    except Exception:
        return []


def _format_experience_role_checklist(entries: list[dict], max_roles: int = 8) -> str:
    if not entries:
        return ""
    lines = [
        "Expected experience roles from resume (cover these when giving broad summaries):"
    ]
    for e in entries[:max_roles]:
        title = e.get("heading", "Experience")
        date_range = e.get("date_range", "")
        if date_range and date_range.lower() not in title.lower():
            lines.append(f"- {title} ({date_range})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def _is_technology_query(query: str) -> bool:
    q = (query or "").lower()
    terms = [
        "technology",
        "technologies",
        "tech stack",
        "stack",
        "tools",
        "framework",
        "frameworks",
        "languages",
        "used",
        "built with",
        "implemented with",
    ]
    return any(t in q for t in terms)


def _get_known_companies_for_person(me_person: Optional[str]) -> list[str]:
    if not me_person:
        return []
    try:
        meta = st.session_state.resume_metadata.get(me_person)
        if not meta:
            return []
        companies = meta.companies or []
        cleaned = []
        for c in companies:
            c = (c or "").strip()
            if len(c) >= 2:
                cleaned.append(c)
        return cleaned
    except Exception:
        return []


def _extract_company_mentions(query: str, companies: list[str]) -> list[str]:
    q = (query or "").lower()
    hits = []
    for company in companies:
        c = company.lower().strip()
        if not c:
            continue
        if re.search(rf"\b{re.escape(c)}\b", q):
            hits.append(company)
    return hits


def _doc_has_tech_signal(text: str) -> bool:
    t = (text or "").lower()
    tech_terms = [
        "python",
        "java",
        "kotlin",
        "typescript",
        "javascript",
        "react",
        "spring",
        "node",
        "aws",
        "docker",
        "kubernetes",
        "sql",
        "nosql",
        "flask",
        "django",
        "framework",
        "stack",
        "built",
        "implemented",
        "designed",
    ]
    return any(term in t for term in tech_terms)


def _get_allowed_github_handles(collection, me_person: Optional[str]) -> list[str]:
    if collection is None or not me_person:
        return []
    try:
        result = collection.get(
            where={"person_name": {"$eq": me_person}},
            include=["documents"],
        )
        docs = result.get("documents") or []
        handles = set()
        for doc in docs:
            handles.update(_extract_github_handles_from_text(doc or ""))
        return sorted(handles)
    except Exception:
        return []


def tool_github_search(
    query: str,
    search_type: str = "repositories",
    allowed_handles: Optional[list[str]] = None,
) -> str:
    """GitHub search (repositories or users). Requires GITHUB_TOKEN for higher rate limits."""
    if not query or not query.strip():
        return "GitHub search requires a non-empty query."

    allowed = {(h or "").lower() for h in (allowed_handles or []) if (h or "").strip()}
    if allowed:
        q_lower = query.lower()
        if not any(h in q_lower for h in allowed):
            selected = _select_allowed_github_handle(query, sorted(allowed))
            if selected:
                query = selected if search_type == "users" else f"user:{selected}"
            else:
                return "I can only search GitHub using a username that appears in your resume."

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    base = "https://api.github.com/search"
    if search_type == "users":
        url = f"{base}/users"
    else:
        url = f"{base}/repositories"
    try:
        r = requests.get(url, params={"q": query}, headers=headers, timeout=15)
        if r.status_code != 200:
            return f"GitHub API error: {r.status_code} - {r.text[:300]}"
        data = r.json()
        items = data.get("items", [])[:5]
        if not items:
            return "No GitHub results found for that query."
        parts = []
        for i, item in enumerate(items, 1):
            if search_type == "users":
                login = item.get("login", "")
                name = item.get("name") or login
                bio = (item.get("bio") or "")[:200]
                parts.append(f"{i}. {name} (@{login})\n{bio}")
            else:
                full_name = item.get("full_name", "")
                desc = (item.get("description") or "")[:200]
                url = item.get("html_url", "")
                parts.append(f"{i}. {full_name}\n{desc}\n{url}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"GitHub search error: {str(e)}"


# OpenAI tool definitions for the agent
AGENT_TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for WEATHER ONLY. Use only when the recruiter asks about weather (e.g. 'what's the weather in X?'). Pass a query like 'weather in Boston'. Do not use for news, company info, or any other topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search your resume / profile for your background, skills, experience, education, projects. Use whenever the recruiter asks about you. For questions about a specific company or role (e.g. 'What did you do at Boeing?'), use the company or employer name as the query (e.g. 'Boeing' or 'experience at Boeing') to retrieve relevant resume chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query or company/employer name (e.g. 'Boeing', 'experience at X')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search",
            "description": "Search GitHub for repositories or users. Only use when you have found the candidate's GitHub username or profile URL in semantic_search results (e.g. from resume). Pass that username as query; use search_type 'users' to find the profile or 'repositories' with query 'user:username' for their repos. Do not call this with a guessed or generic query if the resume does not mention GitHub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "search_type": {
                        "type": "string",
                        "enum": ["repositories", "users"],
                        "description": "Type of search",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _load_agent_memories() -> list[dict]:
    """Load persistent memory from disk with backward compatibility."""
    if not AGENT_MEMORY_FILE.exists():
        return []
    try:
        with open(AGENT_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = (
            data.get("memories", [])
            if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )
        out: list[dict] = []
        for item in raw:
            if isinstance(item, dict) and "fact" in item:
                fact = str(item.get("fact", "")).strip()
                if not fact:
                    continue
                fact_hash = (
                    item.get("fact_hash")
                    or hashlib.sha1(fact.lower().encode("utf-8")).hexdigest()[:16]
                )
                out.append(
                    {
                        "fact": fact,
                        "person": item.get("person"),
                        "fact_hash": fact_hash,
                        "source": item.get("source", "legacy"),
                        "created_at": item.get(
                            "created_at", datetime.now(timezone.utc).isoformat()
                        ),
                        "memory_type": item.get("memory_type", "fact"),
                        "value": item.get("value"),
                        "confidence": item.get("confidence", 1.0),
                    }
                )
            elif isinstance(item, str):
                fact = item.strip()
                if fact:
                    out.append(
                        {
                            "fact": fact,
                            "person": None,
                            "fact_hash": hashlib.sha1(
                                fact.lower().encode("utf-8")
                            ).hexdigest()[:16],
                            "source": "legacy",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "memory_type": "fact",
                            "value": None,
                            "confidence": 1.0,
                        }
                    )
        return out
    except Exception:
        return []


def _save_agent_memories(memories: list[dict]) -> None:
    """Persist memory list to disk."""
    try:
        with open(AGENT_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"memories": memories}, f, indent=2)
    except Exception:
        pass


def _normalize_fact_for_hash(fact: str) -> str:
    return re.sub(r"\s+", " ", (fact or "").strip().lower())


def _compact_memories(mems: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for m in mems:
        key = (m.get("person"), m.get("memory_type", "fact"), m.get("fact_hash"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    # Cap memories per person
    grouped: dict[Optional[str], list[dict]] = {}
    for m in deduped:
        grouped.setdefault(m.get("person"), []).append(m)

    kept = []
    for person, items in grouped.items():
        items.sort(key=lambda x: x.get("created_at", ""))
        kept.extend(items[-MAX_MEMORIES_PER_PERSON:])

    kept.sort(key=lambda x: x.get("created_at", ""))
    return kept[-MAX_TOTAL_MEMORIES:]


def add_memory(
    fact: str,
    person: Optional[str] = None,
    source: str = "recruiter_context",
    memory_type: str = "fact",
    value: Optional[dict] = None,
    confidence: float = 1.0,
) -> None:
    """Append recruiter-context memory, deduped and size-bounded."""
    value = value or {}
    if (not fact or not fact.strip()) and not value:
        return

    normalized_fact = _normalize_fact_for_hash(fact or "")
    value_blob = json.dumps(value, sort_keys=True) if value else ""
    hash_src = f"{memory_type}|{normalized_fact}|{value_blob}"
    fact_hash = hashlib.sha1(hash_src.encode("utf-8")).hexdigest()[:16]
    entry = {
        "fact": (fact or "").strip(),
        "person": person,
        "fact_hash": fact_hash,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "memory_type": memory_type or "fact",
        "value": value or None,
        "confidence": float(confidence),
    }

    mems = st.session_state.get("agent_memories", [])
    mems.append(entry)
    mems = _compact_memories(mems)
    st.session_state.agent_memories = mems
    _save_agent_memories(mems)


def get_recent_memories(limit: int = 10, me_person: Optional[str] = None) -> str:
    """Return recent persistent memory entries for the agent. When me_person is set, only memories scoped to that person are included (avoids mixing different candidates)."""
    mems: list[dict] = st.session_state.get("agent_memories", [])
    if me_person:
        mems = [m for m in mems if isinstance(m, dict) and m.get("person") == me_person]
    else:
        mems = [m for m in mems if isinstance(m, dict)]
    mems = [m for m in mems if m.get("source") == "recruiter_context"]
    mems = mems[-limit:]
    if not mems:
        return ""
    facts = [m["fact"] if isinstance(m, dict) else str(m) for m in mems]
    return "Remembered context:\n" + "\n".join(f"- {f}" for f in facts)


def _get_structured_memories(
    me_person: Optional[str],
    memory_type: Optional[str] = None,
    source: str = "recruiter_context",
) -> list[dict]:
    mems: list[dict] = st.session_state.get("agent_memories", [])
    out = []
    for m in mems:
        if not isinstance(m, dict):
            continue
        if me_person and m.get("person") != me_person:
            continue
        if source and m.get("source") != source:
            continue
        if memory_type and m.get("memory_type") != memory_type:
            continue
        out.append(m)
    out.sort(key=lambda x: x.get("created_at", ""))
    return out


def store_recruiter_context_from_query(
    query: str, me_person: str, client: OpenAI
) -> tuple[bool, str]:
    fields = _extract_recruiter_context_fields(query, client)
    recruiter_name = (fields.get("recruiter_name") or "").strip() or None
    recruiter_company = (fields.get("recruiter_company") or "").strip() or None
    recruiter_name, recruiter_company = _cleanup_recruiter_identity_fields(
        recruiter_name, recruiter_company
    )
    followup_day = (fields.get("followup_day") or "").strip() or None
    followup_time = (fields.get("followup_time") or "").strip() or None
    followup_timezone = (fields.get("followup_timezone") or "").strip() or None
    display_day, display_time, display_tz = _normalize_followup_display_parts(
        followup_day, followup_time, followup_timezone
    )

    stored = False
    if recruiter_name:
        add_memory(
            fact=f"Recruiter name: {recruiter_name}",
            person=me_person,
            source="recruiter_context",
            memory_type="recruiter_identity",
            value={"recruiter_name": recruiter_name},
            confidence=1.0,
        )
        stored = True
    if recruiter_company:
        add_memory(
            fact=f"Recruiter company: {recruiter_company}",
            person=me_person,
            source="recruiter_context",
            memory_type="recruiter_identity",
            value={"recruiter_company": recruiter_company},
            confidence=1.0,
        )
        stored = True
    if followup_day or followup_time or followup_timezone:
        payload = {
            "followup_day": followup_day,
            "followup_time": followup_time,
            "followup_timezone": followup_timezone,
        }
        parts = [p for p in [display_day, display_time, display_tz] if p]
        add_memory(
            fact=f"Follow-up agreed: {' '.join(parts)}".strip(),
            person=me_person,
            source="recruiter_context",
            memory_type="followup",
            value=payload,
            confidence=1.0,
        )
        stored = True

    if not stored:
        return False, ""

    ack_bits = []
    if recruiter_name:
        ack_bits.append(f"Nice to meet you, {recruiter_name}.")
    elif recruiter_company:
        ack_bits.append("Nice to meet you.")

    if followup_day or followup_time:
        time_parts = [p for p in [display_day, display_time, display_tz] if p]
        ack_bits.append(f"Noted - we'll follow up on {' '.join(time_parts)}.")
    elif "follow up" in (query or "").lower():
        ack_bits.append("Noted - follow-up details saved.")

    return True, " ".join(ack_bits).strip() or "Got it - I've noted that context."


def answer_recruiter_memory_recall(query: str, me_person: str) -> Optional[str]:
    intent = _detect_memory_recall_intent(query)
    if not intent:
        return None

    followups = _get_structured_memories(me_person, memory_type="followup")
    identities = _get_structured_memories(me_person, memory_type="recruiter_identity")

    def latest_value(entries: list[dict], key: str) -> Optional[str]:
        for m in reversed(entries):
            value = m.get("value") if isinstance(m.get("value"), dict) else {}
            v = (value.get(key) if isinstance(value, dict) else None) or ""
            v = str(v).strip()
            if v:
                return v
        return None

    if intent in {"followup_time", "followup"}:
        day = latest_value(followups, "followup_day")
        time = latest_value(followups, "followup_time")
        tz = latest_value(followups, "followup_timezone")
        day, time, tz = _normalize_followup_display_parts(day, time, tz)
        if not day and not time:
            return "I don't have a stored follow-up time yet."
        parts = [p for p in [day, time, tz] if p]
        return f"We agreed to follow up on {' '.join(parts)}."

    if intent == "recruiter_name":
        name = latest_value(identities, "recruiter_name")
        if not name:
            return "I don't have your name saved yet."
        return f"You're {name}."

    if intent == "recruiter_company":
        company = latest_value(identities, "recruiter_company")
        if not company:
            return "I don't have your company saved yet."
        return f"You're from {company}."

    return None


def summarize_and_store(
    user_msg: str, assistant_msg: str, client: OpenAI, me_person: Optional[str] = None
) -> None:
    """Extract only recruiter-context facts (not resume facts) and persist them."""
    if not me_person:
        return
    prompt = """Extract up to 3 short recruiter-context facts to remember for later.

INCLUDE ONLY:
- recruiter name/title/company if explicitly stated by the recruiter
- scheduling preferences or follow-up commitments
- interview logistics or role requirements

DO NOT INCLUDE:
- candidate resume facts (skills, education, projects, work history)
- anything inferred or guessed

Recruiter message: {user}

Return JSON: {{"facts": ["fact1", "fact2"]}}"""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt.format(user=user_msg[:700])}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        data = json.loads(text.replace("```json", "").replace("```", "").strip())
        for fact in data.get("facts", [])[:3]:
            if isinstance(fact, str) and fact.strip():
                add_memory(fact.strip(), person=me_person, source="recruiter_context")
    except Exception:
        pass


def _response_mentions_other_candidate(
    answer: str, me_person: str, known_names: list[str]
) -> bool:
    if not answer or not me_person:
        return False
    answer_lower = answer.lower()
    for name in known_names:
        if not name or name == me_person:
            continue
        name_lower = name.lower()
        if len(name_lower) < 3:
            continue
        if re.search(rf"\b{re.escape(name_lower)}\b", answer_lower):
            return True
    return False


def _enforce_agent_response(answer: str, me_person: str, known_names: list[str]) -> str:
    if not answer or not answer.strip():
        return "I don't have enough information from my resume to answer that."
    if _response_mentions_other_candidate(answer, me_person, known_names):
        return "I can only speak about my own resume and experience."
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", answer)]
    current_year = datetime.now().astimezone().year
    if years and max(years) <= current_year - 2:
        answer = re.sub(r"\brecently\b", "previously", answer, flags=re.IGNORECASE)
    return answer


def _ground_answer_with_semantic_context(
    answer: str,
    semantic_context_blocks: list[str],
    user_message: str,
    me_person: str,
    client: OpenAI,
) -> str:
    """
    Rewrite an answer so every factual claim is supported by retrieved resume context.
    Unsupported claims are removed instead of guessed.
    """
    if not answer or not answer.strip() or not semantic_context_blocks:
        return answer
    context = "\n\n---\n\n".join(semantic_context_blocks[-4:])
    prompt = """You are a strict factual editor for a resume assistant.

Candidate: {me_person}
User question: {user_message}

Task:
1. Keep only claims supported by the provided context.
2. Remove unsupported numbers, technologies, metrics, and details.
3. If a specific detail is uncertain, omit it instead of guessing.
4. Keep first-person voice.
5. Keep the response concise and coherent.
6. Do not use speculative phrases like "might have", "likely", or "probably" about technologies.

Context:
{context}

Original answer:
{answer}

Return only the corrected answer text."""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(
                        me_person=me_person,
                        user_message=user_message[:500],
                        context=context[:12000],
                        answer=answer[:4000],
                    ),
                }
            ],
            temperature=0,
            max_tokens=900,
        )
        corrected = (resp.choices[0].message.content or "").strip()
        return corrected or answer
    except Exception:
        return answer


def _is_self_summary_query(query: str) -> bool:
    q = (query or "").lower().strip()
    if not q:
        return False
    patterns = [
        r"\btell me about yourself\b",
        r"\bintroduce yourself\b",
        r"\bgive me an intro\b",
        r"\bwalk me through your background\b",
        r"\bsummar(y|ize) your background\b",
        r"\bwho are you\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _build_profile_context_for_intro(
    collection, me_person: str, max_chunks: int = 40
) -> str:
    """
    Build a broad, deterministic profile context so intro questions cover all major roles.
    """
    if collection is None or not me_person:
        return ""
    try:
        result = collection.get(
            where={"person_name": {"$eq": me_person}},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(docs)
        if not docs:
            return ""

        type_priority = {
            "header": 0,
            "experience": 1,
            "education": 2,
            "projects": 3,
            "skills": 4,
            "other": 5,
        }

        rows = []
        for idx, doc in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            chunk_type = (meta.get("chunk_type") or "other").lower()
            rows.append(
                {
                    "text": doc or "",
                    "chunk_type": chunk_type,
                    "chunk_id": int(meta.get("chunk_id") or 0),
                    "priority": type_priority.get(chunk_type, 6),
                }
            )

        rows.sort(key=lambda r: (r["priority"], r["chunk_id"]))
        experience_rows = [r for r in rows if r["chunk_type"] == "experience"]
        non_experience_rows = [r for r in rows if r["chunk_type"] != "experience"]
        selected = experience_rows[:25] + non_experience_rows[: max(0, max_chunks - 25)]
        selected = selected[:max_chunks]
        parts = []
        for i, row in enumerate(selected, 1):
            parts.append(
                f"[Intro Source {i}: {row['chunk_type']}]\n{row['text'][:700]}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""


def run_agent(
    user_message: str,
    conversation_history: list[dict],
    memory_context: str,
    client: OpenAI,
    collection,
    me_person: Optional[str],
) -> tuple[str, list[str]]:
    """
    Run the recruiter-impersonation agent with tool calling.
    Returns (final_answer, list of tool names used).
    """
    if not me_person or resolve_target_person([me_person]) is None:
        return TARGET_RESUME_MISSING_ERROR, []

    name = me_person
    system_content = AGENT_SYSTEM_PROMPT_TEMPLATE.replace("[NAME]", name)
    system_content = system_content.replace(
        "[CURRENT_DATETIME_CONTEXT]", get_current_datetime_context()
    )
    system_content = system_content.replace(
        "[MEMORY_CONTEXT]",
        memory_context if memory_context else "(No additional memory for this turn.)",
    )
    known_names = (
        list(st.session_state.resume_metadata.keys())
        if "resume_metadata" in st.session_state
        else []
    )
    allowed_github_handles = _get_allowed_github_handles(collection, me_person)

    # Deterministic GitHub path: avoids model missing the username in tool args.
    if _is_github_query(user_message):
        if not allowed_github_handles:
            return (
                "I can't pull GitHub projects because I couldn't find a GitHub username/link in my indexed resume.",
                ["github_search"],
            )
        handle = (
            _select_allowed_github_handle(user_message, allowed_github_handles)
            or allowed_github_handles[0]
        )
        if _wants_repo_examples(user_message):
            repos_result = tool_github_search(
                query=f"user:{handle}",
                search_type="repositories",
                allowed_handles=allowed_github_handles,
            )
            if _is_github_error_result(repos_result):
                return (
                    f"My GitHub is https://github.com/{handle}, but I couldn't fetch repository details right now.",
                    ["github_search"],
                )
            return f"Here are examples from my GitHub (@{handle}):\n\n{repos_result}", [
                "github_search"
            ]
        profile_result = tool_github_search(
            query=handle,
            search_type="users",
            allowed_handles=allowed_github_handles,
        )
        if _is_github_error_result(profile_result):
            return f"My GitHub is https://github.com/{handle}.", ["github_search"]
        return f"My GitHub is https://github.com/{handle}.\n\n{profile_result}", [
            "github_search"
        ]

    effective_query = _augment_followup_query_with_recent_subject(
        user_message, conversation_history
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]
    tools_used: list[str] = []
    semantic_context_blocks: list[str] = []

    # Deterministic semantic prefetch for resume questions so retrieval is always available.
    if not _is_weather_query(user_message):
        prefetch_k = 16 if _is_broad_experience_query(user_message) else 10
        prefetch = tool_semantic_search(
            effective_query,
            collection,
            client,
            me_person,
            k=prefetch_k,
        )
        if not prefetch.startswith("No matching resume excerpts found"):
            semantic_context_blocks.append(prefetch)
            tools_used.append("semantic_search")
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Prefetched resume excerpts for this question. "
                        "Prioritize these grounded facts and do not invent missing details.\n\n"
                        f"{prefetch[:12000]}"
                    ),
                }
            )

        # For technology-specific questions, run a second targeted prefetch to capture stack details.
        if _is_technology_query(effective_query):
            tech_prefetch_query = (
                f"{effective_query} technologies tools tech stack frameworks "
                f"languages implementation built using"
            )
            tech_prefetch = tool_semantic_search(
                tech_prefetch_query,
                collection,
                client,
                me_person,
                k=max(prefetch_k, 14),
            )
            if (
                tech_prefetch
                and not tech_prefetch.startswith("No matching resume excerpts found")
                and tech_prefetch not in semantic_context_blocks
            ):
                semantic_context_blocks.append(tech_prefetch)
                tools_used.append("semantic_search")
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Additional targeted excerpts for technology/tooling details:\n\n"
                            f"{tech_prefetch[:12000]}"
                        ),
                    }
                )

    if _is_self_summary_query(user_message):
        intro_context = _build_profile_context_for_intro(collection, me_person)
        if intro_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use this prefetched resume context for a complete intro. "
                        "Mention major internships/roles with dates and avoid saying older roles are 'recent'.\n\n"
                        f"{intro_context}"
                    ),
                }
            )
    if _is_self_summary_query(user_message) or _is_detailed_experience_query(
        user_message
    ):
        entries = _collect_experience_entries(collection, me_person)
        checklist = _format_experience_role_checklist(entries)
        if checklist:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"{checklist}\n"
                        "For broad prompts like 'tell me about yourself' or 'work experience', include each role briefly "
                        "if supported by context. Keep a natural conversational style."
                    ),
                }
            )
    for m in conversation_history[-18:]:
        role = m.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content[:1200]})
    messages.append({"role": "user", "content": user_message})

    max_rounds = 10
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=messages,
            tools=AGENT_TOOLS_DEF,
            tool_choice="auto",
            temperature=0,
        )
        choice = resp.choices[0]
        if not choice.message.tool_calls:
            final_answer = choice.message.content or ""
            final_answer = _ground_answer_with_semantic_context(
                final_answer,
                semantic_context_blocks,
                user_message,
                me_person,
                client,
            )
            final_answer = _enforce_agent_response(final_answer, me_person, known_names)
            return final_answer, tools_used

        # Append assistant message with tool_calls once
        assistant_msg = choice.message
        messages.append(
            {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            }
        )
        for tc in assistant_msg.tool_calls:
            name_tc = tc.function.name
            try:
                args = (
                    json.loads(tc.function.arguments) if tc.function.arguments else {}
                )
            except json.JSONDecodeError:
                args = {}
            tools_used.append(name_tc)

            if name_tc not in {"web_search", "semantic_search", "github_search"}:
                result = "This tool is not enabled."
            elif name_tc == "web_search":
                result = tool_web_search(args.get("query", ""))
            elif name_tc == "semantic_search":
                tool_query = args.get("query", "")
                tool_query = _augment_followup_query_with_recent_subject(
                    tool_query, conversation_history
                )
                result = tool_semantic_search(
                    tool_query,
                    collection,
                    client,
                    me_person,
                    k=8,
                )
                semantic_context_blocks.append(result)
            elif name_tc == "github_search":
                if not allowed_github_handles:
                    result = "I can't use GitHub search because my resume does not include a GitHub username or profile URL."
                else:
                    result = tool_github_search(
                        args.get("query", args.get("q", "")),
                        args.get("search_type", "repositories"),
                        allowed_handles=allowed_github_handles,
                    )
            else:
                result = "This tool is not enabled."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:8000],
                }
            )

    return (
        "I don't have enough information from my resume to answer that. Please try rephrasing.",
        tools_used,
    )


# Streamlit UI
# st.title("📄 Multi-Resume RAG Chatbot")
# st.markdown("Upload multiple resumes to search and query candidate information.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_chroma_client()
if "collection" not in st.session_state:
    st.session_state.collection = None
if "resume_metadata" not in st.session_state:
    st.session_state.resume_metadata = {}
if "me_person" not in st.session_state:
    st.session_state.me_person = None
if "agent_memories" not in st.session_state:
    st.session_state.agent_memories = _compact_memories(_load_agent_memories())
if "candidate_registry" not in st.session_state:
    st.session_state.candidate_registry = {}

# Sidebar
with st.sidebar:
    # Logo/Brand section
    st.markdown(
        """
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🎯</span>
        <h2 style="margin: 0.5rem 0; font-size: 1.4rem;">ResumeAI</h2>
        <p style="font-size: 0.85rem; opacity: 0.8;">Recruiter chat simulation</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please provide an OpenAI API key")
        st.stop()

    client = OpenAI(api_key=api_key)

    st.markdown("### 📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop PDF resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="resume_uploader",
    )

    if uploaded_files and len(uploaded_files) > 50:
        st.warning("Max 50 files allowed. Using first 50.")
        uploaded_files = uploaded_files[:50]

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("🔨 Build", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.resume_metadata = {}
            st.session_state.collection = None
            st.session_state.all_chunks = []
            st.session_state.me_person = None
            st.session_state.candidate_registry = {}
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### 👤 Recruiter Mode")
    st.caption(f"Chat is fixed to: **{TARGET_PERSON}**")
    if st.session_state.resume_metadata:
        loaded_names = list(st.session_state.resume_metadata.keys())
        resolved_target = resolve_target_person(loaded_names)
        if resolved_target:
            st.success(f"Target resume loaded as: {resolved_target}")
        else:
            st.error(TARGET_RESUME_MISSING_ERROR)
        st.caption(f"{len(loaded_names)} candidate resume(s) indexed")
    else:
        st.caption("Upload resume PDFs and click **Build**.")

# Main Header (recruiter POV: who they're chatting with)
resolved_target_name = resolve_target_person(
    list(st.session_state.resume_metadata.keys())
)
if resolved_target_name:
    st.session_state.me_person = resolved_target_name
else:
    st.session_state.me_person = None
me_name = st.session_state.me_person
if me_name:
    st.markdown(
        f"""
    <div class="main-header">
        <h1>🎯 ResumeAI</h1>
        <p>Chatting with <strong>{me_name}</strong> — ask anything you'd ask in a screening or networking call.</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">The candidate answers as themselves using their resume.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
    <div class="main-header">
        <h1>🎯 ResumeAI</h1>
        <p><strong>Recruiter mode requires Sanjeev's resume.</strong> Upload resumes and click Build.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Stats row
num_candidates = len(st.session_state.resume_metadata)
num_chunks = len(st.session_state.get("all_chunks", []))
num_messages = len(st.session_state.messages)

st.markdown(
    f"""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-number">{num_candidates}</div>
        <div class="stat-label">Candidates</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{num_chunks}</div>
        <div class="stat-label">Indexed Chunks</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{num_messages // 2}</div>
        <div class="stat-label">Questions Asked</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{(TARGET_PERSON[:12] + "…") if len(TARGET_PERSON) > 12 else TARGET_PERSON}</div>
        <div class="stat-label">Target Persona</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Build vector store
if build_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
    else:
        # Reset state for deterministic rebuilds.
        st.session_state.collection = None
        st.session_state.resume_metadata = {}
        st.session_state.candidate_registry = {}
        st.session_state.all_chunks = []
        st.session_state.me_person = None

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
                    rejected_files.append(
                        (pdf_file.name, errors[0] if errors else "Unknown error")
                    )
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
                    rejected_files.append(
                        (
                            pdf_file.name,
                            errors[0] if errors else "Content validation failed",
                        )
                    )
                    progress.progress((i + 1) / len(uploaded_files))
                    continue

                for warn in warnings:
                    status.write(f"⚠️ {pdf_file.name}: {warn}")

                # Sanitize text before processing
                text = sanitize_text_for_llm(text)
                # ========================================

                # DEBUG: Show extracted text length
                status.write(
                    f"   📝 Extracted {len(text)} chars, {len(text.split())} words"
                )

                # Extract metadata using LLM
                metadata = extract_resume_metadata(text, pdf_file.name, client)
                if metadata.person_name == "Unknown" or metadata.person_name is None:
                    status.write(
                        f"⚠️ {pdf_file.name}: Could not extract person name - skipping"
                    )
                    rejected_count += 1
                    rejected_files.append(
                        (pdf_file.name, "Could not identify person name in document")
                    )
                    progress.progress((i + 1) / len(uploaded_files))
                    continue

                original_name = metadata.person_name.strip()
                display_name = original_name
                if display_name in st.session_state.resume_metadata:
                    stem = Path(pdf_file.name).stem[:24]
                    display_name = f"{original_name} ({stem})"
                metadata.person_name = display_name

                base_candidate_id = _make_candidate_id(display_name, pdf_file.name)
                candidate_id = base_candidate_id
                suffix = 1
                while candidate_id in st.session_state.candidate_registry:
                    suffix += 1
                    candidate_id = f"{base_candidate_id}_{suffix}"
                metadata.candidate_id = candidate_id

                st.session_state.resume_metadata[metadata.person_name] = metadata
                st.session_state.candidate_registry[candidate_id] = {
                    "person_name": metadata.person_name,
                    "source_file": metadata.source_file,
                }

                # DEBUG: Show extracted metadata
                status.write(f"   👤 Name: {metadata.person_name}")
                status.write(f"   🛠️ Skills: {metadata.skills[:5]}...")

                # Chunk with metadata
                chunks = semantic_chunk_resume(text, metadata)
                all_chunks.extend(chunks)

                status.write(
                    f"✅ {pdf_file.name}: {metadata.person_name} ({len(chunks)} chunks)"
                )
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
                all_chunks, st.session_state.chroma_client, client
            )
            st.session_state.collection = collection

            # Store chunks for debugging
            st.session_state.all_chunks = all_chunks

            names_now = list(st.session_state.resume_metadata.keys())
            resolved_target = resolve_target_person(names_now)
            st.session_state.me_person = resolved_target

            status.update(label="✅ Vector store built!", state="complete")
            st.success(
                f"Processed {processed_count} resume(s) with {len(all_chunks)} chunks"
            )
            if not resolved_target:
                st.error(TARGET_RESUME_MISSING_ERROR)
            else:
                st.success(f"Recruiter chat target set to: {resolved_target}")

            # Show rejected files if any
            if rejected_count > 0:
                st.warning(f"{rejected_count} file(s) were rejected:")
                for filename, reason in rejected_files:
                    st.caption(f"• {filename}: {reason}")

            # Rerun so the sidebar/header reflect updated build state.
            st.rerun()

        elif rejected_count > 0 and processed_count == 0:
            # All files were rejected
            status.update(label="⚠️ No files could be processed", state="complete")
            st.error(
                f"All {rejected_count} file(s) were rejected due to security or validation issues:"
            )
            for filename, reason in rejected_files:
                st.write(f"• **{filename}**: {reason}")
            st.info("Please upload valid resume PDFs without any suspicious content.")

        else:
            # No files uploaded or some other issue
            status.update(label="❌ Processing failed", state="error")
            st.error("No valid chunks created. Please check your PDF files.")


# Chat interface
if st.session_state.messages:
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
else:
    # Empty state
    if not st.session_state.collection:
        st.markdown(
            """
        <div class="empty-state">
            <div class="empty-state-icon">📄</div>
            <h3>No resumes loaded yet</h3>
            <p>Upload PDF resumes in the sidebar and click "Build" to get started</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <h3>Ready!</h3>
            <p>Ask recruiter-style questions. Chat is locked to Sanjeev's resume.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

chat_placeholder = (
    f"Ask {me_name} anything..." if me_name else TARGET_RESUME_MISSING_ERROR
)
if query := st.chat_input(chat_placeholder):
    st.session_state.messages.append({"role": "user", "content": query})
    conversation_history = st.session_state.messages[:-1]

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    if st.session_state.collection is not None and not me_name:
        with st.chat_message("assistant", avatar="🤖"):
            st.error(TARGET_RESUME_MISSING_ERROR)
            st.session_state.messages.append(
                {"role": "assistant", "content": TARGET_RESUME_MISSING_ERROR}
            )
        st.stop()

    if st.session_state.collection is not None and me_name:
        # Router lane 1: recruiter-context intake (name/company/follow-up logistics).
        if _is_recruiter_context_query(query, client):
            stored, reply = store_recruiter_context_from_query(query, me_name, client)
            if stored:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                st.stop()

        # Router lane 2: explicit memory recall questions.
        recall_reply = answer_recruiter_memory_recall(query, me_name)
        if recall_reply:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(recall_reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": recall_reply}
                )
            st.stop()

    if st.session_state.collection is not None:
        query_ok, query_error = run_query_guardrails(
            query, client, conversation_history
        )
        if not query_ok:
            with st.chat_message("assistant", avatar="🤖"):
                st.warning(query_error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": query_error}
                )
            st.stop()

    if st.session_state.collection is None:
        with st.chat_message("assistant", avatar="🤖"):
            st.error("Please upload resumes and build the vector store first.")
    elif me_name:
        # Agent mode: recruiter-impersonation with tools (memory scoped to current person to avoid mixing candidates)
        memory_context = get_recent_memories(limit=10, me_person=me_name)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                answer, tools_used = run_agent(
                    query,
                    conversation_history,
                    memory_context,
                    client,
                    st.session_state.collection,
                    me_name,
                )
            response_check = check_response_safety(answer)
            if not response_check.passed:
                answer = (
                    "I don't have that information in your materials. Try rephrasing or ask about something else."
                    if response_check.reason == "Response may reveal AI identity"
                    else "I encountered an issue generating a safe response. Please try rephrasing your question."
                )
            st.markdown(answer)
            if tools_used:
                st.caption("🔧 Tools used: " + ", ".join(tools_used))
            st.session_state.messages.append({"role": "assistant", "content": answer})
            summarize_and_store(query, answer, client, me_person=me_name)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            msg = TARGET_RESUME_MISSING_ERROR
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
