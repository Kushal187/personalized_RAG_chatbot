from __future__ import annotations

import io
import json
import os
import re
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Optional

import chromadb
from openai import OpenAI
from PyPDF2 import PdfReader

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class RAGEngine:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        self.openai = OpenAI(api_key=api_key)
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.answer_model = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
        self.extraction_model = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
        self.collection_name = os.getenv("COLLECTION_NAME", "resumes")
        self.target_person = os.getenv("TARGET_PERSON", "Sanjeev Kushal Pendekanti")
        self.max_memories_per_person = int(os.getenv("MAX_MEMORIES_PER_PERSON", "50"))
        self.max_total_memories = int(os.getenv("MAX_TOTAL_MEMORIES", "300"))

        max_file_size_mb = float(os.getenv("MAX_FILE_SIZE_MB", "2"))
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        data_dir = self._resolve_data_dir(os.getenv("DATA_DIR", "data"))
        self.data_dir = data_dir
        self.chroma_dir = data_dir / "chroma"
        self.registry_path = data_dir / "candidate_registry.json"
        self.memory_path = data_dir / "agent_memory.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self._get_or_create_collection()
        self.candidate_registry = self._load_registry()
        self.agent_memories = self._compact_memories(self._load_agent_memories())
        self.chat_history: list[dict[str, str]] = []

    @staticmethod
    def _ensure_writable_dir(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _resolve_data_dir(self, configured_dir: str) -> Path:
        configured_path = Path(configured_dir)
        local_fallback = Path.cwd() / "data"
        tmp_fallback = Path(gettempdir()) / "resumeai-data"

        candidates = [configured_path, local_fallback, tmp_fallback]
        seen: set[str] = set()

        for candidate in candidates:
            candidate_key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if self._ensure_writable_dir(candidate):
                if candidate != configured_path:
                    print(
                        f"WARNING: DATA_DIR '{configured_dir}' is not writable. "
                        f"Using fallback '{candidate}'."
                    )
                return candidate

        raise RuntimeError(
            "Could not find a writable data directory. "
            "Set DATA_DIR to a writable path."
        )

    def _get_or_create_collection(self):
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _load_registry(self) -> dict[str, dict]:
        if not self.registry_path.exists():
            return {}

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
        return {}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.candidate_registry, f, indent=2)

    @staticmethod
    def _normalize_name(name: str) -> str:
        lowered = (name or "").strip().lower()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    @staticmethod
    def _name_tokens(name: str) -> set[str]:
        return {t for t in RAGEngine._normalize_name(name).split(" ") if t}

    @staticmethod
    def _file_stem(filename: str) -> str:
        stem = Path(filename).stem.strip()
        return stem if stem else "Unknown"

    @staticmethod
    def _is_contextual_followup_query(
        query: str, conversation_history: Optional[list[dict]]
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
        self, query: str, conversation_history: Optional[list[dict]]
    ) -> str:
        if not self._is_contextual_followup_query(query, conversation_history):
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

    @staticmethod
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

    @staticmethod
    def _extract_recruiter_identity(text: str) -> dict[str, Optional[str]]:
        src = (text or "").strip()
        name = None
        company = None

        name_patterns = [
            r"\b(?:i['’]m|i am|my name is)\s+([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2})(?=\s+(?:from|at|with)\b|[.,;!]|$)",
            r"\bthis is\s+([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2})(?=\s+(?:from|at|with)\b|[.,;!]|$)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, src, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip().rstrip(".,")
                break

        company_patterns = [
            r"\bfrom\s+([A-Z][A-Za-z0-9& \-]{1,50}?)(?=[\.,;]|$)",
            r"\bwith\s+([A-Z][A-Za-z0-9& \-]{1,50}?)(?=[\.,;]|$)",
            r"\bat\s+([A-Z][A-Za-z0-9& \-]{1,50}?)(?=[\.,;]|$)",
        ]
        for pattern in company_patterns:
            match = re.search(pattern, src, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip().rstrip(".,")
            if any(
                tok in candidate.lower()
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
            company = candidate
            break
        return {"name": name, "company": company}

    @staticmethod
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
            tokens = [token for token in re.split(r"\s+", name) if token]
            if tokens and all(token.islower() for token in tokens):
                name = " ".join(token.capitalize() for token in tokens)

        return name, company

    def _extract_recruiter_context_fields(self, query: str) -> dict[str, Optional[str]]:
        query = query or ""
        identity = self._extract_recruiter_identity(query)
        followup = self._extract_followup_details(query)
        out = {
            "recruiter_name": identity.get("name"),
            "recruiter_company": identity.get("company"),
            "followup_day": followup.get("day"),
            "followup_time": followup.get("time"),
            "followup_timezone": followup.get("timezone"),
        }
        if any(v for v in out.values()):
            return out

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
            resp = self.openai.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt.format(text=query[:500])}],
                temperature=0,
                max_tokens=180,
                response_format={"type": "json_object"},
            )
            data = self._safe_json(resp.choices[0].message.content or "{}")
            return {
                "recruiter_name": data.get("recruiter_name"),
                "recruiter_company": data.get("recruiter_company"),
                "followup_day": data.get("followup_day"),
                "followup_time": data.get("followup_time"),
                "followup_timezone": data.get("followup_timezone"),
            }
        except Exception:
            return out

    @staticmethod
    def _normalize_followup_display_parts(
        day: Optional[str], time_str: Optional[str], timezone_str: Optional[str]
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

    def _is_recruiter_context_query(self, query: str) -> bool:
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
        if any(marker in q for marker in explicit_markers):
            fields = self._extract_recruiter_context_fields(query)
            return any(v for v in fields.values())
        return False

    @staticmethod
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
            p in q
            for p in ["which company", "what company am i from", "where am i from"]
        ):
            return "recruiter_company"
        return None

    def _classify_query_intent(
        self, query: str, conversation_history: Optional[list[dict]]
    ) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"intent": "resume", "is_pure_context": False}

        if self._detect_memory_recall_intent(q):
            return {"intent": "memory_recall", "is_pure_context": False}

        recent = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    recent.append(
                        f"{'Recruiter' if role == 'user' else 'Candidate'}: {content[:220]}"
                    )
        history_text = "\n".join(recent) if recent else "(none)"

        prompt = """Classify this recruiter message for a resume-chat assistant.
Return JSON with:
- intent: one of ["recruiter_context", "memory_recall", "resume", "offtopic"]
- is_pure_context: true/false

Definitions:
- recruiter_context: recruiter introduces self/company or follow-up logistics (name/company/time/timezone)
- memory_recall: asks to recall stored logistics/identity ("what time did we agree")
- resume: asks about candidate background/experience/skills/projects/education
- offtopic: unrelated to above

Set is_pure_context=true ONLY when the message is mostly logistics/intro and does NOT request resume details.
If message contains both recruiter intro/logistics and resume questions, choose "resume" and is_pure_context=false.

Recent conversation:
{history}

Message:
{query}
"""
        try:
            resp = self.openai.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.format(history=history_text, query=q[:700]),
                    }
                ],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            data = self._safe_json(resp.choices[0].message.content or "{}")
            intent = str(data.get("intent") or "").strip().lower()
            pure = bool(data.get("is_pure_context", False))
            allowed = {"recruiter_context", "memory_recall", "resume", "offtopic"}
            if intent not in allowed:
                intent = "resume"
            return {"intent": intent, "is_pure_context": pure}
        except Exception:
            fields = self._extract_recruiter_context_fields(q)
            has_context = any(v for v in fields.values())
            return {
                "intent": "recruiter_context" if has_context else "resume",
                "is_pure_context": bool(has_context),
            }

    def _make_candidate_id(self, person_name: str, source_file: str) -> str:
        key = f"{person_name}|{source_file}".lower().encode("utf-8")
        return hashlib.sha1(key).hexdigest()[:16]

    def _safe_json(self, text: str) -> dict:
        body = (text or "").strip()
        if not body:
            return {}

        try:
            value = json.loads(body)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", body, flags=re.DOTALL)
        if not match:
            return {}

        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _load_agent_memories(self) -> list[dict]:
        if not self.memory_path.exists():
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
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
            return out
        except Exception:
            return []

    def _save_agent_memories(self) -> None:
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump({"memories": self.agent_memories}, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _normalize_fact_for_hash(fact: str) -> str:
        return re.sub(r"\s+", " ", (fact or "").strip().lower())

    def _compact_memories(self, mems: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for memory in mems:
            key = (
                memory.get("person"),
                memory.get("memory_type", "fact"),
                memory.get("fact_hash"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(memory)

        grouped: dict[Optional[str], list[dict]] = {}
        for memory in deduped:
            grouped.setdefault(memory.get("person"), []).append(memory)

        kept = []
        for person, items in grouped.items():
            items.sort(key=lambda x: x.get("created_at", ""))
            kept.extend(items[-self.max_memories_per_person :])

        kept.sort(key=lambda x: x.get("created_at", ""))
        return kept[-self.max_total_memories :]

    def _add_memory(
        self,
        fact: str,
        person: Optional[str] = None,
        source: str = "recruiter_context",
        memory_type: str = "fact",
        value: Optional[dict] = None,
        confidence: float = 1.0,
    ) -> None:
        value = value or {}
        if (not fact or not fact.strip()) and not value:
            return

        normalized_fact = self._normalize_fact_for_hash(fact or "")
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
        self.agent_memories.append(entry)
        self.agent_memories = self._compact_memories(self.agent_memories)
        self._save_agent_memories()

    def _get_structured_memories(
        self,
        me_person: Optional[str],
        memory_type: Optional[str] = None,
        source: str = "recruiter_context",
    ) -> list[dict]:
        out = []
        for memory in self.agent_memories:
            if not isinstance(memory, dict):
                continue
            if me_person and memory.get("person") != me_person:
                continue
            if source and memory.get("source") != source:
                continue
            if memory_type and memory.get("memory_type") != memory_type:
                continue
            out.append(memory)
        out.sort(key=lambda x: x.get("created_at", ""))
        return out

    def _store_recruiter_context_from_query(
        self, query: str, me_person: str
    ) -> tuple[bool, str]:
        fields = self._extract_recruiter_context_fields(query)
        recruiter_name = (fields.get("recruiter_name") or "").strip() or None
        recruiter_company = (fields.get("recruiter_company") or "").strip() or None
        recruiter_name, recruiter_company = self._cleanup_recruiter_identity_fields(
            recruiter_name, recruiter_company
        )
        followup_day = (fields.get("followup_day") or "").strip() or None
        followup_time = (fields.get("followup_time") or "").strip() or None
        followup_timezone = (fields.get("followup_timezone") or "").strip() or None
        display_day, display_time, display_tz = self._normalize_followup_display_parts(
            followup_day, followup_time, followup_timezone
        )

        stored = False
        if recruiter_name:
            self._add_memory(
                fact=f"Recruiter name: {recruiter_name}",
                person=me_person,
                source="recruiter_context",
                memory_type="recruiter_identity",
                value={"recruiter_name": recruiter_name},
                confidence=1.0,
            )
            stored = True
        if recruiter_company:
            self._add_memory(
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
            self._add_memory(
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

        return True, (" ".join(ack_bits).strip() or "Got it - I've noted that context.")

    def _answer_recruiter_memory_recall(
        self, query: str, me_person: str
    ) -> Optional[str]:
        intent = self._detect_memory_recall_intent(query)
        if not intent:
            return None

        followups = self._get_structured_memories(me_person, memory_type="followup")
        identities = self._get_structured_memories(
            me_person, memory_type="recruiter_identity"
        )

        def latest_value(entries: list[dict], key: str) -> Optional[str]:
            for memory in reversed(entries):
                value = memory.get("value") if isinstance(memory.get("value"), dict) else {}
                v = (value.get(key) if isinstance(value, dict) else None) or ""
                v = str(v).strip()
                if v:
                    return v
            return None

        if intent in {"followup_time", "followup"}:
            day = latest_value(followups, "followup_day")
            time_val = latest_value(followups, "followup_time")
            tz = latest_value(followups, "followup_timezone")
            day, time_val, tz = self._normalize_followup_display_parts(day, time_val, tz)
            if not day and not time_val:
                return "I don't have a stored follow-up time yet."
            parts = [p for p in [day, time_val, tz] if p]
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

    def _append_history(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})
        self.chat_history = self.chat_history[-40:]

    def _extract_text(self, raw_pdf: bytes) -> str:
        text_pdfplumber = ""
        text_pypdf = ""

        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(io.BytesIO(raw_pdf)) as pdf:
                    pages = [(p.extract_text() or "").strip() for p in pdf.pages]
                text_pdfplumber = "\n\n".join([p for p in pages if p])
            except Exception:
                text_pdfplumber = ""

        try:
            reader = PdfReader(io.BytesIO(raw_pdf))
            pages = [(p.extract_text() or "").strip() for p in reader.pages]
            text_pypdf = "\n\n".join([p for p in pages if p])
        except Exception:
            text_pypdf = ""

        best = text_pdfplumber if len(text_pdfplumber) >= len(text_pypdf) else text_pypdf
        best = best.replace("\t", " ")
        best = re.sub(r"[ ]{2,}", " ", best)
        best = re.sub(r"\n{3,}", "\n\n", best)
        return best.strip()

    def _extract_metadata(self, text: str, filename: str) -> dict:
        prompt = f"""
Extract candidate metadata from this resume text.
Return JSON with keys:
- person_name (string)
- skills (array of strings)
- current_role (string or null)
- companies (array of strings)
- experience_years (integer or null)

Resume text:
{text[:12000]}
"""

        try:
            resp = self.openai.chat.completions.create(
                model=self.extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not include markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or "{}"
            payload = self._safe_json(raw)
        except Exception:
            payload = {}

        person_name = str(payload.get("person_name") or "").strip()
        if not person_name:
            person_name = self._file_stem(filename)

        skills = payload.get("skills")
        if not isinstance(skills, list):
            skills = []
        skills = [str(s).strip() for s in skills if str(s).strip()][:30]

        companies = payload.get("companies")
        if not isinstance(companies, list):
            companies = []
        companies = [str(c).strip() for c in companies if str(c).strip()][:30]

        role = payload.get("current_role")
        role = str(role).strip() if role else None

        years = payload.get("experience_years")
        if isinstance(years, (int, float)):
            years = int(years)
        else:
            years = None

        return {
            "person_name": person_name,
            "skills": skills,
            "current_role": role,
            "companies": companies,
            "experience_years": years,
        }

    def _chunk_text(self, text: str, max_chunk_chars: int = 1400) -> list[str]:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""

        for part in parts:
            if len(part) > max_chunk_chars:
                sentences = re.split(r"(?<=[.!?])\s+", part)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current) + len(sentence) + 1 <= max_chunk_chars:
                        current = f"{current} {sentence}".strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sentence
                continue

            if len(current) + len(part) + 2 <= max_chunk_chars:
                current = f"{current}\n\n{part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c) > 40]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.openai.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in resp.data]

    def _resolve_target_candidate(self) -> tuple[Optional[str], Optional[str]]:
        if not self.candidate_registry:
            return None, None

        target_norm = self._normalize_name(self.target_person)
        target_tokens = self._name_tokens(self.target_person)

        best_candidate_id = None
        best_person_name = None
        best_score = -1

        for candidate_id, item in self.candidate_registry.items():
            person_name = str(item.get("person_name") or "").strip()
            person_norm = self._normalize_name(person_name)
            if person_norm == target_norm:
                return candidate_id, person_name

            overlap = len(self._name_tokens(person_name) & target_tokens)
            if overlap > best_score:
                best_score = overlap
                best_candidate_id = candidate_id
                best_person_name = person_name

        if best_score >= 2:
            return best_candidate_id, best_person_name
        return None, None

    def clear_index(self) -> None:
        with self.lock:
            self._clear_index_locked()

    def _clear_index_locked(self) -> None:
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self._get_or_create_collection()
        self.candidate_registry = {}
        self._save_registry()

    def build_index(self, files: list[tuple[str, bytes]]) -> dict:
        if not files:
            raise ValueError("Please upload at least one PDF file.")

        with self.lock:
            self._clear_index_locked()

            all_docs: list[str] = []
            all_ids: list[str] = []
            all_meta: list[dict] = []

            processed_count = 0
            rejected: list[dict] = []

            for filename, raw in files:
                if not filename.lower().endswith(".pdf"):
                    rejected.append({"filename": filename, "reason": "Only PDF files are supported."})
                    continue

                if len(raw) > self.max_file_size_bytes:
                    rejected.append(
                        {
                            "filename": filename,
                            "reason": f"File exceeds {self.max_file_size_bytes // (1024 * 1024)}MB limit.",
                        }
                    )
                    continue

                text = self._extract_text(raw)
                if not text:
                    rejected.append({"filename": filename, "reason": "Could not extract text from PDF."})
                    continue

                metadata = self._extract_metadata(text, filename)
                person_name = metadata["person_name"]
                candidate_id = self._make_candidate_id(person_name, filename)

                chunks = self._chunk_text(text)
                if not chunks:
                    rejected.append({"filename": filename, "reason": "No usable text chunks extracted."})
                    continue

                for idx, chunk in enumerate(chunks):
                    all_docs.append(chunk)
                    all_ids.append(f"{candidate_id}_{idx}")
                    all_meta.append(
                        {
                            "person_name": person_name,
                            "candidate_id": candidate_id,
                            "source_file": filename,
                            "chunk_id": idx,
                            "skills": ", ".join(metadata["skills"]),
                            "companies": ", ".join(metadata["companies"]),
                            "experience_years": metadata["experience_years"] or 0,
                        }
                    )

                self.candidate_registry[candidate_id] = {
                    "person_name": person_name,
                    "source_file": filename,
                    "skills": metadata["skills"],
                    "companies": metadata["companies"],
                    "experience_years": metadata["experience_years"],
                    "chunk_count": len(chunks),
                }
                processed_count += 1

            if not all_docs:
                self._save_registry()
                return {
                    "processed_count": 0,
                    "rejected_count": len(rejected),
                    "rejected_files": rejected,
                    "candidate_count": 0,
                    "chunk_count": 0,
                    "candidates": [],
                    "target_person": None,
                    "target_loaded": False,
                }

            batch_size = 50
            for start in range(0, len(all_docs), batch_size):
                docs_batch = all_docs[start : start + batch_size]
                ids_batch = all_ids[start : start + batch_size]
                meta_batch = all_meta[start : start + batch_size]
                embeddings = self._embed(docs_batch)
                self.collection.add(
                    ids=ids_batch,
                    documents=docs_batch,
                    embeddings=embeddings,
                    metadatas=meta_batch,
                )

            self._save_registry()
            _, resolved_target = self._resolve_target_candidate()

            return {
                "processed_count": processed_count,
                "rejected_count": len(rejected),
                "rejected_files": rejected,
                "candidate_count": len(self.candidate_registry),
                "chunk_count": int(self.collection.count()),
                "candidates": [
                    item.get("person_name", "Unknown") for item in self.candidate_registry.values()
                ],
                "target_person": resolved_target,
                "target_loaded": bool(resolved_target),
            }

    def get_status(self) -> dict:
        with self.lock:
            _, resolved_target = self._resolve_target_candidate()
            return {
                "candidate_count": len(self.candidate_registry),
                "chunk_count": int(self.collection.count()),
                "candidates": [
                    item.get("person_name", "Unknown") for item in self.candidate_registry.values()
                ],
                "target_person": resolved_target,
                "target_loaded": bool(resolved_target),
            }

    def chat(self, query: str) -> dict:
        prompt_query = (query or "").strip()
        if not prompt_query:
            raise ValueError("Query cannot be empty.")

        with self.lock:
            if int(self.collection.count()) == 0:
                raise ValueError("No resume index found. Build the index first.")

            target_candidate_id, resolved_target = self._resolve_target_candidate()
            me_person = resolved_target or self.target_person
            conversation_history = self.chat_history[-18:]
            intent_info = self._classify_query_intent(prompt_query, conversation_history)

            # Match app.py behavior: route recruiter intros/logistics before resume retrieval.
            context_detected = intent_info.get("intent") == "recruiter_context" or (
                self._is_recruiter_context_query(prompt_query)
            )
            if context_detected and me_person:
                stored, reply = self._store_recruiter_context_from_query(
                    prompt_query, me_person
                )
                if stored and intent_info.get("is_pure_context", False):
                    self._append_history("user", prompt_query)
                    self._append_history("assistant", reply)
                    return {
                        "answer": reply,
                        "sources": [],
                        "target_person": resolved_target,
                    }

            # Match app.py behavior: explicit memory-recall lane.
            if intent_info.get("intent") == "memory_recall" and me_person:
                recall_reply = self._answer_recruiter_memory_recall(prompt_query, me_person)
                if recall_reply:
                    self._append_history("user", prompt_query)
                    self._append_history("assistant", recall_reply)
                    return {
                        "answer": recall_reply,
                        "sources": [],
                        "target_person": resolved_target,
                    }

            if intent_info.get("intent") == "offtopic":
                msg = (
                    "I'm designed to answer questions about uploaded resumes and recruiter context. "
                    "Please ask about candidate background, skills, projects, or follow-up details."
                )
                self._append_history("user", prompt_query)
                self._append_history("assistant", msg)
                return {"answer": msg, "sources": [], "target_person": resolved_target}

            effective_query = self._augment_followup_query_with_recent_subject(
                prompt_query, conversation_history
            )
            query_embedding = self._embed([effective_query])[0]

            where = {"candidate_id": target_candidate_id} if target_candidate_id else None
            search = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=8,
                where=where,
            )

            docs = (search.get("documents") or [[]])[0]
            metadatas = (search.get("metadatas") or [[]])[0]

            if not docs:
                search = self.collection.query(query_embeddings=[query_embedding], n_results=8)
                docs = (search.get("documents") or [[]])[0]
                metadatas = (search.get("metadatas") or [[]])[0]

            if not docs:
                raise ValueError("I couldn't find relevant resume context for that question.")

            context_blocks: list[str] = []
            sources: list[dict] = []

            for doc, meta in zip(docs, metadatas):
                person_name = str(meta.get("person_name", "Unknown"))
                source_file = str(meta.get("source_file", "Unknown"))
                snippet = (doc or "").strip()
                snippet = re.sub(r"\s+", " ", snippet)
                context_blocks.append(f"[{person_name} | {source_file}]\n{snippet}")
                sources.append(
                    {
                        "person_name": person_name,
                        "source_file": source_file,
                        "snippet": snippet[:240],
                    }
                )

            if resolved_target:
                system_prompt = (
                    "You are a recruiter-facing assistant answering questions about the target candidate. "
                    "Use only the provided context. If context is insufficient, say what is missing. "
                    "Do not invent facts. Keep answers concise and specific. "
                    "Do not provide a full resume dump unless the user explicitly asks for a summary or overview. "
                    "If the user introduces themselves or greets you, reply briefly and invite a specific question."
                )
            else:
                system_prompt = (
                    "You are a resume assistant. Use only the provided context. "
                    "If context is insufficient, say what is missing. Do not invent facts. "
                    "Do not provide a full profile summary unless explicitly requested."
                )

            user_prompt = (
                f"Question:\n{effective_query}\n\n"
                f"Context:\n{chr(10).join(context_blocks[:8])}\n\n"
                "Answer with recruiter-friendly bullet points when helpful."
            )

            response = self.openai.chat.completions.create(
                model=self.answer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            answer = (response.choices[0].message.content or "").strip()

            if not answer:
                answer = "I couldn't generate a reliable answer from the current resume context."
            self._append_history("user", prompt_query)
            self._append_history("assistant", answer)

            return {
                "answer": answer,
                "sources": sources[:5],
                "target_person": resolved_target,
            }
