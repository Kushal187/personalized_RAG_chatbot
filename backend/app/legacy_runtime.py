from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Optional


class SessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class StreamlitShim:
    def __init__(self) -> None:
        self.session_state = SessionState()

    # UI no-ops to satisfy legacy app.py calls in core section.
    def set_page_config(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None


class InMemoryUpload:
    def __init__(self, name: str, raw: bytes) -> None:
        self.name = name
        self._raw = raw
        self._buffer = io.BytesIO(raw)
        self.size = len(raw)

    def read(self, *args, **kwargs):
        return self._buffer.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._buffer.seek(*args, **kwargs)

    def tell(self):
        return self._buffer.tell()


class LegacyRuntime:
    def __init__(self, data_dir: str = "data") -> None:
        self.root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self._resolve_data_dir(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._st = StreamlitShim()
        self.ns = self._load_legacy_namespace()
        self._init_state()

    def _resolve_data_dir(self, configured_dir: str) -> Path:
        configured_path = Path(configured_dir)
        local_fallback = self.root_dir / "data"
        tmp_fallback = Path(gettempdir()) / "resumeai-data"

        for candidate in [configured_path, local_fallback, tmp_fallback]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_probe"
                with open(probe, "w", encoding="utf-8") as f:
                    f.write("ok")
                probe.unlink(missing_ok=True)
                if candidate != configured_path:
                    print(
                        f"WARNING: DATA_DIR '{configured_dir}' is not writable. "
                        f"Using fallback '{candidate}'."
                    )
                return candidate
            except Exception:
                continue

        raise RuntimeError(
            "Could not find a writable data directory. Set DATA_DIR to a writable path."
        )

    def _load_legacy_namespace(self) -> dict[str, Any]:
        app_path = self.root_dir / "app.py"
        source = app_path.read_text(encoding="utf-8")

        cutoff = source.find("# Streamlit UI")
        if cutoff == -1:
            raise RuntimeError("Could not locate '# Streamlit UI' marker in app.py")

        core_source = source[:cutoff]
        core_source = core_source.replace("import streamlit as st\n", "")

        module_name = "legacy_core"
        module = types.ModuleType(module_name)
        module.__file__ = str(app_path)
        module.st = self._st
        sys.modules[module_name] = module
        exec(compile(core_source, str(app_path), "exec"), module.__dict__)

        # Keep legacy memory file in backend data dir.
        module.__dict__["AGENT_MEMORY_FILE"] = self.data_dir / "agent_memory.json"
        return module.__dict__

    def _init_state(self) -> None:
        st = self._st
        ns = self.ns
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chroma_client" not in st.session_state:
            st.session_state.chroma_client = ns["get_chroma_client"]()
        if "collection" not in st.session_state:
            st.session_state.collection = None
        if "resume_metadata" not in st.session_state:
            st.session_state.resume_metadata = {}
        if "me_person" not in st.session_state:
            st.session_state.me_person = None
        if "agent_memories" not in st.session_state:
            st.session_state.agent_memories = ns["_compact_memories"](
                ns["_load_agent_memories"]()
            )
        if "candidate_registry" not in st.session_state:
            st.session_state.candidate_registry = {}
        if "all_chunks" not in st.session_state:
            st.session_state.all_chunks = []

    def _client(self):
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        return self.ns["OpenAI"](api_key=api_key)

    def clear_index(self) -> None:
        st = self._st
        st.session_state.messages = []
        st.session_state.resume_metadata = {}
        st.session_state.collection = None
        st.session_state.all_chunks = []
        st.session_state.me_person = None
        st.session_state.candidate_registry = {}

    def build_index(self, files: list[tuple[str, bytes]]) -> dict:
        if not files:
            raise ValueError("Please upload at least one PDF file.")

        self.clear_index()
        st = self._st
        ns = self.ns
        client = self._client()

        all_chunks = []
        processed_count = 0
        rejected_files: list[dict[str, str]] = []

        for filename, raw in files:
            pdf_file = InMemoryUpload(filename, raw)

            try:
                passed, errors, _warnings = ns["run_file_guardrails"](pdf_file, client)
                if not passed:
                    rejected_files.append(
                        {
                            "filename": filename,
                            "reason": (errors[0] if errors else "Unknown error"),
                        }
                    )
                    continue

                raw_text = ns["extract_text_from_pdf"](pdf_file)
                if not raw_text.strip():
                    rejected_files.append(
                        {"filename": filename, "reason": "Empty text extracted"}
                    )
                    continue

                text = ns["normalize_text"](raw_text)

                passed, errors, _warnings = ns["run_content_guardrails"](text, client)
                if not passed:
                    rejected_files.append(
                        {
                            "filename": filename,
                            "reason": (
                                errors[0] if errors else "Content validation failed"
                            ),
                        }
                    )
                    continue

                text = ns["sanitize_text_for_llm"](text)

                metadata = ns["extract_resume_metadata"](text, filename, client)
                if metadata.person_name == "Unknown" or metadata.person_name is None:
                    rejected_files.append(
                        {
                            "filename": filename,
                            "reason": "Could not identify person name in document",
                        }
                    )
                    continue

                original_name = metadata.person_name.strip()
                display_name = original_name
                if display_name in st.session_state.resume_metadata:
                    stem = Path(filename).stem[:24]
                    display_name = f"{original_name} ({stem})"
                metadata.person_name = display_name

                base_candidate_id = ns["_make_candidate_id"](display_name, filename)
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

                chunks = ns["semantic_chunk_resume"](text, metadata)
                all_chunks.extend(chunks)
                processed_count += 1

            except Exception as exc:
                rejected_files.append({"filename": filename, "reason": str(exc)})

        if all_chunks:
            collection = ns["build_collection"](
                all_chunks, st.session_state.chroma_client, client
            )
            st.session_state.collection = collection
            st.session_state.all_chunks = all_chunks
            names_now = list(st.session_state.resume_metadata.keys())
            st.session_state.me_person = ns["resolve_target_person"](names_now)

        rejected_count = len(rejected_files)
        return {
            "processed_count": processed_count,
            "rejected_count": rejected_count,
            "rejected_files": rejected_files,
            "candidate_count": len(st.session_state.resume_metadata),
            "chunk_count": len(st.session_state.get("all_chunks", [])),
            "candidates": list(st.session_state.resume_metadata.keys()),
            "target_person": st.session_state.me_person,
            "target_loaded": bool(st.session_state.me_person),
        }

    def get_status(self) -> dict:
        st = self._st
        return {
            "candidate_count": len(st.session_state.resume_metadata),
            "chunk_count": len(st.session_state.get("all_chunks", [])),
            "candidates": list(st.session_state.resume_metadata.keys()),
            "target_person": st.session_state.me_person,
            "target_loaded": bool(st.session_state.me_person),
        }

    def chat(self, query: str) -> dict:
        prompt_query = (query or "").strip()
        if not prompt_query:
            raise ValueError("Query cannot be empty.")

        st = self._st
        ns = self.ns
        client = self._client()

        if st.session_state.collection is None:
            raise ValueError("Please upload resumes and build the vector store first.")

        st.session_state.messages.append({"role": "user", "content": prompt_query})
        conversation_history = st.session_state.messages[:-1]
        intent_info: dict[str, Any] = {"intent": "resume", "is_pure_context": False}

        me_name = st.session_state.me_person
        if st.session_state.collection is not None and not me_name:
            msg = ns["TARGET_RESUME_MISSING_ERROR"]
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return {"answer": msg, "sources": [], "target_person": None}

        if st.session_state.collection is not None and me_name:
            intent_info = ns["classify_query_intent"](
                prompt_query, conversation_history, client
            )

            context_detected = intent_info.get("intent") == "recruiter_context" or (
                ns["_is_recruiter_context_query"](prompt_query, client)
            )
            if context_detected:
                stored, reply = ns["store_recruiter_context_from_query"](
                    prompt_query, me_name, client
                )
                if stored and intent_info.get("is_pure_context", False):
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    return {
                        "answer": reply,
                        "sources": [],
                        "target_person": me_name,
                    }

            if intent_info.get("intent") == "memory_recall" or ns[
                "_detect_memory_recall_intent"
            ](prompt_query):
                recall_reply = ns["answer_recruiter_memory_recall"](
                    prompt_query, me_name
                )
                if recall_reply:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": recall_reply}
                    )
                    return {
                        "answer": recall_reply,
                        "sources": [],
                        "target_person": me_name,
                    }

        if st.session_state.collection is not None:
            query_ok, query_error = ns["run_query_guardrails"](
                prompt_query,
                client,
                conversation_history,
                intent_hint=intent_info.get("intent"),
            )
            if not query_ok:
                msg = query_error or "Query blocked by guardrails."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return {"answer": msg, "sources": [], "target_person": me_name}

        memory_context = ns["get_recent_memories"](limit=10, me_person=me_name)
        answer, _tools_used = ns["run_agent"](
            prompt_query,
            conversation_history,
            memory_context,
            client,
            st.session_state.collection,
            me_name,
            intent_hint=intent_info.get("intent"),
        )
        response_check = ns["check_response_safety"](answer)
        if not response_check.passed:
            answer = (
                "I don't have that information in your materials. Try rephrasing or ask about something else."
                if response_check.reason == "Response may reveal AI identity"
                else "I encountered an issue generating a safe response. Please try rephrasing your question."
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        return {
            "answer": answer,
            "sources": [],
            "target_person": me_name,
        }
