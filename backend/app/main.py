from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .legacy_runtime import LegacyRuntime
from .schemas import BuildResponse, ChatRequest, ChatResponse, StatusResponse

load_dotenv()

app = FastAPI(title="ResumeAI API", version="1.0.0")
engine = LegacyRuntime(data_dir=os.getenv("DATA_DIR", "data"))

cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
allow_all_origins = cors_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(**engine.get_status())


@app.post("/api/index/clear")
def clear_index() -> dict:
    engine.clear_index()
    return {"status": "ok"}


@app.post("/api/index/build", response_model=BuildResponse)
async def build_index(files: list[UploadFile] = File(...)) -> BuildResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF file.")

    ingested: list[tuple[str, bytes]] = []
    for file in files:
        filename = file.filename or "resume.pdf"
        raw = await file.read()
        ingested.append((filename, raw))

    try:
        result = engine.build_index(ingested)
        return BuildResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Build failed: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = engine.chat(payload.query)
        return ChatResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
