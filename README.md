# ResumeAI (Render-Ready Fullstack)

This repo now includes a full UI revamp with:

- `frontend/` React + TypeScript (Vite)
- `backend/` FastAPI + Python RAG pipeline
- `render.yaml` blueprint to deploy both services on Render

The old Streamlit app remains in `app.py` as legacy code, but the new default architecture is frontend + backend.

## Architecture

- Frontend uploads resume PDFs, triggers index build, and chats with the assistant.
- Backend extracts PDF text, chunks resumes, embeds with OpenAI, stores vectors in Chroma, and answers recruiter-style questions.
- Candidate metadata and vectors persist in `DATA_DIR`.

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── rag_engine.py
│   │   └── schemas.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── styles.css
│   │   └── types.ts
│   ├── package.json
│   └── .env.example
├── render.yaml
└── app.py  # legacy Streamlit app
```

## Local Development

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY in .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
cp .env.example .env
# optionally set VITE_API_BASE_URL (default: http://localhost:8000)
npm install
npm run dev
```

Open `http://localhost:5173`.

## Render Deployment

This repo includes `render.yaml` for two services:

1. `resumeai-backend` (Web Service)
2. `resumeai-frontend` (Static Site)

### Steps

1. Push this repo to GitHub.
2. In Render, create a new Blueprint and select the repo.
3. Set `OPENAI_API_KEY` on `resumeai-backend`.
4. Ensure frontend `VITE_API_BASE_URL` points to backend URL.
5. Deploy.

## Notes

- Backend supports only PDF uploads.
- Default per-file limit is controlled by `MAX_FILE_SIZE_MB` (default `2`).
- For production persistence, keep a Render persistent disk mounted to `DATA_DIR` (configured as `/var/data` in `render.yaml`).
