import type { BuildResponse, ChatResponse, StatusResponse } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

async function parseJson<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let detail = "Request failed";
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) {
      detail = body.detail;
    }
  } catch {
    detail = response.statusText || detail;
  }

  throw new Error(detail);
}

export async function getStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE}/api/status`);
  return parseJson<StatusResponse>(response);
}

export async function clearIndex(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/index/clear`, {
    method: "POST",
  });
  await parseJson<{ status: string }>(response);
}

export async function buildIndex(files: File[]): Promise<BuildResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/api/index/build`, {
    method: "POST",
    body: formData,
  });

  return parseJson<BuildResponse>(response);
}

export async function sendChat(query: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  return parseJson<ChatResponse>(response);
}
