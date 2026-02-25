export type ChatRole = "user" | "assistant";

export interface SourceItem {
  person_name: string;
  source_file: string;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  sources?: SourceItem[];
}

export interface StatusResponse {
  candidate_count: number;
  chunk_count: number;
  candidates: string[];
  target_person: string | null;
  target_loaded: boolean;
}

export interface RejectedFile {
  filename: string;
  reason: string;
}

export interface BuildResponse {
  processed_count: number;
  rejected_count: number;
  rejected_files: RejectedFile[];
  candidate_count: number;
  chunk_count: number;
  candidates: string[];
  target_person: string | null;
  target_loaded: boolean;
}

export interface ChatResponse {
  answer: string;
  sources: SourceItem[];
  target_person: string | null;
}
