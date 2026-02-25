import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { buildIndex, clearIndex, getStatus, sendChat } from "./api";
import type { ChatMessage, StatusResponse } from "./types";

type ViewMode = "landing" | "chat";

interface UploadedFile {
  id: string;
  file: File;
  name: string;
  size: number;
}

const targetName = "Sanjeev Kushal Pendekanti";
const initialMessageId = crypto.randomUUID();

const initialStatus: StatusResponse = {
  candidate_count: 0,
  chunk_count: 0,
  candidates: [],
  target_person: null,
  target_loaded: false,
};

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const [view, setView] = useState<ViewMode>("landing");
  const [status, setStatus] = useState<StatusResponse>(initialStatus);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: initialMessageId,
      role: "assistant",
      content:
        "Hello! I'm here to help you learn about Kushal's background and experience. Upload resumes and ask anything.",
    },
  ]);
  const [messageTimes, setMessageTimes] = useState<Record<string, string>>({
    [initialMessageId]: new Date().toISOString(),
  });
  const [inputValue, setInputValue] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [panelInfo, setPanelInfo] = useState<string | null>(null);
  const [panelError, setPanelError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  const questionCount = useMemo(
    () => messages.filter((m) => m.role === "user").length,
    [messages]
  );

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  async function refreshStatus(): Promise<void> {
    try {
      const next = await getStatus();
      setStatus(next);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Could not load status.";
      setPanelError(msg);
    }
  }

  function stampMessage(id: string): void {
    setMessageTimes((prev) => ({ ...prev, [id]: new Date().toISOString() }));
  }

  function addMessage(message: ChatMessage): void {
    stampMessage(message.id);
    setMessages((prev) => [...prev, message]);
  }

  async function rebuildIndex(nextFiles: UploadedFile[]): Promise<void> {
    setIsIndexing(true);
    setPanelError(null);
    setPanelInfo("Indexing resumes...");
    try {
      if (nextFiles.length === 0) {
        await clearIndex();
        await refreshStatus();
        setPanelInfo("Index cleared.");
        return;
      }

      const result = await buildIndex(nextFiles.map((f) => f.file));
      await refreshStatus();
      if (result.rejected_count > 0) {
        setPanelError(`${result.rejected_count} file(s) were rejected during indexing.`);
      } else {
        setPanelInfo(`Indexed ${result.processed_count} resume(s).`);
      }

      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `I've received ${nextFiles.length} resume(s). The information has been processed. Feel free to ask me any questions!`,
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Indexing failed.";
      setPanelError(msg);
      setPanelInfo(null);
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `I couldn't process the resumes: ${msg}`,
      });
    } finally {
      setIsIndexing(false);
    }
  }

  async function handleFileUpload(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const rawFiles = Array.from(event.target.files || []).filter((file) =>
      file.name.toLowerCase().endsWith(".pdf")
    );

    if (rawFiles.length === 0) {
      setPanelError("Only PDF files are supported.");
      return;
    }

    const existingByKey = new Set(uploadedFiles.map((f) => `${f.name}-${f.size}`));
    const newEntries: UploadedFile[] = rawFiles
      .filter((f) => !existingByKey.has(`${f.name}-${f.size}`))
      .map((file) => ({
        id: crypto.randomUUID(),
        file,
        name: file.name,
        size: file.size,
      }));

    const nextFiles = [...uploadedFiles, ...newEntries].slice(0, 50);
    setUploadedFiles(nextFiles);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    await rebuildIndex(nextFiles);
  }

  async function removeFile(fileId: string): Promise<void> {
    const nextFiles = uploadedFiles.filter((file) => file.id !== fileId);
    setUploadedFiles(nextFiles);
    await rebuildIndex(nextFiles);
  }

  async function sendMessage(): Promise<void> {
    const question = inputValue.trim();
    if (!question || isTyping) return;

    if (status.chunk_count === 0) {
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Please upload and index resumes first.",
      });
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };
    addMessage(userMessage);
    setInputValue("");
    setIsTyping(true);

    try {
      const reply = await sendChat(question);
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: reply.answer,
        sources: reply.sources,
      });
      await refreshStatus();
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Chat failed.";
      addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `I hit an error: ${msg}`,
      });
    } finally {
      setIsTyping(false);
    }
  }

  function onComposerSubmit(event: FormEvent): void {
    event.preventDefault();
    void sendMessage();
  }

  function runQuickQuestion(question: string): void {
    setInputValue(question);
    setTimeout(() => {
      void sendMessage();
    }, 80);
  }

  if (view === "landing") {
    return (
      <div className="landing-root">
        <header className="landing-header">
          <div className="landing-header-inner">
            <div className="logo-group">
              <div className="logo-mark">K</div>
              <span className="logo-text">Kushal AI</span>
            </div>
            <button className="outline-btn" onClick={() => setView("chat")}>
              Get Started
            </button>
          </div>
        </header>

        <main className="landing-main">
          <section className="hero-section">
            <div className="hero-chip">AI-Powered Recruitment Assistant</div>
            <h1>
              Meet Kushal through
              <br />
              intelligent conversation
            </h1>
            <p>
              An AI assistant that answers questions about Kushal's experience, skills,
              and qualifications. Upload resumes and get instant, accurate responses.
            </p>
            <div className="hero-actions">
              <button className="primary-btn" onClick={() => setView("chat")}>
                Start Conversation
              </button>
              <button className="outline-btn">Learn More</button>
            </div>
          </section>

          <section className="feature-grid">
            <article className="feature-card">
              <div className="feature-icon">AI</div>
              <h3>Intelligent Responses</h3>
              <p>Detailed answers about experience, projects, and technical expertise.</p>
            </article>
            <article className="feature-card">
              <div className="feature-icon">PDF</div>
              <h3>Resume Analysis</h3>
              <p>Upload multiple resumes and keep responses grounded in real profile data.</p>
            </article>
            <article className="feature-card">
              <div className="feature-icon">FAST</div>
              <h3>Instant Answers</h3>
              <p>Get immediate candidate-focused answers for recruiter workflows.</p>
            </article>
          </section>
        </main>

        <footer className="landing-footer">For recruiters and hiring managers</footer>
      </div>
    );
  }

  return (
    <div className="chat-root">
      <header className="chat-header">
        <div className="chat-header-inner">
          <div className="chat-header-left">
            <button className="ghost-btn" onClick={() => setView("landing")}>Back</button>
            <div className="logo-group">
              <div className="logo-mark">K</div>
              <div>
                <p className="logo-text">Kushal AI</p>
                <p className="logo-subtext">Ask me anything about Kushal</p>
              </div>
            </div>
          </div>

          <button
            className="outline-btn"
            disabled={isIndexing}
            onClick={() => fileInputRef.current?.click()}
          >
            {isIndexing ? "Indexing..." : "Upload Resume"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf"
            onChange={(event) => {
              void handleFileUpload(event);
            }}
            className="hidden-input"
          />
        </div>
      </header>

      <main className="chat-body">
        <section className="chat-main">
          <div className="messages-wrap">
            {messages.map((message) => (
              <article
                key={message.id}
                className={`message-row ${message.role === "user" ? "user" : "assistant"}`}
              >
                {message.role === "assistant" ? <div className="avatar assistant">K</div> : null}
                <div className={`message-bubble ${message.role === "user" ? "user" : "assistant"}`}>
                  <p>{message.content}</p>
                  <p className={`msg-time ${message.role === "user" ? "user" : "assistant"}`}>
                    {formatTime(messageTimes[message.id] ?? new Date().toISOString())}
                  </p>
                  {message.sources && message.sources.length > 0 ? (
                    <div className="sources-box">
                      <p className="sources-title">Sources</p>
                      <ul>
                        {message.sources.map((source, index) => (
                          <li key={`${message.id}-${index}`}>
                            {source.person_name} · {source.source_file}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
                {message.role === "user" ? <div className="avatar user">You</div> : null}
              </article>
            ))}

            {isTyping ? (
              <article className="message-row assistant">
                <div className="avatar assistant">K</div>
                <div className="typing-bubble">
                  <span />
                  <span />
                  <span />
                </div>
              </article>
            ) : null}
            <div ref={endRef} />
          </div>

          <form className="composer-row" onSubmit={onComposerSubmit}>
            <input
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder="Ask about experience, skills, projects..."
              disabled={isTyping}
            />
            <button type="submit" disabled={isTyping || !inputValue.trim()}>
              Send
            </button>
          </form>
        </section>

        <aside className="chat-side">
          <section className="side-card">
            <h3>Uploaded Resumes</h3>
            {uploadedFiles.length === 0 ? (
              <div className="empty-files">
                <div className="empty-icon">PDF</div>
                <p>No resumes uploaded yet</p>
              </div>
            ) : (
              <div className="file-list">
                {uploadedFiles.map((file) => (
                  <article key={file.id} className="file-row">
                    <div className="file-icon">PDF</div>
                    <div className="file-meta">
                      <p className="file-name">{file.name}</p>
                      <p className="file-size">{formatBytes(file.size)}</p>
                    </div>
                    <button
                      className="remove-btn"
                      onClick={() => {
                        void removeFile(file.id);
                      }}
                      aria-label={`Remove ${file.name}`}
                    >
                      x
                    </button>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section className="side-card">
            <h3>Quick Questions</h3>
            <div className="quick-list">
              {[
                "Tell me about your experience",
                "What are your key skills?",
                "Describe your recent projects",
                "What's your educational background?",
              ].map((question) => (
                <button
                  key={question}
                  onClick={() => runQuickQuestion(question)}
                  disabled={isTyping}
                >
                  {question}
                </button>
              ))}
            </div>
          </section>

          <section className="side-card compact">
            <h3>Index Status</h3>
            <p>{status.candidate_count} candidate(s), {status.chunk_count} chunks</p>
            <p>
              Target: {status.target_loaded ? status.target_person : `Missing (${targetName})`}
            </p>
            <p>Questions: {questionCount}</p>
            <button
              className="outline-btn small"
              onClick={() => {
                setUploadedFiles([]);
                void rebuildIndex([]);
              }}
              disabled={isIndexing}
            >
              Clear Index
            </button>
            {panelInfo ? <p className="panel-info">{panelInfo}</p> : null}
            {panelError ? <p className="panel-error">{panelError}</p> : null}
          </section>
        </aside>
      </main>
    </div>
  );
}
