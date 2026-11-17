
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const suggestions = [
  {
    icon: "⌕",
    title: "Understand a concept",
    description: "Explore important concepts in your PDFs.",
    question: "Explain the main concepts covered in my uploaded documents.",
  },
  {
    icon: "▤",
    title: "Summarize research",
    description: "Discover key findings and conclusions.",
    question: "Summarize the key findings across my uploaded documents.",
  },
  {
    icon: "⇄",
    title: "Compare information",
    description: "Compare findings across different documents.",
    question: "Compare the important findings across my documents.",
  },
  {
    icon: "✧",
    title: "Explore deeper",
    description: "Discover insights from your research library.",
    question: "What are the most important insights in my documents?",
  },
];

function Icon({ name }) {
  const icons = {
    plus: "＋",
    search: "⌕",
    book: "▤",
    menu: "☰",
    copy: "▢",
    check: "✓",
    close: "×",
    file: "▤",
    sparkle: "✳",
    clock: "◷",
    arrow: "↗",
    upload: "↑",
    trash: "⌫",
  };

  return <span className="icon">{icons[name] || "•"}</span>;
}

function formatTime() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getSourceName(source) {
  return String(
    source.source ||
    source.filename ||
    source.document ||
    "Document source"
  ).split(/[\\/]/).pop();
}

function formatDuration(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) return "—";
  return number < 1
    ? `${Math.round(number * 1000)} ms`
    : `${number.toFixed(2)} s`;
}

function EvidenceMap({ sources }) {
  const [selected, setSelected] = useState(0);

  if (!sources?.length) return null;

  const visible = sources.slice(0, 6);
  const positions = [
    [50, 14],
    [82, 35],
    [82, 72],
    [50, 88],
    [18, 72],
    [18, 35],
  ];

  const active = visible[selected] || visible[0];

  return (
    <section className="insight-panel evidence-panel">
      <div className="insight-heading">
        <div>
          <span className="panel-kicker">RETRIEVAL VISUALIZATION</span>
          <h3>Evidence constellation</h3>
          <p>Explore the passages supporting this answer.</p>
        </div>
        <span className="insight-count">
          {sources.length} SOURCES
        </span>
      </div>

      <div className="constellation-layout">
        <div className="constellation">
          <div className="constellation-orbit orbit-one" />
          <div className="constellation-orbit orbit-two" />

          <svg
            className="constellation-lines"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {visible.map((source, index) => (
              <line
                key={source.id || index}
                x1="50"
                y1="50"
                x2={positions[index][0]}
                y2={positions[index][1]}
                className={
                  selected === index ? "line-active" : ""
                }
              />
            ))}
          </svg>

          <div className="constellation-core">
            <span>✳</span>
            <small>YOUR QUESTION</small>
          </div>

          {visible.map((source, index) => (
            <button
              key={source.id || index}
              type="button"
              className={`constellation-node ${selected === index ? "selected" : ""
                }`}
              style={{
                left: `${positions[index][0]}%`,
                top: `${positions[index][1]}%`,
              }}
              onClick={() => setSelected(index)}
              title={getSourceName(source)}
            >
              {source.id || `S${index + 1}`}
            </button>
          ))}
        </div>

        <div className="evidence-detail">
          <div className="detail-topline">
            <span>{active.id || `S${selected + 1}`}</span>
            <span>RETRIEVED PASSAGE</span>
          </div>

          <h4>{getSourceName(active)}</h4>
          <div className="detail-page">
            PAGE {active.page ?? "—"}
          </div>

          <p>
            {active.text ||
              active.content ||
              "Evidence retrieved from this document."}
          </p>

          <div className="detail-footer">
            <span className="evidence-dot" />
            Connected to this response
          </div>
        </div>
      </div>

      {sources.length > 6 && (
        <p className="constellation-note">
          Showing 6 of {sources.length} sources. All sources are
          available in the source list.
        </p>
      )}
    </section>
  );
}

function ResearchTrace({ timings = {}, sources = [] }) {
  const steps = [
    ["01", "Your question", "Query received"],
    ["02", "Hybrid retrieval", "Semantic + keyword search"],
    ["03", "Evidence ranking", "Relevant passages"],
    ["04", "Gemini response", "Grounded answer"],
  ];

  return (
    <section className="insight-panel trace-panel">
      <div className="insight-heading">
        <div>
          <span className="panel-kicker">BEHIND THE ANSWER</span>
          <h3>Research pipeline</h3>
          <p>How AtlasQuery generated this response.</p>
        </div>
        <span className="trace-live">
          <i /> PROCESS COMPLETE
        </span>
      </div>

      <div className="pipeline-track">
        {steps.map(([number, title, description], index) => (
          <div className="pipeline-step" key={number}>
            <div className={`pipeline-number step-${index}`}>
              {number}
            </div>
            <div>
              <strong>{title}</strong>
              <span>{description}</span>
            </div>
            {index < 3 && <div className="pipeline-connector" />}
          </div>
        ))}
      </div>

      <div className="trace-metrics">
        <div className="trace-metric">
          <span>RETRIEVAL</span>
          <strong>
            {formatDuration(
              timings.retrieval_seconds ?? timings.retrieval
            )}
          </strong>
        </div>
        <div className="trace-metric">
          <span>GENERATION</span>
          <strong>
            {formatDuration(
              timings.generation_seconds ?? timings.generation
            )}
          </strong>
        </div>
        <div className="trace-metric">
          <span>TOTAL TIME</span>
          <strong>
            {formatDuration(
              timings.total_seconds ?? timings.total
            )}
          </strong>
        </div>
        <div className="trace-metric">
          <span>EVIDENCE</span>
          <strong>{sources.length}</strong>
        </div>
      </div>
    </section>
  );
}

function Message({ message }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showInsights, setShowInsights] = useState(true);

  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className={`message-row ${message.role}`}>
      {message.role === "assistant" && (
        <div className="assistant-avatar">
          <Icon name="sparkle" />
        </div>
      )}

      <div className="message-content">
        <div className="message-label">
          <span>
            {message.role === "user" ? "You" : "AtlasQuery"}
          </span>
          <span className="message-time">{message.time}</span>
        </div>

        {message.role === "user" ? (
          <div className="user-bubble">{message.content}</div>
        ) : (
          <>
            <div className="answer-text">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ node, ...props }) => (
                    <a
                      {...props}
                      target="_blank"
                      rel="noopener noreferrer"
                    />
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>

            {message.warnings?.length > 0 && (
              <div className="answer-warning">
                <span>ⓘ</span>
                <div>
                  {message.warnings.map((warning, index) => (
                    <p key={index}>{warning}</p>
                  ))}
                </div>
              </div>
            )}

            {message.sources?.length > 0 && (
              <>
                <div className="insights-toolbar">
                  <div>
                    <span className="insights-toolbar-icon">✳</span>
                    <div>
                      <strong>Explore the evidence</strong>
                      <span>
                        Visual breakdown of retrieved knowledge
                      </span>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      setShowInsights((value) => !value)
                    }
                  >
                    {showInsights ? "Hide insights −" : "Show insights +"}
                  </button>
                </div>

                {showInsights && (
                  <div className="research-visuals">
                    <EvidenceMap sources={message.sources} />
                    <ResearchTrace
                      timings={message.timings}
                      sources={message.sources}
                    />
                  </div>
                )}

                <div className="sources-section">
                  <button
                    type="button"
                    className="sources-toggle"
                    onClick={() => setExpanded((value) => !value)}
                  >
                    <span>
                      <Icon name="file" />
                      {message.sources.length} evidence sources
                    </span>
                    <span>{expanded ? "−" : "+"}</span>
                  </button>

                  {expanded && (
                    <div className="source-list">
                      {message.sources.map((source, index) => (
                        <div
                          className="source-card"
                          key={source.id || index}
                        >
                          <div className="source-heading">
                            <span className="source-id">
                              {source.id || `S${index + 1}`}
                            </span>
                            <span className="source-page">
                              Page {source.page ?? "N/A"}
                            </span>
                          </div>

                          <strong className="source-name">
                            {getSourceName(source)}
                          </strong>

                          <p>
                            {source.text ||
                              source.content ||
                              "Evidence retrieved from document."}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}

            <div className="message-footer">
              <button type="button" onClick={copyAnswer}>
                <Icon name={copied ? "check" : "copy"} />
                {copied ? "Copied" : "Copy answer"}
              </button>

              {message.latency !== undefined && (
                <span>
                  <Icon name="clock" />
                  {formatDuration(message.latency)}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [checking, setChecking] = useState(true);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState("");

  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const loadingRef = useRef(false);

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (!response.ok) throw new Error("Health check failed.");

        const data = await response.json();

        if (active) setConnected(data.status === "ok");
      } catch (err) {
        console.error("Backend connection error:", err);
        if (active) setConnected(false);
      } finally {
        if (active) setChecking(false);
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 30000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, loading]);

  async function uploadPdfs(fileList) {
    const files = Array.from(fileList || []);

    if (!files.length || uploading) return;

    const invalid = files.filter(
      (file) =>
        file.type !== "application/pdf" &&
        !file.name.toLowerCase().endsWith(".pdf")
    );

    if (invalid.length) {
      setError(
        `Only PDF files are supported: ${invalid
          .map((file) => file.name)
          .join(", ")}`
      );
      return;
    }

    const oversized = files.filter(
      (file) => file.size > 25 * 1024 * 1024
    );

    if (oversized.length) {
      setError(
        `Each PDF must be 25 MB or smaller: ${oversized
          .map((file) => file.name)
          .join(", ")}`
      );
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setUploading(true);
    setError("");
    setUploadProgress(`Uploading ${files.length} PDF(s)...`);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `PDF upload failed (${response.status}).`
        );
      }

      const returned = Array.isArray(data.documents)
        ? data.documents
        : [];

      if (!returned.length) {
        throw new Error(
          "The backend did not return an indexed documents array."
        );
      }

      const normalized = returned.map((item) => ({
        id: item.document_id || item.id,
        filename: item.filename || item.name || "Uploaded PDF",
        pages: item.pages ?? item.page_count ?? null,
        chunks: item.chunks ?? item.chunk_count ?? null,
        status: item.status || "indexed",
      })).filter((item) => item.id);

      if (!normalized.length) {
        throw new Error(
          "The upload response did not contain document IDs."
        );
      }

      setDocuments((previous) => {
        const existing = new Map(
          previous.map((document) => [document.id, document])
        );

        normalized.forEach((document) =>
          existing.set(document.id, document)
        );

        return Array.from(existing.values());
      });

      setUploadProgress(
        `${normalized.length} PDF(s) added to your research library.`
      );
    } catch (err) {
      console.error("PDF upload error:", err);
      setError(err.message || "Unable to upload PDFs.");
      setUploadProgress("");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function handleFileSelection(event) {
    uploadPdfs(event.target.files);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    uploadPdfs(event.dataTransfer.files);
  }

  function removeDocument(documentId) {
    setDocuments((previous) =>
      previous.filter((document) => document.id !== documentId)
    );
  }

  function newConversation() {
    if (loadingRef.current) return;

    setMessages([]);
    setHistory([]);
    setInput("");
    setError("");

    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  async function sendMessage(value) {
    const question = (value ?? input).trim();

    if (!question || loadingRef.current) return;

    if (!connected) {
      setError("Backend is not connected. Please try again.");
      return;
    }

    if (documents.length === 0) {
      setError("Upload at least one PDF before asking a question.");
      return;
    }

    if (question.length > 4000) {
      setError("Your question is too long. Maximum 4000 characters.");
      return;
    }

    loadingRef.current = true;

    const previousHistory = [...history];

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      time: formatTime(),
    };

    setMessages((previous) => [...previous, userMessage]);
    setInput("");
    setError("");
    setLoading(true);
    setSidebarOpen(false);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          history: previousHistory.slice(-12),
          document_ids: documents.map((document) => document.id),
        }),
      });

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error("Invalid response received from the backend.");
      }

      if (!response.ok) {
        const detail = data.detail;

        let message =
          typeof detail === "string"
            ? detail
            : `Request failed (${response.status}).`;

        if (response.status === 429) {
          message = "Gemini API rate limit reached. Please try later.";
        } else if (response.status === 503) {
          message = "Gemini is temporarily unavailable. Please try again.";
        } else if (
          response.status === 401 ||
          response.status === 403
        ) {
          message = "API authentication failed. Check backend configuration.";
        }

        throw new Error(message);
      }

      if (
        typeof data.answer !== "string" ||
        !data.answer.trim()
      ) {
        throw new Error("The AI returned an empty response.");
      }

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        time: formatTime(),
        sources: Array.isArray(data.sources) ? data.sources : [],
        latency: data.latency_seconds,
        timings: data.timings || {},
        warnings: Array.isArray(data.warnings) ? data.warnings : [],
        evidenceStatus: data.evidence_status || "unknown",
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);

      setHistory((previous) => [
        ...previous,
        { role: "user", content: question },
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      console.error("AtlasQuery chat error:", err);
      setError(
        err.message || "Unable to generate an answer. Please try again."
      );
    } finally {
      loadingRef.current = false;
      setLoading(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  async function retryConnection() {
    setChecking(true);

    try {
      const response = await fetch(`${API_URL}/api/health`);
      const data = await response.json();

      if (!response.ok || data.status !== "ok") {
        throw new Error("Backend unavailable.");
      }

      setConnected(true);
      setError("");
    } catch {
      setConnected(false);
      setError("Unable to connect to the backend.");
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="atlas-app">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        hidden
        onChange={handleFileSelection}
      />

      <aside
        className={`sidebar ${sidebarOpen ? "" : "sidebar-hidden"
          }`}
      >
        <div className="brand">
          <div className="brand-icon">
            <Icon name="sparkle" />
          </div>

          <div className="brand-info">
            <h2>AtlasQuery</h2>
            <span>DOCUMENT INTELLIGENCE</span>
          </div>

          <button
            type="button"
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            ×
          </button>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={newConversation}
          disabled={loading}
        >
          <Icon name="plus" />
          New conversation
        </button>

        <div className="sidebar-section">
          <div className="section-heading">WORKSPACE</div>

          <div className="sidebar-item active">
            <Icon name="book" />
            <span>Document research</span>
          </div>

          <div className="sidebar-item">
            <Icon name="search" />
            <span>Knowledge explorer</span>
            <span className="coming-soon">SOON</span>
          </div>
        </div>

        <div className="sidebar-section documents-sidebar-section">
          <div className="documents-section-heading">
            <span>YOUR DOCUMENTS</span>
            <span className="document-count">{documents.length}</span>
          </div>

          <div
            className={`upload-dropzone ${isDragging ? "dragging" : ""
              }`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="upload-dropzone-icon">
              <Icon name="upload" />
            </div>

            <div className="upload-dropzone-copy">
              <strong>
                {uploading ? "Indexing PDFs..." : "Add research PDFs"}
              </strong>
              <span>PDF files · Up to 25 MB each</span>
            </div>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              aria-label="Upload PDFs"
            >
              {uploading ? "..." : "+"}
            </button>
          </div>

          {uploadProgress && (
            <div className="upload-feedback">{uploadProgress}</div>
          )}

          <div className="uploaded-document-list">
            {documents.map((document) => (
              <div
                className="uploaded-document"
                key={document.id}
                title={document.filename}
              >
                <span className="uploaded-pdf-icon">PDF</span>

                <div className="uploaded-document-copy">
                  <strong>{document.filename}</strong>
                  <span>
                    {document.pages ?? "—"} pages
                    {document.chunks != null
                      ? ` · ${document.chunks} chunks`
                      : ""}
                  </span>
                </div>

                <button
                  type="button"
                  className="remove-document"
                  onClick={() => removeDocument(document.id)}
                  aria-label={`Remove ${document.filename} from this chat`}
                  title="Remove from this chat"
                >
                  ×
                </button>
              </div>
            ))}

            {documents.length === 0 && (
              <p className="documents-empty">
                Your uploaded PDFs will appear here.
              </p>
            )}
          </div>
        </div>

        <div className="sidebar-section history-section">
          <div className="section-heading">RECENT ACTIVITY</div>

          {messages.filter((message) => message.role === "user").length === 0 ? (
            <div className="empty-history">
              Your conversations will appear here.
            </div>
          ) : (
            messages
              .filter((message) => message.role === "user")
              .map((message) => (
                <div
                  className="history-item"
                  key={message.id}
                  title={message.content}
                >
                  <span>◷</span>
                  <span>{message.content}</span>
                </div>
              ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="backend-status">
            <span
              className={`status-dot ${connected ? "online" : "offline"
                }`}
            />
            <span>
              {checking
                ? "Checking backend..."
                : connected
                  ? "Backend connected"
                  : "Backend disconnected"}
            </span>
          </div>

          <button
            type="button"
            className="connection-retry"
            onClick={retryConnection}
            disabled={checking}
          >
            {checking ? "Checking..." : "Refresh connection"}
          </button>

          <div className="sidebar-profile">
            <div className="profile-avatar">A</div>
            <div>
              <strong>Research Workspace</strong>
              <span>Personal environment</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-workspace">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && (
              <button
                type="button"
                className="menu-button"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
              >
                <Icon name="menu" />
              </button>
            )}

            <div>
              <div className="topbar-eyebrow">
                YOUR AI RESEARCH WORKSPACE
              </div>
              <h1>Document Research</h1>
            </div>
          </div>

          <div className="topbar-right">
            <div className="topbar-document-count">
              <Icon name="file" />
              {documents.length} documents
            </div>

            <div
              className={`connection-pill ${connected ? "connected" : ""
                }`}
            >
              <span
                className={`status-dot ${connected ? "online" : "offline"
                  }`}
              />
              {checking
                ? "Connecting..."
                : connected
                  ? "RAG Engine Active"
                  : "API Offline"}
            </div>
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">
                <Icon name="sparkle" />
              </div>

              <div className="welcome-eyebrow">
                INTELLIGENCE STARTS HERE
              </div>

              <h2>
                Explore your documents.
                <br />
                <span>Discover what matters.</span>
              </h2>

              <p className="welcome-description">
                Upload your research papers and reports. Ask questions
                across multiple PDFs and receive evidence-backed answers
                powered by hybrid retrieval.
              </p>

              <button
                type="button"
                className="welcome-upload-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                <Icon name="upload" />
                {uploading ? "Uploading..." : "Upload research PDFs"}
              </button>

              <div className="suggestion-grid">
                {suggestions.map((item) => (
                  <button
                    type="button"
                    className="suggestion-card"
                    key={item.title}
                    onClick={() => sendMessage(item.question)}
                    disabled={
                      loading || !connected || documents.length === 0
                    }
                  >
                    <span className="suggestion-icon">{item.icon}</span>
                    <div>
                      <strong>{item.title}</strong>
                      <p>{item.description}</p>
                    </div>
                    <span className="suggestion-arrow">↗</span>
                  </button>
                ))}
              </div>

              <div className="engine-info">
                <span>✳</span>
                Powered by Gemini AI
                <span>·</span>
                FastEmbed + Qdrant
                <span>·</span>
                Hybrid retrieval
              </div>

              {!connected && !checking && (
                <button
                  type="button"
                  className="connection-retry"
                  onClick={retryConnection}
                >
                  Reconnect to backend
                </button>
              )}
            </div>
          ) : (
            <div className="conversation-container">
              <div className="conversation-heading">
                <span>CONVERSATION</span>
                <button
                  type="button"
                  onClick={newConversation}
                  disabled={loading}
                >
                  <Icon name="plus" />
                  New chat
                </button>
              </div>

              {messages.map((message) => (
                <Message message={message} key={message.id} />
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="assistant-avatar">
                    <Icon name="sparkle" />
                  </div>

                  <div className="message-content">
                    <div className="message-label">
                      AtlasQuery
                      <span className="thinking-label">REASONING</span>
                    </div>

                    <div className="loading-indicator">
                      <span />
                      <span />
                      <span />
                      <p>
                        Searching {documents.length} document(s) and
                        generating an answer...
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button
              type="button"
              onClick={() => setError("")}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        <footer className="input-area">
          <form className="input-container" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                documents.length
                  ? "Ask anything across your uploaded PDFs..."
                  : "Upload PDFs to start your research..."
              }
              rows={2}
              maxLength={4000}
              disabled={loading || documents.length === 0}
            />

            <div className="input-bottom">
              <div className="input-hint">
                <span className="input-engine-icon">✳</span>
                <span>Gemini AI</span>
                <span className="input-divider">·</span>
                <span>Multi-document RAG</span>
                <span className="input-divider">·</span>
                <span>{documents.length} PDFs</span>
              </div>

              <button
                type="submit"
                className="send-button"
                disabled={
                  !input.trim() ||
                  loading ||
                  !connected ||
                  documents.length === 0
                }
                aria-label="Send message"
              >
                {loading ? "•••" : "↑"}
              </button>
            </div>
          </form>

          <div className="disclaimer">
            AI-generated responses may contain errors. Always verify
            important claims against your original documents.
          </div>
        </footer>
      </main>
    </div>
  );
}
