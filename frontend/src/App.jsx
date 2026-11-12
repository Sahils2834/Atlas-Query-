
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
    description: "Explain important concepts from your documents.",
    question: "Explain the main concepts covered in my documents.",
  },
  {
    icon: "▤",
    title: "Summarize research",
    description: "Discover the key findings and conclusions.",
    question: "Give me a summary of the key findings in my documents.",
  },
  {
    icon: "⇄",
    title: "Compare information",
    description: "Identify similarities and differences.",
    question: "Identify important similarities and differences in my documents.",
  },
  {
    icon: "✧",
    title: "Explore deeper",
    description: "Find important insights from your documents.",
    question: "What are the most important insights from the indexed documents?",
  },
];

function Icon({ name }) {
  const icons = {
    plus: "＋",
    search: "⌕",
    book: "▤",
    menu: "☰",
    send: "↑",
    copy: "▢",
    check: "✓",
    close: "×",
    file: "▤",
    sparkle: "✳",
    clock: "◷",
    arrow: "↗",
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
    source.source || source.filename || source.document || "Document source"
  ).split(/[\\/]/).pop();
}

function formatDuration(value) {
  const number = Number(value);

  if (!Number.isFinite(number) || number < 0) return "—";

  return `${number.toFixed(2)}s`;
}

function EvidenceMap({ sources }) {
  const [activeSource, setActiveSource] = useState(0);

  if (!sources?.length) return null;

  const active = sources[activeSource] || sources[0];
  const count = Math.min(sources.length, 6);
  const visibleSources = sources.slice(0, count);

  const positions = [
    { x: 50, y: 13 },
    { x: 83, y: 37 },
    { x: 78, y: 75 },
    { x: 50, y: 88 },
    { x: 19, y: 72 },
    { x: 16, y: 34 },
  ];

  return (
    <section className="insight-panel evidence-panel">
      <div className="insight-heading">
        <div>
          <span className="panel-kicker">RETRIEVAL VISUALIZATION</span>
          <h3>Evidence constellation</h3>
          <p>Explore the document fragments behind this answer.</p>
        </div>

        <span className="insight-count">
          {sources.length} {sources.length === 1 ? "SOURCE" : "SOURCES"}
        </span>
      </div>

      <div className="constellation-layout">
        <div className="constellation">
          <div className="constellation-orbit orbit-one" />
          <div className="constellation-orbit orbit-two" />

          <div className="constellation-center">
            <span>✳</span>
            <small>YOUR QUESTION</small>
          </div>

          <svg
            className="constellation-lines"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            {visibleSources.map((source, index) => {
              const position = positions[index];

              return (
                <line
                  key={source.id || index}
                  x1="50"
                  y1="50"
                  x2={position.x}
                  y2={position.y}
                  className={
                    activeSource === index
                      ? "constellation-line active"
                      : "constellation-line"
                  }
                />
              );
            })}
          </svg>

          {visibleSources.map((source, index) => {
            const position = positions[index];

            return (
              <button
                key={source.id || index}
                type="button"
                className={`constellation-node ${activeSource === index ? "selected" : ""
                  }`}
                style={{
                  left: `${position.x}%`,
                  top: `${position.y}%`,
                }}
                onClick={() => setActiveSource(index)}
                aria-label={`Inspect ${getSourceName(source)}`}
                title={getSourceName(source)}
              >
                <span>{source.id || `S${index + 1}`}</span>
              </button>
            );
          })}
        </div>

        <div className="evidence-detail">
          <div className="evidence-detail-top">
            <span className="evidence-source-tag">
              {active.id || `S${activeSource + 1}`}
            </span>
            <span className="evidence-page">
              PAGE {active.page ?? "N/A"}
            </span>
          </div>

          <h4>{getSourceName(active)}</h4>

          <p>
            {active.text ||
              active.content ||
              "Evidence retrieved from this document."}
          </p>

          <div className="evidence-detail-footer">
            <span>
              <span className="evidence-status-dot" />
              Retrieved evidence
            </span>
            <span>
              {activeSource + 1} / {sources.length}
            </span>
          </div>

          {sources.length > 1 && (
            <div className="evidence-switcher">
              <button
                type="button"
                onClick={() =>
                  setActiveSource((current) =>
                    current === 0 ? sources.length - 1 : current - 1
                  )
                }
                aria-label="Previous source"
              >
                ←
              </button>

              <div className="evidence-switcher-dots">
                {sources.map((source, index) => (
                  <button
                    key={source.id || index}
                    type="button"
                    className={activeSource === index ? "active" : ""}
                    onClick={() => setActiveSource(index)}
                    aria-label={`Select source ${index + 1}`}
                  />
                ))}
              </div>

              <button
                type="button"
                onClick={() =>
                  setActiveSource((current) =>
                    (current + 1) % sources.length
                  )
                }
                aria-label="Next source"
              >
                →
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function ResearchPipeline({ timings = {} }) {
  const steps = [
    {
      number: "01",
      title: "Question",
      description: "User query received",
      time: null,
    },
    {
      number: "02",
      title: "Retrieval",
      description: "Hybrid document search",
      time: timings.retrieval_seconds ?? timings.retrieval,
    },
    {
      number: "03",
      title: "Reranking",
      description: "Evidence relevance ranking",
      time: null,
    },
    {
      number: "04",
      title: "Generation",
      description: "Gemini response",
      time: timings.generation_seconds ?? timings.generation,
    },
  ];

  return (
    <section className="insight-panel pipeline-panel">
      <div className="insight-heading">
        <div>
          <span className="panel-kicker">BEHIND THE ANSWER</span>
          <h3>Research pipeline</h3>
          <p>How your question moves through AtlasQuery.</p>
        </div>

        <span className="pipeline-live">
          <span />
          PROCESS COMPLETE
        </span>
      </div>

      <div className="pipeline-track">
        {steps.map((step, index) => (
          <div className="pipeline-step" key={step.number}>
            <div className={`pipeline-number step-${index}`}>
              {step.number}
            </div>

            <div className="pipeline-step-copy">
              <strong>{step.title}</strong>
              <span>{step.description}</span>

              {step.time !== null && step.time !== undefined && (
                <small>{formatDuration(step.time)}</small>
              )}
            </div>

            {index < 3 && <div className="pipeline-connector" />}
          </div>
        ))}
      </div>
    </section>
  );
}

function ResearchMetrics({ message }) {
  const timings = message.timings || {};
  const retrievalTime =
    timings.retrieval_seconds ?? timings.retrieval;
  const generationTime =
    timings.generation_seconds ?? timings.generation;

  const metrics = [
    {
      label: "RETRIEVAL",
      value: formatDuration(retrievalTime),
    },
    {
      label: "GENERATION",
      value: formatDuration(generationTime),
    },
    {
      label: "TOTAL TIME",
      value: formatDuration(message.latency),
    },
    {
      label: "EVIDENCE",
      value: String(message.sources?.length || 0).padStart(2, "0"),
    },
  ];

  const evidenceStatus = String(
    message.evidenceStatus || "unknown"
  ).toLowerCase();

  return (
    <section className="insight-panel metrics-panel">
      <div className="metrics-heading">
        <div>
          <span className="panel-kicker">SYSTEM TELEMETRY</span>
          <h3>Research activity</h3>
        </div>

        <span className="trace-live">
          <span />
          LIVE TRACE
        </span>
      </div>

      <div className="trace-metrics">
        {metrics.map((metric) => (
          <div className="trace-metric" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
      </div>

      {evidenceStatus !== "unknown" && (
        <div className="evidence-status-row">
          <span>Evidence status</span>
          <span className={`evidence-status-label ${evidenceStatus}`}>
            {evidenceStatus.replaceAll("_", " ")}
          </span>
        </div>
      )}
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
        {message.role === "user" ? (
          <>
            <div className="message-label">
              <span>You</span>
              <span className="message-time">{message.time}</span>
            </div>

            <div className="user-bubble">{message.content}</div>
          </>
        ) : (
          <>
            <div className="message-label">
              <span>AtlasQuery</span>
              <span className="message-time">{message.time}</span>
            </div>

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
                        Visual breakdown of the retrieved knowledge
                      </span>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      setShowInsights((previous) => !previous)
                    }
                    aria-expanded={showInsights}
                  >
                    {showInsights ? "Hide insights −" : "Show insights +"}
                  </button>
                </div>

                {showInsights && (
                  <div className="research-visuals">
                    <EvidenceMap sources={message.sources} />
                    <ResearchPipeline timings={message.timings} />
                    <ResearchMetrics message={message} />
                  </div>
                )}

                <div className="sources-section">
                  <button
                    type="button"
                    className="sources-toggle"
                    onClick={() => setExpanded((previous) => !previous)}
                    aria-expanded={expanded}
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

                          <div className="source-name">
                            {getSourceName(source)}
                          </div>

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
              <button
                type="button"
                onClick={copyAnswer}
                title="Copy answer"
              >
                <Icon name={copied ? "check" : "copy"} />
                {copied ? "Copied" : "Copy answer"}
              </button>

              {Number.isFinite(Number(message.latency)) &&
                message.latency !== undefined && (
                  <span>
                    <Icon name="clock" />
                    {Number(message.latency).toFixed(2)}s
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

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const loadingRef = useRef(false);

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch(`${API_URL}/api/health`);

        if (!response.ok) {
          throw new Error("Backend health check failed.");
        }

        const data = await response.json();

        if (active) {
          setConnected(data.status === "ok");
        }
      } catch (err) {
        console.error("Backend connection error:", err);

        if (active) {
          setConnected(false);
        }
      } finally {
        if (active) {
          setChecking(false);
        }
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

  function newConversation() {
    if (loadingRef.current) return;

    setMessages([]);
    setHistory([]);
    setInput("");
    setError("");

    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }

  async function sendMessage(value) {
    const question = (value ?? input).trim();

    if (!question || loadingRef.current) return;

    if (!connected) {
      setError(
        "Backend is not connected. Please check your FastAPI server."
      );
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

        let errorMessage =
          typeof detail === "string"
            ? detail
            : `Request failed with status ${response.status}.`;

        if (response.status === 429) {
          errorMessage =
            "API request limit reached. Please check your Gemini quota and try again later.";
        } else if (response.status === 503) {
          errorMessage =
            "Gemini service is temporarily unavailable. Please try again later.";
        } else if (response.status === 401 || response.status === 403) {
          errorMessage =
            "API authentication failed. Please check the backend API configuration.";
        } else if (response.status >= 500 && !detail) {
          errorMessage =
            "The backend encountered an error. Please try again.";
        }

        throw new Error(errorMessage);
      }

      if (
        !data.answer ||
        typeof data.answer !== "string" ||
        !data.answer.trim()
      ) {
        throw new Error(
          "The AI returned an empty response. Please try again."
        );
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
        requestId: data.request_id || null,
        tokenUsage: data.token_usage || {},
        finishReason: data.finish_reason || null,
      };

      setMessages((previous) => [...previous, assistantMessage]);

      setHistory((previous) => [
        ...previous,
        {
          role: "user",
          content: question,
        },
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (err) {
      console.error("AtlasQuery chat error:", err);

      setError(
        err.message || "Unable to generate an answer. Please try again."
      );
    } finally {
      loadingRef.current = false;
      setLoading(false);

      requestAnimationFrame(() => {
        textareaRef.current?.focus();
      });
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
        throw new Error("Backend is unavailable.");
      }

      setConnected(true);
      setError("");
    } catch {
      setConnected(false);
      setError("Unable to connect. Check that FastAPI is running.");
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="atlas-app">
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
          <span>New conversation</span>
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

        <div className="sidebar-section history-section">
          <div className="section-heading">RECENT ACTIVITY</div>

          {messages.filter((m) => m.role === "user").length === 0 ? (
            <div className="empty-history">
              Your conversations will appear here.
            </div>
          ) : (
            messages
              .filter((m) => m.role === "user")
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
                Ask questions, uncover insights, and explore your
                indexed research papers with AI-powered retrieval
                and evidence-backed answers.
              </p>

              <div className="suggestion-grid">
                {suggestions.map((item) => (
                  <button
                    type="button"
                    className="suggestion-card"
                    key={item.title}
                    onClick={() => sendMessage(item.question)}
                    disabled={loading || !connected}
                  >
                    <span className="suggestion-icon">
                      {item.icon}
                    </span>

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
                Hybrid document retrieval
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
                <Message
                  message={message}
                  key={message.id}
                />
              ))}

              {loading && (
                <div className="message-row assistant">
                  <div className="assistant-avatar">
                    <Icon name="sparkle" />
                  </div>

                  <div className="message-content">
                    <div className="message-label">
                      AtlasQuery
                      <span className="thinking-label">
                        REASONING
                      </span>
                    </div>

                    <div className="loading-indicator">
                      <span />
                      <span />
                      <span />

                      <p>
                        Searching documents and generating an answer...
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
          <form
            className="input-container"
            onSubmit={handleSubmit}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your indexed documents..."
              rows={2}
              maxLength={4000}
              disabled={loading}
            />

            <div className="input-bottom">
              <div className="input-hint">
                <span className="input-engine-icon">✳</span>
                <span>Gemini AI</span>
                <span className="input-divider">·</span>
                <span>Document RAG</span>
              </div>

              <button
                type="submit"
                className="send-button"
                disabled={!input.trim() || loading || !connected}
                aria-label="Send message"
              >
                {loading ? "•••" : "↑"}
              </button>
            </div>
          </form>

          <div className="disclaimer">
            AI-generated responses may contain errors.
            Always verify important claims against original documents.
          </div>
        </footer>
      </main>
    </div>
  );
}
