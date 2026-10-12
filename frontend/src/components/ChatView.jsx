import React, { useState, useEffect, useRef } from "react";
import api from "../api/client";
import { Send, Bot, User, FileText, Copy, Check, Sparkles, Zap, DollarSign } from "lucide-react";

export default function ChatView() {
  const [messages, setMessages] = useState([
    {
      id: "intro",
      role: "assistant",
      text: "Hello! I am your enterprise RAG assistant. Ask me questions about your uploaded documents, or select a specific document to filter by.",
      citations: [],
      latency_ms: 0,
      tokens: 0,
      cost_usd: 0,
      from_cache: false,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [filterDoc, setFilterDoc] = useState("");
  const [copiedId, setCopiedId] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Load documents for scoping dropdown
    api
      .get("/documents")
      .then((res) => setDocuments(res.data || []))
      .catch((err) => console.error("Failed to load docs", err));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const queryText = input.trim();
    setInput("");

    const userMsg = {
      id: "u_" + Date.now(),
      role: "user",
      text: queryText,
      target_doc: filterDoc,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const payload = { query: queryText };
      if (filterDoc) payload.filter_doc = filterDoc;

      const res = await api.post("/ask", payload);
      const data = res.data;

      const botMsg = {
        id: "b_" + Date.now(),
        role: "assistant",
        text: data.answer || "No response received.",
        citations: data.citations || [],
        tokens: data.tokens || 0,
        cost_usd: data.cost_usd || 0,
        from_cache: data.from_cache || false,
        latency_ms: data.latency_ms || 0,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errMsg = err.response?.data?.detail || "Error communicating with server.";
      setMessages((prev) => [
        ...prev,
        {
          id: "err_" + Date.now(),
          role: "assistant",
          text: `⚠️ ${errMsg}`,
          citations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", padding: "20px", display: "flex", flexDirection: "column", height: "calc(100vh - 80px)" }}>
      
      {/* Scope / Filter Bar */}
      <div className="card-glass" style={{ padding: "12px 18px", marginBottom: "16px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Sparkles size={18} color="var(--primary)" />
          <span style={{ fontSize: "0.88rem", fontWeight: 700 }}>Query Scope:</span>
          <select
            value={filterDoc}
            onChange={(e) => setFilterDoc(e.target.value)}
            style={{
              padding: "6px 12px",
              borderRadius: "8px",
              border: "1px solid var(--border)",
              fontSize: "0.85rem",
              background: "#ffffff",
              fontWeight: 500
            }}
          >
            <option value="">All Documents (Global Workspace Search)</option>
            {documents.map((d) => (
              <option key={d.doc_id} value={d.source_filename}>
                📄 {d.source_filename}
              </option>
            ))}
          </select>
        </div>

        <div style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
          {documents.length} document(s) indexed in your tenant
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="card-glass" style={{ flex: 1, padding: "20px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "16px" }}>
        {messages.map((m) => {
          const isUser = m.role === "user";
          return (
            <div
              key={m.id}
              style={{
                display: "flex",
                flexDirection: isUser ? "row-reverse" : "row",
                gap: "12px",
                alignItems: "flex-start",
              }}
            >
              {/* Avatar */}
              <div
                style={{
                  width: "36px",
                  height: "36px",
                  borderRadius: "10px",
                  background: isUser ? "var(--primary)" : "#f1f5f9",
                  color: isUser ? "white" : "var(--primary)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                {isUser ? <User size={18} /> : <Bot size={18} />}
              </div>

              {/* Bubble */}
              <div
                style={{
                  maxWidth: "80%",
                  background: isUser ? "var(--primary)" : "#ffffff",
                  color: isUser ? "white" : "var(--text-main)",
                  border: isUser ? "none" : "1px solid var(--border)",
                  borderRadius: "14px",
                  padding: "14px 18px",
                  boxShadow: "0 2px 8px rgba(0, 0, 0, 0.04)",
                  position: "relative",
                }}
              >
                <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, fontSize: "0.95rem" }}>
                  {m.text}
                </div>

                {/* Citations block for assistant */}
                {!isUser && m.citations && m.citations.length > 0 && (
                  <div style={{ marginTop: "12px", paddingTop: "10px", borderTop: "1px dashed var(--border)" }}>
                    <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#64748b", marginBottom: "6px" }}>
                      Source References:
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {m.citations.map((c, i) => (
                        <span
                          key={i}
                          style={{
                            background: "#f1f5f9",
                            border: "1px solid #e2e8f0",
                            borderRadius: "6px",
                            padding: "3px 8px",
                            fontSize: "0.75rem",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            color: "#334155",
                          }}
                        >
                          <FileText size={12} color="var(--primary)" />
                          <strong>{c.source_filename}</strong> (Page {c.page_number})
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Telemetry metadata footer */}
                {!isUser && m.id !== "intro" && (
                  <div
                    style={{
                      marginTop: "10px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      fontSize: "0.74rem",
                      color: "#94a3b8",
                    }}
                  >
                    <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                      {m.from_cache ? (
                        <span style={{ color: "#16a34a", fontWeight: 600, display: "flex", alignItems: "center", gap: "2px" }}>
                          <Zap size={12} /> Cache Hit (&lt;5ms)
                        </span>
                      ) : (
                        <span>Latency: {m.latency_ms ? `${m.latency_ms}ms` : "~1.2s"}</span>
                      )}
                      <span>Tokens: {m.tokens}</span>
                      <span>Cost: ${m.cost_usd ? m.cost_usd.toFixed(6) : "0.00"}</span>
                    </div>

                    <button
                      onClick={() => handleCopy(m.id, m.text)}
                      style={{
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        color: copiedId === m.id ? "#16a34a" : "#94a3b8",
                        display: "flex",
                        alignItems: "center",
                        gap: "3px",
                      }}
                    >
                      {copiedId === m.id ? <Check size={13} /> : <Copy size={13} />}
                      {copiedId === m.id ? "Copied" : "Copy"}
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && (
          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <div style={{ width: "36px", height: "36px", borderRadius: "10px", background: "#f1f5f9", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--primary)" }}>
              <Bot size={18} />
            </div>
            <div style={{ background: "#ffffff", border: "1px solid var(--border)", borderRadius: "14px", padding: "12px 18px", fontSize: "0.9rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "8px" }}>
              <div className="spinner" style={{ width: "14px", height: "14px", border: "2px solid #cbd5e1", borderTopColor: "var(--primary)", borderRadius: "50%", animation: "spin 0.6s linear infinite" }}></div>
              Searching vector store and generating grounded answer...
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <form onSubmit={handleSend} style={{ marginTop: "16px", display: "flex", gap: "10px" }}>
        <input
          className="input-field"
          type="text"
          placeholder={filterDoc ? `Ask about "${filterDoc}"...` : "Ask any question about your documents..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          style={{ flex: 1 }}
        />
        <button type="submit" className="btn-primary" disabled={loading || !input.trim()} style={{ display: "flex", alignItems: "center", gap: "8px", padding: "0 22px" }}>
          <Send size={16} />
          Send
        </button>
      </form>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
