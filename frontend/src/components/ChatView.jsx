import React, { useState, useEffect, useRef } from "react";
import api from "../api/client";
import CitationInspector from "./CitationInspector";
import { 
  Send, Bot, User, FileText, Copy, Check, Sparkles, Zap, 
  Mic, MicOff, Volume2, VolumeX, Trash2, RotateCcw, 
  Plus, MessageSquare, Download, ChevronRight, PanelLeftClose, PanelLeft,
  Search, ShieldCheck
} from "lucide-react";

export default function ChatView() {
  const [messages, setMessages] = useState([]);
  const [rawHistory, setRawHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [documents, setDocuments] = useState([]);
  const [filterDoc, setFilterDoc] = useState("");
  const [copiedId, setCopiedId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeCitation, setActiveCitation] = useState(null);

  // Voice States
  const [isRecording, setIsRecording] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState("");
  const [autoVoiceOver, setAutoVoiceOver] = useState(false);
  const [playingMessageId, setPlayingMessageId] = useState(null);

  const messagesEndRef = useRef(null);
  const currentAudioRef = useRef(null);
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Load documents and chat history
  useEffect(() => {
    api
      .get("/documents")
      .then((res) => setDocuments(res.data || []))
      .catch((err) => console.error("Failed to load documents", err));

    loadHistory();

    return () => {
      stopVoicePlayback();
      stopRecording();
    };
  }, []);

  const loadHistory = async () => {
    try {
      setHistoryLoading(true);
      const res = await api.get("/chat/history");
      const historyItems = res.data || [];
      setRawHistory(historyItems);

      if (historyItems.length === 0) {
        setMessages([]);
      } else {
        const parsed = [];
        historyItems.forEach((h) => {
          parsed.push({
            id: `u_${h.id}`,
            role: "user",
            text: h.user_query,
            target_doc: h.target_doc || "",
            timestamp: h.timestamp,
          });
          parsed.push({
            id: `b_${h.id}`,
            role: "assistant",
            text: h.bot_answer,
            citations: h.citations || [],
            tokens: h.tokens || 0,
            cost_usd: h.cost_usd || 0,
            latency_ms: h.latency_ms || 0,
            from_cache: false,
            timestamp: h.timestamp,
          });
        });
        setMessages(parsed);
      }
    } catch (err) {
      console.error("Failed to load chat history", err);
    } finally {
      setHistoryLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    stopVoicePlayback();
  };

  const handleClearHistory = async () => {
    if (!window.confirm("Are you sure you want to permanently clear your chat history?")) return;
    try {
      await api.delete("/chat/history");
      setRawHistory([]);
      setMessages([]);
      stopVoicePlayback();
    } catch (err) {
      alert("Failed to clear chat history: " + (err.response?.data?.detail || err.message));
    }
  };

  const handleExportBrief = () => {
    if (messages.length === 0) {
      alert("No messages to export yet.");
      return;
    }

    let brief = `# Enterprise AI RAG Intelligence Brief\nGenerated: ${new Date().toLocaleString()}\nDocument Filter: ${filterDoc || "All Tenant Documents"}\n\n---\n\n`;
    
    messages.forEach((m) => {
      if (m.role === "user") {
        brief += `### 👤 Question:\n${m.text}\n\n`;
      } else {
        brief += `### 🤖 Grounded Answer:\n${m.text}\n\n`;
        if (m.citations && m.citations.length > 0) {
          brief += `**Sources:**\n`;
          m.citations.forEach((c) => {
            brief += `- ${c.source_filename} (Page ${c.page_number})\n`;
          });
          brief += `\n`;
        }
        brief += `---\n\n`;
      }
    });

    const blob = new Blob([brief], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rag_brief_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Submit Text Query
  const handleSend = async (e) => {
    if (e) e.preventDefault();
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

      // Update history sidebar
      setRawHistory((prev) => [
        { id: Date.now(), user_query: queryText, timestamp: new Date().toLocaleTimeString() },
        ...prev,
      ]);

      if (autoVoiceOver && data.answer) {
        playVoiceOver(botMsg.id, data.answer);
      }
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

  // Voice Playback (TTS)
  const stopVoicePlayback = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setPlayingMessageId(null);
  };

  const playVoiceOver = async (msgId, text) => {
    if (playingMessageId === msgId) {
      stopVoicePlayback();
      return;
    }
    stopVoicePlayback();
    setPlayingMessageId(msgId);

    try {
      const res = await api.post(
        "/voice/tts",
        { text: text.slice(0, 1500) },
        { responseType: "blob" }
      );
      const audioUrl = URL.createObjectURL(res.data);
      const audio = new Audio(audioUrl);
      currentAudioRef.current = audio;

      audio.onended = () => {
        setPlayingMessageId(null);
        currentAudioRef.current = null;
      };
      audio.onerror = () => fallbackBrowserTTS(msgId, text);
      await audio.play();
    } catch (err) {
      fallbackBrowserTTS(msgId, text);
    }
  };

  const fallbackBrowserTTS = (msgId, text) => {
    if (!window.speechSynthesis) {
      setPlayingMessageId(null);
      return;
    }
    const cleanText = text.replace(/[*_#`[\]()]/g, "");
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.onend = () => setPlayingMessageId(null);
    utterance.onerror = () => setPlayingMessageId(null);
    window.speechSynthesis.speak(utterance);
  };

  // Voice Input (Speech-to-Text)
  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const startRecording = async () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      try {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onstart = () => {
          setIsRecording(true);
          setRecordingStatus("Listening... speak your question now");
        };

        recognition.onresult = (event) => {
          const transcript = Array.from(event.results)
            .map((result) => result[0].transcript)
            .join("");
          setInput(transcript);
        };

        recognition.onerror = () => {
          setRecordingStatus("Speech error, click mic to retry.");
          stopRecording();
        };

        recognition.onend = () => {
          setIsRecording(false);
          setRecordingStatus("");
        };

        recognitionRef.current = recognition;
        recognition.start();
        return;
      } catch (e) {
        console.warn("Native recognition fallback", e);
      }
    }

    // MediaRecorder fallback
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        stream.getTracks().forEach((track) => track.stop());

        setLoading(true);
        setRecordingStatus("Synthesizing audio & querying RAG...");

        try {
          const formData = new FormData();
          formData.append("file", audioBlob, "voice.wav");

          const res = await api.post("/voice/conversation", formData, { responseType: "blob" });
          const rawUserText = res.headers["x-user-text"];
          const rawLlmText = res.headers["x-llm-text"];
          const userText = rawUserText ? decodeURIComponent(rawUserText) : "Spoken question";
          const llmText = rawLlmText ? decodeURIComponent(rawLlmText) : "Audio response synthesized";

          const userMsg = { id: "u_" + Date.now(), role: "user", text: userText };
          const botMsg = { id: "b_" + Date.now(), role: "assistant", text: llmText };
          setMessages((prev) => [...prev, userMsg, botMsg]);

          const audioUrl = URL.createObjectURL(res.data);
          const audio = new Audio(audioUrl);
          currentAudioRef.current = audio;
          setPlayingMessageId(botMsg.id);
          audio.onended = () => setPlayingMessageId(null);
          await audio.play();
        } catch (err) {
          alert("Voice query failed: " + (err.response?.data?.detail || err.message));
        } finally {
          setLoading(false);
          setRecordingStatus("");
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingStatus("Recording audio... click mic again to send");
    } catch {
      alert("Microphone access unavailable or denied.");
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch {}
      recognitionRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try { mediaRecorderRef.current.stop(); } catch {}
      mediaRecorderRef.current = null;
    }
    setIsRecording(false);
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Starter prompts
  const starterPrompts = [
    { title: "Executive Summary", query: "Summarize the key findings, conclusions, and objectives of the document." },
    { title: "Technical Architecture", query: "What technical architecture, technologies, and methodologies are outlined?" },
    { title: "Risk & Limitations", query: "Extract all potential risks, caveats, bottlenecks, or constraints mentioned." },
    { title: "Action Items & Next Steps", query: "What actionable recommendations, milestones, and next steps are proposed?" },
  ];

  return (
    <div style={{ display: "flex", height: "calc(100vh - 72px)", overflow: "hidden", background: "#f8fafc" }}>
      
      {/* 1. Left Collapsible Thread Sidebar */}
      {sidebarOpen && (
        <aside style={{
          width: "280px",
          background: "#ffffff",
          borderRight: "1px solid #e2e8f0",
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
          transition: "width 0.2s ease",
          zIndex: 10
        }}>
          {/* New Chat Button */}
          <div style={{ padding: "16px 16px 12px" }}>
            <button
              onClick={startNewChat}
              className="btn-primary"
              style={{
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px",
                padding: "10px 14px",
                fontSize: "0.9rem"
              }}
            >
              <Plus size={18} />
              New Chat Thread
            </button>
          </div>

          {/* History Header */}
          <div style={{
            padding: "8px 16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "0.76rem",
            fontWeight: 700,
            textTransform: "uppercase",
            color: "#64748b",
            letterSpacing: "0.05em"
          }}>
            <span>Recent Sessions</span>
            <span>{rawHistory.length}</span>
          </div>

          {/* History Thread List */}
          <div style={{ flex: 1, overflowY: "auto", padding: "0 10px 16px", display: "flex", flexDirection: "column", gap: "4px" }}>
            {historyLoading ? (
              <div style={{ padding: "20px", textAlign: "center", color: "#94a3b8", fontSize: "0.82rem" }}>
                <RotateCcw className="animate-spin" size={16} style={{ margin: "0 auto 6px" }} />
                Loading sessions...
              </div>
            ) : rawHistory.length === 0 ? (
              <div style={{ padding: "24px 16px", textAlign: "center", color: "#94a3b8", fontSize: "0.82rem" }}>
                No past chat sessions yet. Ask your first question!
              </div>
            ) : (
              rawHistory.map((h, i) => (
                <div
                  key={h.id || i}
                  onClick={() => {
                    setInput(h.user_query);
                  }}
                  style={{
                    padding: "9px 12px",
                    borderRadius: "8px",
                    cursor: "pointer",
                    fontSize: "0.84rem",
                    color: "#334155",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.15s ease",
                    background: "transparent",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = "#f1f5f9"}
                  onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                >
                  <MessageSquare size={14} color="#64748b" style={{ flexShrink: 0 }} />
                  <span style={{
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    flex: 1,
                    fontWeight: 500
                  }}>
                    {h.user_query}
                  </span>
                </div>
              ))
            )}
          </div>

          {/* Sidebar Footer */}
          <div style={{ padding: "12px 16px", borderTop: "1px solid #f1f5f9", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 600 }}>
              {documents.length} Docs Indexed
            </span>
            {rawHistory.length > 0 && (
              <button
                onClick={handleClearHistory}
                style={{
                  background: "none",
                  border: "none",
                  color: "#ef4444",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px"
                }}
              >
                <Trash2 size={12} /> Clear All
              </button>
            )}
          </div>
        </aside>
      )}

      {/* 2. Main Chat Workspace */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
        
        {/* Top Control Bar */}
        <div style={{
          padding: "12px 20px",
          background: "#ffffff",
          borderBottom: "1px solid #e2e8f0",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "10px",
          zIndex: 5
        }}>
          {/* Left: Sidebar Toggle + Document Scope */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Hide Sessions Sidebar" : "Show Sessions Sidebar"}
              style={{
                background: "transparent",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                padding: "6px",
                color: "#64748b",
                cursor: "pointer",
                display: "flex",
                alignItems: "center"
              }}
            >
              {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
            </button>

            {/* Scope Dropdown */}
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Sparkles size={16} color="#2563eb" />
              <select
                value={filterDoc}
                onChange={(e) => setFilterDoc(e.target.value)}
                style={{
                  padding: "6px 12px",
                  borderRadius: "8px",
                  border: "1px solid #cbd5e1",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  color: "#0f172a",
                  background: "#ffffff",
                  maxWidth: "280px"
                }}
              >
                <option value="">All Documents (Tenant Scope)</option>
                {documents.map((d) => (
                  <option key={d.doc_id} value={d.source_filename}>
                    📄 {d.source_filename}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Right Controls */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {/* Auto Voice Over Switch */}
            <button
              onClick={() => setAutoVoiceOver(!autoVoiceOver)}
              title="Automatically read assistant responses aloud"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "6px 12px",
                borderRadius: "8px",
                fontSize: "0.82rem",
                fontWeight: 600,
                border: "1px solid " + (autoVoiceOver ? "#16a34a" : "#cbd5e1"),
                background: autoVoiceOver ? "#f0fdf4" : "#ffffff",
                color: autoVoiceOver ? "#16a34a" : "#64748b",
                cursor: "pointer",
                transition: "all 0.15s ease"
              }}
            >
              {autoVoiceOver ? <Volume2 size={15} /> : <VolumeX size={15} />}
              Auto Voice-Over: {autoVoiceOver ? "ON" : "OFF"}
            </button>

            {/* Export Brief */}
            <button
              onClick={handleExportBrief}
              title="Export this conversation as a grounded Markdown brief"
              className="btn-secondary"
              style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.82rem", padding: "6px 12px" }}
            >
              <Download size={14} />
              Export Brief
            </button>
          </div>
        </div>

        {/* Messages Stream Container */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column", gap: "18px" }}>
          
          {/* Welcome Screen & Starter Suggestions when empty */}
          {messages.length === 0 && (
            <div style={{ maxWidth: "720px", margin: "40px auto", textAlign: "center" }}>
              <div style={{
                width: "56px",
                height: "56px",
                background: "linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)",
                borderRadius: "16px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 16px",
                boxShadow: "0 10px 25px -5px rgba(37, 99, 235, 0.3)"
              }}>
                <Bot size={30} color="white" />
              </div>

              <h2 style={{ fontSize: "1.6rem", fontWeight: 800, color: "#0f172a", marginBottom: "8px" }}>
                Enterprise Document Intelligence
              </h2>
              <p style={{ color: "#64748b", fontSize: "0.95rem", marginBottom: "28px" }}>
                Ask questions across your ingested documents. Powered by hybrid keyword-vector retrieval, Upstash Redis sub-5ms caching, and hallucination-free citations.
              </p>

              {/* Starter Prompt Cards */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", textAlign: "left" }}>
                {starterPrompts.map((s, idx) => (
                  <div
                    key={idx}
                    className="suggestion-card"
                    onClick={() => {
                      setInput(s.query);
                    }}
                  >
                    <div style={{ fontWeight: 700, fontSize: "0.9rem", color: "#1e293b", marginBottom: "4px" }}>
                      {s.title}
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "#64748b", lineHeight: 1.4 }}>
                      {s.query}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Conversation Bubbles */}
          {messages.map((m) => {
            const isUser = m.role === "user";
            const isSpeaking = playingMessageId === m.id;

            return (
              <div
                key={m.id}
                style={{
                  display: "flex",
                  flexDirection: isUser ? "row-reverse" : "row",
                  gap: "12px",
                  alignItems: "flex-start",
                  maxWidth: "880px",
                  margin: "0 auto",
                  width: "100%"
                }}
              >
                {/* Avatar */}
                <div style={{
                  width: "36px",
                  height: "36px",
                  borderRadius: "10px",
                  background: isUser ? "linear-gradient(135deg, #2563eb, #3b82f6)" : "#ffffff",
                  color: isUser ? "white" : "#2563eb",
                  border: isUser ? "none" : "1px solid #e2e8f0",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  boxShadow: "0 2px 6px rgba(0,0,0,0.04)"
                }}>
                  {isUser ? <User size={18} /> : <Bot size={18} />}
                </div>

                {/* Bubble */}
                <div
                  className={m.from_cache ? "glow-cache-hit" : ""}
                  style={{
                    maxWidth: "80%",
                    background: isUser ? "#2563eb" : "#ffffff",
                    color: isUser ? "white" : "#0f172a",
                    border: isUser ? "none" : "1px solid #e2e8f0",
                    borderRadius: "16px",
                    padding: "16px 20px",
                    boxShadow: isUser 
                      ? "0 4px 14px rgba(37, 99, 235, 0.2)" 
                      : "0 4px 18px rgba(15, 23, 42, 0.04)",
                    position: "relative"
                  }}
                >
                  {/* Message Text */}
                  <div style={{
                    whiteSpace: "pre-wrap",
                    lineHeight: 1.65,
                    fontSize: "0.95rem",
                    fontWeight: 400
                  }}>
                    {m.text}
                  </div>

                  {/* Interactive Citations Block */}
                  {!isUser && m.citations && m.citations.length > 0 && (
                    <div style={{ marginTop: "14px", paddingTop: "12px", borderTop: "1px dashed #e2e8f0" }}>
                      <div style={{ fontSize: "0.76rem", fontWeight: 700, color: "#64748b", textTransform: "uppercase", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                        <ShieldCheck size={14} color="#16a34a" /> Grounded Evidence (Click to inspect):
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                        {m.citations.map((c, i) => (
                          <button
                            key={i}
                            onClick={() => setActiveCitation(c)}
                            title="Click to view verbatim chunk context & similarity score"
                            style={{
                              background: "#f8fafc",
                              border: "1px solid #cbd5e1",
                              borderRadius: "8px",
                              padding: "4px 10px",
                              fontSize: "0.78rem",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "6px",
                              color: "#1e293b",
                              cursor: "pointer",
                              transition: "all 0.15s ease"
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.borderColor = "#2563eb";
                              e.currentTarget.style.background = "#eff6ff";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.borderColor = "#cbd5e1";
                              e.currentTarget.style.background = "#f8fafc";
                            }}
                          >
                            <FileText size={13} color="#2563eb" />
                            <strong>{c.source_filename}</strong>
                            <span style={{ color: "#64748b" }}>• p.{c.page_number}</span>
                            <ChevronRight size={12} color="#94a3b8" />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Assistant Footer Telemetry & Actions */}
                  {!isUser && m.id !== "intro" && (
                    <div style={{
                      marginTop: "12px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      fontSize: "0.75rem",
                      color: "#94a3b8",
                      flexWrap: "wrap",
                      gap: "8px"
                    }}>
                      {/* Telemetry Stats */}
                      <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                        {m.from_cache ? (
                          <span style={{ color: "#16a34a", fontWeight: 700, display: "flex", alignItems: "center", gap: "3px" }}>
                            <Zap size={13} /> Sub-5ms Redis Cache Hit
                          </span>
                        ) : (
                          <span>Latency: {m.latency_ms ? `${m.latency_ms}ms` : "~1.1s"}</span>
                        )}
                        <span>Tokens: {m.tokens}</span>
                        <span>Cost: ${m.cost_usd ? m.cost_usd.toFixed(5) : "0.00"}</span>
                      </div>

                      {/* Action buttons */}
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        {/* Voice Over Button */}
                        <button
                          onClick={() => playVoiceOver(m.id, m.text)}
                          title={isSpeaking ? "Pause Voice Over" : "Listen with Voice Over"}
                          style={{
                            background: isSpeaking ? "#eff6ff" : "none",
                            border: isSpeaking ? "1px solid #93c5fd" : "none",
                            borderRadius: "6px",
                            padding: "3px 8px",
                            cursor: "pointer",
                            color: isSpeaking ? "#2563eb" : "#64748b",
                            display: "flex",
                            alignItems: "center",
                            gap: "4px",
                            fontWeight: isSpeaking ? 700 : 500
                          }}
                        >
                          {isSpeaking ? <VolumeX size={14} /> : <Volume2 size={14} />}
                          {isSpeaking ? "Speaking..." : "Voice Over"}
                        </button>

                        {/* Copy Button */}
                        <button
                          onClick={() => handleCopy(m.id, m.text)}
                          style={{
                            background: "none",
                            border: "none",
                            cursor: "pointer",
                            color: copiedId === m.id ? "#16a34a" : "#94a3b8",
                            display: "flex",
                            alignItems: "center",
                            gap: "3px"
                          }}
                        >
                          {copiedId === m.id ? <Check size={13} /> : <Copy size={13} />}
                          {copiedId === m.id ? "Copied" : "Copy"}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {/* Assistant Generation Spinner */}
          {loading && (
            <div style={{ display: "flex", gap: "12px", alignItems: "center", maxWidth: "880px", margin: "0 auto", width: "100%" }}>
              <div style={{ width: "36px", height: "36px", borderRadius: "10px", background: "#ffffff", border: "1px solid #e2e8f0", display: "flex", alignItems: "center", justifyContent: "center", color: "#2563eb" }}>
                <Bot size={18} />
              </div>
              <div style={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: "14px", padding: "14px 20px", fontSize: "0.9rem", color: "#64748b", display: "flex", alignItems: "center", gap: "10px" }}>
                <div style={{ width: "16px", height: "16px", border: "2px solid #cbd5e1", borderTopColor: "#2563eb", borderRadius: "50%", animation: "spin 0.6s linear infinite" }}></div>
                Synthesizing answer from retrieved document chunks...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Active Audio Waveform Banner when recording */}
        {isRecording && (
          <div style={{
            background: "#fef2f2",
            borderTop: "1px solid #fecaca",
            padding: "10px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            color: "#b91c1c"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
              <div style={{ display: "flex", gap: "3px", alignItems: "center", height: "24px" }}>
                <div className="wave-bar" />
                <div className="wave-bar" />
                <div className="wave-bar" />
                <div className="wave-bar" />
                <div className="wave-bar" />
              </div>
              <span style={{ fontSize: "0.88rem", fontWeight: 600 }}>
                {recordingStatus || "Listening to speech... Speak clearly"}
              </span>
            </div>

            <button
              onClick={stopRecording}
              style={{
                background: "#ef4444",
                color: "white",
                border: "none",
                borderRadius: "8px",
                padding: "6px 14px",
                cursor: "pointer",
                fontSize: "0.82rem",
                fontWeight: 600
              }}
            >
              Stop & Send
            </button>
          </div>
        )}

        {/* Input Bar Area */}
        <div style={{ padding: "16px 24px 20px", background: "#ffffff", borderTop: "1px solid #e2e8f0" }}>
          <form
            onSubmit={handleSend}
            style={{
              maxWidth: "880px",
              margin: "0 auto",
              display: "flex",
              alignItems: "center",
              gap: "10px",
              background: "#ffffff",
              border: "1px solid #cbd5e1",
              borderRadius: "14px",
              padding: "6px 8px 6px 14px",
              boxShadow: "0 4px 16px rgba(0,0,0,0.04)"
            }}
          >
            {/* Microphone Button */}
            <button
              type="button"
              onClick={toggleRecording}
              title={isRecording ? "Stop voice recording" : "Speak question with microphone"}
              style={{
                background: isRecording ? "#fee2e2" : "transparent",
                border: "none",
                color: isRecording ? "#ef4444" : "#64748b",
                cursor: "pointer",
                padding: "8px",
                borderRadius: "10px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}
            >
              {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
            </button>

            {/* Text Input */}
            <input
              type="text"
              placeholder={filterDoc ? `Ask about "${filterDoc}" (or click mic)...` : "Ask any question about your documents (or click mic)..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              style={{
                flex: 1,
                border: "none",
                outline: "none",
                fontSize: "0.95rem",
                background: "transparent",
                fontFamily: "inherit",
                color: "#0f172a"
              }}
            />

            {/* Send Button */}
            <button
              type="submit"
              className="btn-primary"
              disabled={loading || !input.trim()}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "8px 18px",
                borderRadius: "10px",
                fontSize: "0.88rem"
              }}
            >
              <Send size={15} />
              Send
            </button>
          </form>

          <div style={{ textAlign: "center", fontSize: "0.72rem", color: "#94a3b8", marginTop: "8px" }}>
            Answers are grounded strictly in your tenant documents • Press Enter to send
          </div>
        </div>
      </div>

      {/* 3. Citation Inspector Drawer */}
      <CitationInspector
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
