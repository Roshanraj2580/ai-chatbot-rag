import React, { useState, useEffect, useRef } from "react";
import api from "../api/client";
import { 
  Send, Bot, User, FileText, Copy, Check, Sparkles, Zap, 
  Mic, MicOff, Volume2, VolumeX, Trash2, RotateCcw, AlertCircle
} from "lucide-react";

export default function ChatView() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [documents, setDocuments] = useState([]);
  const [filterDoc, setFilterDoc] = useState("");
  const [copiedId, setCopiedId] = useState(null);

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

  // 1. Load documents and persistent chat history on mount
  useEffect(() => {
    // Load documents
    api
      .get("/documents")
      .then((res) => setDocuments(res.data || []))
      .catch((err) => console.error("Failed to load documents", err));

    // Load user's saved chat history
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

      if (historyItems.length === 0) {
        setMessages([
          {
            id: "intro",
            role: "assistant",
            text: "Hello! I am your enterprise RAG assistant. Ask me questions about your uploaded documents, or select a specific document to filter by.",
            citations: [],
            tokens: 0,
            cost_usd: 0,
            from_cache: false,
          },
        ]);
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
      setMessages([
        {
          id: "intro",
          role: "assistant",
          text: "Welcome! Ask me any question about your documents.",
          citations: [],
        },
      ]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm("Are you sure you want to clear your entire chat history?")) return;
    try {
      await api.delete("/chat/history");
      setMessages([
        {
          id: "intro",
          role: "assistant",
          text: "Chat history cleared. You can start a fresh conversation!",
          citations: [],
        },
      ]);
      stopVoicePlayback();
    } catch (err) {
      alert("Failed to clear chat history: " + (err.response?.data?.detail || err.message));
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // 2. Text Query Handler
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

      // If Auto Voice Over is enabled, automatically speak answer
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

  // 3. Voice Over / TTS Playback
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
      // First try backend TTS endpoint (/voice/tts)
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
      audio.onerror = () => {
        fallbackBrowserTTS(msgId, text);
      };
      await audio.play();
    } catch (err) {
      // Fallback to browser Web Speech Synthesis if backend TTS fails
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
    utterance.pitch = 1.0;
    utterance.onend = () => setPlayingMessageId(null);
    utterance.onerror = () => setPlayingMessageId(null);
    window.speechSynthesis.speak(utterance);
  };

  // 4. Voice Input (Speech-to-Text Recording)
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
          setRecordingStatus("Listening... speak your question");
        };

        recognition.onresult = (event) => {
          const transcript = Array.from(event.results)
            .map((result) => result[0].transcript)
            .join("");
          setInput(transcript);
        };

        recognition.onerror = (event) => {
          console.warn("Speech recognition error:", event.error);
          setRecordingStatus("Error listening. Try again.");
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
        console.warn("Native speech recognition failed, falling back to audio recording", e);
      }
    }

    // Fallback: Audio recording via MediaRecorder sending to backend /voice/conversation
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        stream.getTracks().forEach((track) => track.stop());

        // Send to backend /voice/conversation
        setLoading(true);
        setRecordingStatus("Transcribing & analyzing with AI...");

        try {
          const formData = new FormData();
          formData.append("file", audioBlob, "voice_input.wav");

          const res = await api.post("/voice/conversation", formData, {
            responseType: "blob",
          });

          // Extract headers
          const rawUserText = res.headers["x-user-text"];
          const rawLlmText = res.headers["x-llm-text"];
          const userText = rawUserText ? decodeURIComponent(rawUserText) : "Spoken question";
          const llmText = rawLlmText ? decodeURIComponent(rawLlmText) : "Voice response generated";

          const userMsg = { id: "u_" + Date.now(), role: "user", text: userText };
          const botMsg = { id: "b_" + Date.now(), role: "assistant", text: llmText };
          setMessages((prev) => [...prev, userMsg, botMsg]);

          // Play returned audio response
          const audioUrl = URL.createObjectURL(res.data);
          const audio = new Audio(audioUrl);
          currentAudioRef.current = audio;
          setPlayingMessageId(botMsg.id);
          audio.onended = () => setPlayingMessageId(null);
          await audio.play();
        } catch (err) {
          alert("Voice processing error: " + (err.response?.data?.detail || err.message));
        } finally {
          setLoading(false);
          setRecordingStatus("");
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingStatus("Recording audio... click mic again to send");
    } catch (err) {
      alert("Microphone access denied or unavailable.");
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {}
      recognitionRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch {}
      mediaRecorderRef.current = null;
    }
    setIsRecording(false);
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", padding: "20px", display: "flex", flexDirection: "column", height: "calc(100vh - 80px)" }}>
      
      {/* Top Scope & Controls Bar */}
      <div className="card-glass" style={{ padding: "12px 18px", marginBottom: "16px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
        
        {/* Document Scope Dropdown */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Sparkles size={18} color="var(--primary)" />
          <span style={{ fontSize: "0.88rem", fontWeight: 700 }}>Scope:</span>
          <select
            value={filterDoc}
            onChange={(e) => setFilterDoc(e.target.value)}
            style={{
              padding: "6px 12px",
              borderRadius: "8px",
              border: "1px solid var(--border)",
              fontSize: "0.85rem",
              background: "#ffffff",
              fontWeight: 500,
              maxWidth: "260px"
            }}
          >
            <option value="">All Documents (Global)</option>
            {documents.map((d) => (
              <option key={d.doc_id} value={d.source_filename}>
                📄 {d.source_filename}
              </option>
            ))}
          </select>
        </div>

        {/* Action Controls: Auto Voice-Over & Clear History */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          
          {/* Auto Voice Over Toggle */}
          <button
            onClick={() => setAutoVoiceOver(!autoVoiceOver)}
            title="Auto-read assistant answers aloud"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "6px 12px",
              borderRadius: "8px",
              fontSize: "0.82rem",
              fontWeight: 600,
              border: "1px solid " + (autoVoiceOver ? "#16a34a" : "var(--border)"),
              background: autoVoiceOver ? "#f0fdf4" : "#ffffff",
              color: autoVoiceOver ? "#16a34a" : "#64748b",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            {autoVoiceOver ? <Volume2 size={15} /> : <VolumeX size={15} />}
            Auto Voice-Over: {autoVoiceOver ? "ON" : "OFF"}
          </button>

          {/* Clear History Button */}
          <button
            onClick={handleClearHistory}
            title="Clear saved chat history for your account"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "6px 12px",
              borderRadius: "8px",
              fontSize: "0.82rem",
              fontWeight: 600,
              border: "1px solid var(--border)",
              background: "#ffffff",
              color: "#ef4444",
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
          >
            <Trash2 size={14} />
            Clear History
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="card-glass" style={{ flex: 1, padding: "20px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "16px" }}>
        
        {historyLoading && (
          <div style={{ textAlign: "center", padding: "20px", color: "#64748b", fontSize: "0.88rem" }}>
            <RotateCcw className="animate-spin" size={18} style={{ margin: "0 auto 8px" }} />
            Loading your persistent chat history...
          </div>
        )}

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

              {/* Message Bubble */}
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

                {/* Telemetry & Action Bar for Assistant */}
                {!isUser && m.id !== "intro" && (
                  <div
                    style={{
                      marginTop: "10px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      fontSize: "0.74rem",
                      color: "#94a3b8",
                      flexWrap: "wrap",
                      gap: "8px",
                    }}
                  >
                    <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                      {m.from_cache ? (
                        <span style={{ color: "#16a34a", fontWeight: 600, display: "flex", alignItems: "center", gap: "2px" }}>
                          <Zap size={12} /> Cache Hit (&lt;5ms)
                        </span>
                      ) : (
                        <span>Latency: {m.latency_ms ? `${m.latency_ms}ms` : "~1.1s"}</span>
                      )}
                      <span>Tokens: {m.tokens}</span>
                      <span>Cost: ${m.cost_usd ? m.cost_usd.toFixed(5) : "0.00"}</span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      {/* Voice Over (Read Aloud) Button */}
                      <button
                        onClick={() => playVoiceOver(m.id, m.text)}
                        title={isSpeaking ? "Stop Voice Over" : "Read aloud with Voice Over"}
                        style={{
                          background: isSpeaking ? "#eff6ff" : "none",
                          border: isSpeaking ? "1px solid #93c5fd" : "none",
                          borderRadius: "6px",
                          padding: "3px 7px",
                          cursor: "pointer",
                          color: isSpeaking ? "#2563eb" : "#64748b",
                          display: "flex",
                          alignItems: "center",
                          gap: "4px",
                          fontWeight: isSpeaking ? 700 : 500,
                        }}
                      >
                        {isSpeaking ? <VolumeX size={14} /> : <Volume2 size={14} />}
                        {isSpeaking ? "Speaking..." : "Voice Over"}
                      </button>

                      {/* Copy to Clipboard */}
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
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Loading Indicator */}
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

        {/* Active Speech Recording Banner */}
        {isRecording && (
          <div style={{
            background: "#fef2f2",
            border: "1px solid #fecaca",
            padding: "10px 16px",
            borderRadius: "10px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            color: "#b91c1c",
            fontSize: "0.88rem",
            fontWeight: 600
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span className="pulse-mic-dot" style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#ef4444" }}></span>
              {recordingStatus || "Listening to your voice..."}
            </div>
            <button
              onClick={stopRecording}
              style={{
                background: "#ef4444",
                color: "white",
                border: "none",
                borderRadius: "6px",
                padding: "4px 10px",
                cursor: "pointer",
                fontSize: "0.8rem",
                fontWeight: 600
              }}
            >
              Finish & Send
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input & Voice Controls */}
      <form onSubmit={handleSend} style={{ marginTop: "16px", display: "flex", gap: "10px", alignItems: "center" }}>
        
        {/* Microphone Voice Button */}
        <button
          type="button"
          onClick={toggleRecording}
          title={isRecording ? "Stop recording voice" : "Speak question with microphone"}
          style={{
            width: "44px",
            height: "44px",
            borderRadius: "10px",
            border: isRecording ? "2px solid #ef4444" : "1px solid var(--border)",
            background: isRecording ? "#fef2f2" : "#ffffff",
            color: isRecording ? "#ef4444" : "#475569",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            transition: "all 0.2s ease"
          }}
        >
          {isRecording ? <MicOff size={20} className="animate-pulse" /> : <Mic size={20} />}
        </button>

        {/* Text Input Field */}
        <input
          className="input-field"
          type="text"
          placeholder={filterDoc ? `Ask about "${filterDoc}" (or click mic)...` : "Ask any question about your documents (or click mic)..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          style={{ flex: 1 }}
        />

        {/* Send Button */}
        <button
          type="submit"
          className="btn-primary"
          disabled={loading || !input.trim()}
          style={{ display: "flex", alignItems: "center", gap: "8px", padding: "0 22px", height: "44px" }}
        >
          <Send size={16} />
          Send
        </button>
      </form>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse-dot {
          0% { transform: scale(0.95); opacity: 0.8; }
          50% { transform: scale(1.3); opacity: 1; }
          100% { transform: scale(0.95); opacity: 0.8; }
        }
        .pulse-mic-dot {
          animation: pulse-dot 1s infinite ease-in-out;
        }
      `}</style>
    </div>
  );
}
