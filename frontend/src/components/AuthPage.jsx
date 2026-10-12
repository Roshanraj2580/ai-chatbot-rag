import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { Bot, Lock, User, ArrowRight } from "lucide-react";

export default function AuthPage() {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isRegister) {
        await register(username, password);
      } else {
        await login(username, password);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || "Authentication failed. Try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "20px" }}>
      <div className="card-glass" style={{ width: "100%", maxWidth: "420px", padding: "36px 32px" }}>
        
        {/* Header Logo */}
        <div style={{ textAlign: "center", marginBottom: "28px" }}>
          <div style={{
            width: "56px",
            height: "56px",
            background: "rgba(37, 99, 235, 0.1)",
            color: "var(--primary)",
            borderRadius: "16px",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "12px"
          }}>
            <Bot size={32} />
          </div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
            AI RAG Platform
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "4px" }}>
            {isRegister ? "Create your tenant workspace" : "Sign in with your credentials"}
          </p>
        </div>

        {error && (
          <div style={{
            background: "#fee2e2",
            border: "1px solid #fca5a5",
            color: "#991b1b",
            padding: "10px 14px",
            borderRadius: "8px",
            fontSize: "0.88rem",
            marginBottom: "20px"
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, color: "#475569", marginBottom: "6px" }}>
              Username
            </label>
            <div style={{ position: "relative" }}>
              <User size={18} style={{ position: "absolute", left: "12px", top: "13px", color: "#94a3b8" }} />
              <input
                className="input-field"
                style={{ paddingLeft: "38px" }}
                type="text"
                required
                placeholder="Enter username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, color: "#475569", marginBottom: "6px" }}>
              Password
            </label>
            <div style={{ position: "relative" }}>
              <Lock size={18} style={{ position: "absolute", left: "12px", top: "13px", color: "#94a3b8" }} />
              <input
                className="input-field"
                style={{ paddingLeft: "38px" }}
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn-primary"
            disabled={loading}
            style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", marginTop: "8px" }}
          >
            {loading ? "Processing..." : isRegister ? "Create Account" : "Sign In"}
            <ArrowRight size={16} />
          </button>
        </form>

        <div style={{ textAlign: "center", marginTop: "24px", paddingTop: "20px", borderTop: "1px solid var(--border)" }}>
          <p style={{ fontSize: "0.88rem", color: "var(--text-muted)" }}>
            {isRegister ? "Already have an account?" : "Need a workspace account?"}{" "}
            <button
              onClick={() => { setIsRegister(!isRegister); setError(""); }}
              style={{ background: "none", border: "none", color: "var(--primary)", fontWeight: 700, cursor: "pointer" }}
            >
              {isRegister ? "Sign In" : "Register"}
            </button>
          </p>
          <div style={{ marginTop: "12px", fontSize: "0.78rem", color: "#94a3b8" }}>
            Demo: <code>admin</code> / <code>admin123</code>
          </div>
        </div>

      </div>
    </div>
  );
}
