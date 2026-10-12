import React, { useState, useEffect } from "react";
import api from "../api/client";
import { 
  Activity, Database, Zap, Cpu, RefreshCw, 
  CheckCircle2, AlertTriangle, Layers, Clock, ShieldCheck
} from "lucide-react";

export default function MLOpsDashboard() {
  const [stats, setStats] = useState(null);
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const fetchDashboardData = async () => {
    try {
      setError(null);
      const [statsRes, configRes, healthRes] = await Promise.all([
        api.get("/cache/stats").catch(() => ({ data: { hits: 0, misses: 0, hit_rate: 0, size: 0, backend: "redis" } })),
        api.get("/config").catch(() => ({ data: {} })),
        api.get("/health").catch(() => ({ data: { status: "degraded", chroma_count: 0 } })),
      ]);

      setStats(statsRes.data);
      setConfig(configRes.data);
      setHealth(healthRes.data);
    } catch (err) {
      setError("Failed to fetch MLOps telemetry.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const handleManualRefresh = () => {
    setRefreshing(true);
    fetchDashboardData();
  };

  const totalQueries = (stats?.hits || 0) + (stats?.misses || 0);
  const hitRatePct = totalQueries > 0 
    ? ((stats.hits / totalQueries) * 100).toFixed(1)
    : (stats?.hit_rate ? (stats.hit_rate * 100).toFixed(1) : "0.0");

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "80px 20px" }}>
        <RefreshCw className="animate-spin" size={36} color="#3b82f6" style={{ margin: "0 auto 16px" }} />
        <h3 style={{ fontSize: "1.1rem", color: "#64748b" }}>Loading telemetry metrics...</h3>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "28px 20px" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "28px" }}>
        <div>
          <h2 style={{ fontSize: "1.6rem", fontWeight: 800, color: "#0f172a", margin: 0, letterSpacing: "-0.02em" }}>
            MLOps & System Observability
          </h2>
          <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: "0.9rem" }}>
            Real-time inference telemetry, Upstash Redis cache performance, and vector database status.
          </p>
        </div>
        <button
          onClick={handleManualRefresh}
          disabled={refreshing}
          className="btn-secondary"
          style={{ display: "flex", alignItems: "center", gap: "8px" }}
        >
          <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />
          {refreshing ? "Refreshing..." : "Refresh Metrics"}
        </button>
      </div>

      {error && (
        <div style={{
          background: "#fee2e2",
          border: "1px solid #fca5a5",
          color: "#991b1b",
          padding: "12px 16px",
          borderRadius: "10px",
          marginBottom: "20px",
          display: "flex",
          alignItems: "center",
          gap: "8px"
        }}>
          <AlertTriangle size={18} />
          {error}
        </div>
      )}

      {/* KPI Cards Row */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "18px",
        marginBottom: "28px"
      }}>
        {/* Cache Hit Rate */}
        <div className="card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>Cache Hit Rate</span>
            <div style={{ background: "#eff6ff", color: "#2563eb", padding: "6px", borderRadius: "8px" }}>
              <Zap size={18} />
            </div>
          </div>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "#1d4ed8" }}>
            {hitRatePct}%
          </div>
          <span style={{ fontSize: "0.78rem", color: "#64748b", marginTop: "4px", display: "block" }}>
            {stats?.hits || 0} hits out of {totalQueries} requests
          </span>
        </div>

        {/* Total Cache Queries */}
        <div className="card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>Cache Misses</span>
            <div style={{ background: "#fef3c7", color: "#d97706", padding: "6px", borderRadius: "8px" }}>
              <Clock size={18} />
            </div>
          </div>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "#b45309" }}>
            {stats?.misses || 0}
          </div>
          <span style={{ fontSize: "0.78rem", color: "#64748b", marginTop: "4px", display: "block" }}>
            Fresh LLM generation passes
          </span>
        </div>

        {/* Vector DB Chunks */}
        <div className="card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>Vector Store Chunks</span>
            <div style={{ background: "#f0fdf4", color: "#16a34a", padding: "6px", borderRadius: "8px" }}>
              <Database size={18} />
            </div>
          </div>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "#15803d" }}>
            {health?.chroma_count ?? 0}
          </div>
          <span style={{ fontSize: "0.78rem", color: "#64748b", marginTop: "4px", display: "block" }}>
            Active embeddings in ChromaDB
          </span>
        </div>

        {/* System Health */}
        <div className="card" style={{ padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#64748b", textTransform: "uppercase" }}>System Status</span>
            <div style={{ 
              background: health?.status === "ok" ? "#f0fdf4" : "#fef2f2", 
              color: health?.status === "ok" ? "#16a34a" : "#dc2626", 
              padding: "6px", 
              borderRadius: "8px" 
            }}>
              <ShieldCheck size={18} />
            </div>
          </div>
          <div style={{ fontSize: "1.4rem", fontWeight: 800, color: health?.status === "ok" ? "#15803d" : "#b91c1c", textTransform: "capitalize" }}>
            {health?.status || "Unknown"}
          </div>
          <span style={{ fontSize: "0.78rem", color: "#64748b", marginTop: "4px", display: "block" }}>
            FastAPI 0.115 + Chroma Core
          </span>
        </div>
      </div>

      {/* Grid: Cache Engine & Model Config */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
        {/* Cache Engine Details */}
        <div className="card" style={{ padding: "24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
            <div style={{ background: "#f3e8ff", color: "#9333ea", padding: "8px", borderRadius: "8px" }}>
              <Zap size={20} />
            </div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0 }}>Caching Tier (Upstash Redis)</h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "0.9rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ color: "#64748b" }}>Active Cache Backend</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>{stats?.backend?.toUpperCase() || "REDIS / IN-MEMORY"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ color: "#64748b" }}>Cached Query Entries</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>{stats?.size ?? stats?.cached_keys ?? "Dynamic TTL"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ color: "#64748b" }}>Tenant Isolation</span>
              <span style={{ fontWeight: 600, color: "#16a34a", display: "flex", alignItems: "center", gap: "4px" }}>
                <CheckCircle2 size={16} /> Enforced (Scoped by User ID)
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0" }}>
              <span style={{ color: "#64748b" }}>Cache Invalidation Strategy</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>Auto-clears on document delete</span>
            </div>
          </div>
        </div>

        {/* RAG Engine Parameters */}
        <div className="card" style={{ padding: "24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
            <div style={{ background: "#e0f2fe", color: "#0284c7", padding: "8px", borderRadius: "8px" }}>
              <Cpu size={20} />
            </div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0 }}>RAG Pipeline Architecture</h3>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "0.9rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ color: "#64748b" }}>Embedding Model</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>{config?.embedding_model || "all-MiniLM-L6-v2"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ color: "#64748b" }}>LLM Inference Engine</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>{config?.llm_model || "Groq (llama-3.1-8b-instant)"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid #f1f5f9" }}>
              <span style={{ color: "#64748b" }}>Chunk Size / Overlap</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>
                {config?.chunk_size || 500} chars / {config?.chunk_overlap || 50} overlap
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0" }}>
              <span style={{ color: "#64748b" }}>Top-K Document Retrieval</span>
              <span style={{ fontWeight: 600, color: "#0f172a" }}>{config?.top_k || 5} chunks</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
