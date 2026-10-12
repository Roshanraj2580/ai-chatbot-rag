import React, { useState, useEffect } from "react";
import api from "../api/client";
import { UploadCloud, FileText, Trash2, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";

export default function DocumentHub() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const res = await api.get("/documents");
      setDocuments(res.data || []);
    } catch (err) {
      console.error("Failed to fetch documents", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus({ type: "error", message: "Only PDF documents are supported." });
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setUploadStatus(null);

    try {
      const res = await api.post("/documents", formData);
      setUploadStatus({
        type: "success",
        message: `Successfully ingested "${file.name}" into ${res.data.chunks} chunks.`,
      });
      fetchDocuments();
    } catch (err) {
      const msg = err.response?.data?.detail || "Upload failed.";
      setUploadStatus({ type: "error", message: msg });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleDelete = async (docId, filename) => {
    if (!window.confirm(`Delete "${filename}" and all its vector chunks?`)) return;

    try {
      await api.delete(`/documents/${docId}`);
      setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
    } catch (err) {
      alert("Failed to delete document: " + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", padding: "20px" }}>
      
      {/* Upload Zone */}
      <div className="card-glass" style={{ padding: "32px", textAlign: "center", marginBottom: "24px" }}>
        <div style={{
          width: "60px",
          height: "60px",
          background: "rgba(37, 99, 235, 0.1)",
          color: "var(--primary)",
          borderRadius: "50%",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "12px",
        }}>
          <UploadCloud size={30} />
        </div>
        <h3 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "6px" }}>Upload PDF Documents</h3>
        <p style={{ color: "var(--text-muted)", fontSize: "0.88rem", maxWidth: "450px", margin: "0 auto 18px" }}>
          Files are automatically partitioned and embedded into your isolated tenant workspace using Gemini Embeddings.
        </p>

        <label className="btn-primary" style={{ display: "inline-flex", alignItems: "center", gap: "8px", cursor: uploading ? "not-allowed" : "pointer" }}>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            disabled={uploading}
            style={{ display: "none" }}
          />
          {uploading ? "Ingesting & Embedding..." : "Browse PDF File"}
        </label>

        {uploadStatus && (
          <div style={{
            marginTop: "16px",
            padding: "10px 14px",
            borderRadius: "8px",
            fontSize: "0.85rem",
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            background: uploadStatus.type === "success" ? "#dcfce7" : "#fee2e2",
            color: uploadStatus.type === "success" ? "#166534" : "#991b1b",
            border: `1px solid ${uploadStatus.type === "success" ? "#bbf7d0" : "#fca5a5"}`
          }}>
            {uploadStatus.type === "success" ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            {uploadStatus.message}
          </div>
        )}
      </div>

      {/* Document Library Table */}
      <div className="card-glass" style={{ padding: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <div>
            <h4 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Your Workspace Documents</h4>
            <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: "2px" }}>
              Private documents indexed for your account
            </p>
          </div>
          <button onClick={fetchDocuments} className="btn-outline" style={{ display: "flex", alignItems: "center", gap: "6px", padding: "6px 12px", fontSize: "0.82rem" }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)", fontSize: "0.9rem" }}>
            Loading your documents...
          </div>
        ) : documents.length === 0 ? (
          <div style={{ textAlign: "center", padding: "40px", color: "var(--text-muted)" }}>
            <FileText size={40} style={{ opacity: 0.3, marginBottom: "8px" }} />
            <p style={{ fontSize: "0.95rem", fontWeight: 600 }}>No documents uploaded yet</p>
            <p style={{ fontSize: "0.82rem" }}>Upload your first PDF above to start asking questions.</p>
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left", color: "#64748b", fontSize: "0.78rem", textTransform: "uppercase" }}>
                  <th style={{ padding: "10px 12px" }}>Filename</th>
                  <th style={{ padding: "10px 12px" }}>Pages</th>
                  <th style={{ padding: "10px 12px" }}>Chunks</th>
                  <th style={{ padding: "10px 12px" }}>Uploaded</th>
                  <th style={{ padding: "10px 12px", textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.doc_id} style={{ borderBottom: "1px solid #f1f5f9" }}>
                    <td style={{ padding: "12px", fontWeight: 600, color: "#1e293b", display: "flex", alignItems: "center", gap: "8px" }}>
                      <FileText size={16} color="#ef4444" />
                      {doc.source_filename}
                    </td>
                    <td style={{ padding: "12px", color: "#64748b" }}>{doc.pages || "N/A"}</td>
                    <td style={{ padding: "12px" }}>
                      <span style={{ background: "#e0f2fe", color: "#0369a1", padding: "2px 8px", borderRadius: "6px", fontWeight: 700, fontSize: "0.75rem" }}>
                        {doc.chunks} chunks
                      </span>
                    </td>
                    <td style={{ padding: "12px", color: "#64748b", fontSize: "0.8rem" }}>
                      {doc.ingested_at ? doc.ingested_at.substring(0, 10) : "Just now"}
                    </td>
                    <td style={{ padding: "12px", textAlign: "right" }}>
                      <button
                        onClick={() => handleDelete(doc.doc_id, doc.source_filename)}
                        style={{
                          background: "#fee2e2",
                          border: "none",
                          color: "#dc2626",
                          padding: "6px 10px",
                          borderRadius: "6px",
                          cursor: "pointer",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                          fontSize: "0.78rem",
                          fontWeight: 600,
                        }}
                      >
                        <Trash2 size={13} /> Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}
