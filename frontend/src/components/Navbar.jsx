import React from "react";
import { useAuth } from "../context/AuthContext";
import { Bot, MessageSquare, Files, Activity, LogOut, User } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab }) {
  const { user, logout } = useAuth();

  const navItems = [
    { id: "chat", label: "Chat Assistant", icon: MessageSquare },
    { id: "documents", label: "Document Hub", icon: Files },
    { id: "mlops", label: "MLOps Telemetry", icon: Activity },
  ];

  return (
    <header style={{
      background: "#0f172a",
      color: "white",
      padding: "12px 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      position: "sticky",
      top: 0,
      zIndex: 100,
      boxShadow: "0 4px 20px -2px rgba(0, 0, 0, 0.15)"
    }}>
      {/* Brand */}
      <div style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer" }} onClick={() => setActiveTab("chat")}>
        <div style={{
          background: "#2563eb",
          padding: "8px",
          borderRadius: "10px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center"
        }}>
          <Bot size={22} color="white" />
        </div>
        <div>
          <span style={{ fontWeight: 800, fontSize: "1.1rem", letterSpacing: "-0.02em" }}>AI RAG Platform</span>
          <span style={{ display: "block", fontSize: "0.72rem", color: "#94a3b8", fontWeight: 500 }}>React 19 • JWT Tenant</span>
        </div>
      </div>

      {/* Navigation Buttons */}
      <nav style={{ display: "flex", gap: "6px" }}>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 16px",
                borderRadius: "8px",
                border: "none",
                fontSize: "0.88rem",
                fontWeight: 600,
                cursor: "pointer",
                background: isActive ? "rgba(37, 99, 235, 0.25)" : "transparent",
                color: isActive ? "#60a5fa" : "#cbd5e1",
                transition: "all 0.15s ease",
              }}
            >
              <Icon size={17} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* User profile & Logout */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          background: "#1e293b",
          padding: "6px 12px",
          borderRadius: "9999px",
          fontSize: "0.82rem",
          color: "#cbd5e1"
        }}>
          <User size={14} color="#38bdf8" />
          <span style={{ fontWeight: 600 }}>{user?.username}</span>
          <span style={{
            background: user?.role === "admin" ? "#0284c7" : "#475569",
            color: "white",
            fontSize: "0.68rem",
            padding: "2px 6px",
            borderRadius: "4px",
            fontWeight: 700
          }}>
            {user?.role}
          </span>
        </div>

        <button
          onClick={logout}
          title="Sign Out"
          style={{
            background: "rgba(239, 68, 68, 0.15)",
            border: "none",
            color: "#f87171",
            padding: "8px",
            borderRadius: "8px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}
        >
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
}
