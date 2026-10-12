import React, { useState } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import AuthPage from "./components/AuthPage";
import Navbar from "./components/Navbar";
import ChatView from "./components/ChatView";
import DocumentHub from "./components/DocumentHub";
import MLOpsDashboard from "./components/MLOpsDashboard";
import { RefreshCw } from "lucide-react";

function MainLayout() {
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState("chat");

  if (loading) {
    return (
      <div style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "#f8fafc"
      }}>
        <RefreshCw className="animate-spin" size={40} color="#2563eb" style={{ marginBottom: "16px" }} />
        <p style={{ color: "#64748b", fontWeight: 600 }}>Initializing AI Platform...</p>
      </div>
    );
  }

  if (!user) {
    return <AuthPage />;
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main style={{ flex: 1 }}>
        {activeTab === "chat" && <ChatView />}
        {activeTab === "documents" && <DocumentHub />}
        {activeTab === "mlops" && <MLOpsDashboard />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainLayout />
    </AuthProvider>
  );
}
