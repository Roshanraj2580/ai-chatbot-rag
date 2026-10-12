import React, { createContext, useContext, useState, useEffect } from "react";
import api from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem("rag_user");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [token, setToken] = useState(() => localStorage.getItem("rag_token"));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // If token exists, verify with backend /auth/me
    if (token) {
      api
        .get("/auth/me")
        .then((res) => {
          setUser(res.data);
          localStorage.setItem("rag_user", JSON.stringify(res.data));
        })
        .catch(() => {
          logout();
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }

    const handleAuthExpired = () => logout();
    window.addEventListener("auth-expired", handleAuthExpired);
    return () => window.removeEventListener("auth-expired", handleAuthExpired);
  }, [token]);

  const login = async (username, password) => {
    const res = await api.post("/auth/login", { username, password });
    const { access_token, user_id, role } = res.data;
    const userData = { id: user_id, username, role };

    localStorage.setItem("rag_token", access_token);
    localStorage.setItem("rag_user", JSON.stringify(userData));
    setToken(access_token);
    setUser(userData);
    return userData;
  };

  const register = async (username, password) => {
    const res = await api.post("/auth/register", { username, password });
    const { access_token, user_id, role } = res.data;
    const userData = { id: user_id, username, role };

    localStorage.setItem("rag_token", access_token);
    localStorage.setItem("rag_user", JSON.stringify(userData));
    setToken(access_token);
    setUser(userData);
    return userData;
  };

  const logout = () => {
    localStorage.removeItem("rag_token");
    localStorage.removeItem("rag_user");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
