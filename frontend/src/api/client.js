import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Automatically inject JWT Bearer token into all requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("rag_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Automatically handle 401s (token expiry)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("rag_token");
      localStorage.removeItem("rag_user");
      // Don't auto-redirect if already on login
      if (!window.location.pathname.includes("/login")) {
        window.dispatchEvent(new Event("auth-expired"));
      }
    }
    return Promise.reject(error);
  }
);

export default api;
