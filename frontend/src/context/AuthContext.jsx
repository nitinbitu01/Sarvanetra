// frontend/src/context/AuthContext.jsx
import { createContext, useContext, useState, useCallback } from 'react';
import { getMockResponse } from '../data/mockData';

const AuthContext = createContext(null);

const API = import.meta.env.VITE_API_URL || '/api/v1';

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem('sg_token') || null);
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('sg_user') || 'null'); }
    catch { return null; }
  });

  const login = useCallback(async (username, password) => {
    const res = await fetch(`${API}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    const data = await res.json();
    localStorage.setItem('sg_token', data.access_token);
    localStorage.setItem('sg_user', JSON.stringify({ username: data.username, role: data.role }));
    setToken(data.access_token);
    setUser({ username: data.username, role: data.role });
    return data;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('sg_token');
    localStorage.removeItem('sg_user');
    setToken(null);
    setUser(null);
  }, []);

  const authFetch = useCallback(async (url, opts = {}) => {
    try {
      const res = await fetch(url, {
        ...opts,
        headers: { ...(opts.headers || {}), Authorization: `Bearer ${token}` },
      });
      if (res.status === 401) {
        logout();
        return res;
      }
      if (res.ok) {
        const ct = res.headers.get('content-type') || '';
        if (ct.includes('application/json') || ct.includes('octet-stream') || ct.includes('pdf')) {
          return res;
        }
      }
      // If 404 or HTML (e.g. GitHub Pages static fallback)
      const mock = getMockResponse(url, opts);
      if (mock) return mock;
      return res;
    } catch {
      const mock = getMockResponse(url, opts);
      if (mock) return mock;
      throw new Error('Network error');
    }
  }, [token, logout]);

  return (
    <AuthContext.Provider value={{ token, user, login, logout, authFetch, isAdmin: user?.role === 'admin' }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
export { API };
