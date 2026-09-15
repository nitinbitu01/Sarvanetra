// frontend/src/utils/authClient.js
// Access token stored in MEMORY only — never in localStorage or sessionStorage.
// On page refresh or expiry, /auth/refresh is called automatically using the HTTP-only cookie.

let _accessToken = null;
let _refreshTimer = null;

const ACCESS_TOKEN_EXPIRE_MS = 15 * 60 * 1000;  // 15 minutes
const REFRESH_BEFORE_MS      =  2 * 60 * 1000;  // refresh 2 min before expiry

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

export function setAccessToken(token) {
  _accessToken = token;
  scheduleRefresh();
}

export function getAccessToken() {
  if (_accessToken) return _accessToken;
  try {
    const token = localStorage.getItem('sg_token');
    if (token) return token;
  } catch (e) {}
  return null;
}

export function clearAccessToken() {
  _accessToken = null;
  if (_refreshTimer) {
    clearTimeout(_refreshTimer);
    _refreshTimer = null;
  }
}

function scheduleRefresh() {
  if (_refreshTimer) clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(
    refreshAccessToken,
    ACCESS_TOKEN_EXPIRE_MS - REFRESH_BEFORE_MS,
  );
}

export async function refreshAccessToken() {
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
    });

    if (res.status === 401) {
      clearAccessToken();
      return null;
    }

    if (!res.ok) {
      console.error('[AuthClient] Refresh failed:', res.status);
      return null;
    }

    const data = await res.json();
    if (data.access_token) {
      setAccessToken(data.access_token);
      return data.access_token;
    }
  } catch (e) {
    console.error('[AuthClient] Refresh network error:', e);
  }
  return null;
}

export async function apiFetch(url, options = {}) {
  const token = getAccessToken();
  const normalizedUrl = url.startsWith('/api/v1') ? url.slice(7) : url;
  const fullUrl = url.startsWith('http') ? url : `${API_BASE}${normalizedUrl.startsWith('/') ? '' : '/'}${normalizedUrl}`;

  const res = await fetch(fullUrl, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });

  if (res.status === 401) {
    const newToken = await refreshAccessToken();
    if (!newToken) {
      return res;
    }

    return fetch(fullUrl, {
      ...options,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${newToken}`,
        ...(options.headers || {}),
      },
    });
  }

  return res;
}
