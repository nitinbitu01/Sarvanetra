// frontend/src/utils/auth.js
// Single source of truth for officer token.
// Import getToken() in: FeedbackButtons, HlsPlayer, AlertDetailPage.
// NEVER call sessionStorage directly in component files.

export function getToken() {
  return sessionStorage.getItem('sentinel_officer_token') || localStorage.getItem('sg_token');
}

export function setToken(token) {
  sessionStorage.setItem('sentinel_officer_token', token);
  localStorage.setItem('sg_token', token);
}

export function clearToken() {
  sessionStorage.removeItem('sentinel_officer_token');
  localStorage.removeItem('sg_token');
}
