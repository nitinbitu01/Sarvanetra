// frontend/src/utils/push.js — Web Push subscription helpers (Day 15).

import { API } from '../context/AuthContext';

/**
 * URL-safe base64 → Uint8Array, for applicationServerKey.
 *
 * pushManager.subscribe() will NOT accept the base64 string directly:
 * Chrome throws a DOMException and Firefox fails in a way that looks like a
 * permission problem. This conversion is mandatory, which is why it lives
 * here rather than being reimplemented at each call site.
 */
export function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(base64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i);
  return out;
}

/** Running as an installed PWA (iOS home-screen, or Android/desktop standalone)? */
export function isInstalledPWA() {
  if (window.navigator.standalone === true) return true;               // iOS
  if (window.matchMedia?.('(display-mode: standalone)').matches) return true;
  return false;
}

export function isIOS() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent)
    // iPadOS 13+ reports as Mac; the touch-point check disambiguates.
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

/**
 * iOS Safari, in a browser tab rather than an installed PWA.
 *
 * Push on iOS (16.4+) only works from the home-screen app. A subscription
 * created in a tab appears to succeed and then silently never delivers —
 * so this has to be caught BEFORE subscribing, not diagnosed afterwards.
 */
export function isIosSafariNotInstalled() {
  return isIOS() && !isInstalledPWA();
}

/** True when the page is in a secure context (required for SW + Push). */
export function isSecureContextOk() {
  return window.location.protocol === 'https:'
    || window.location.hostname === 'localhost'
    || window.location.hostname === '127.0.0.1';
}

export function pushSupported() {
  return 'serviceWorker' in navigator && 'PushManager' in window
    && 'Notification' in window;
}

/**
 * Full subscribe flow. MUST be called from a user-gesture handler.
 *
 * Notification.requestPermission() outside a gesture is silently ignored by
 * Chrome/Firefox and rejected on iOS — it neither resolves nor throws, so a
 * page-load call just hangs forever with no diagnosable symptom.
 *
 * @returns {{success: true, subscriptionId: number} | {success: false, reason: string}}
 */
export async function subscribeToPush(authFetch) {
  if (!pushSupported()) return { success: false, reason: 'not_supported' };
  if (!isSecureContextOk()) return { success: false, reason: 'insecure_context' };
  if (isIosSafariNotInstalled()) return { success: false, reason: 'ios_not_installed' };

  try {
    // Fetched, never hardcoded: this is what allows a VAPID key rotation
    // without shipping a new frontend bundle.
    const keyRes = await fetch(`${API}/push/vapid-public-key`);
    if (!keyRes.ok) return { success: false, reason: 'server_error' };
    const { public_key: publicKey } = await keyRes.json();
    if (!publicKey) return { success: false, reason: 'server_error' };

    const permission = await Notification.requestPermission();
    if (permission !== 'granted') {
      return { success: false, reason: 'permission_denied' };
    }

    const reg = await navigator.serviceWorker.ready;

    // Reuse an existing subscription if the browser already has one for this
    // origin — calling subscribe() again with different options throws
    // InvalidStateError rather than replacing it.
    let subscription = await reg.pushManager.getSubscription();
    if (!subscription) {
      subscription = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(publicKey),
      });
    }

    const sub = subscription.toJSON();
    const res = await authFetch(`${API}/push/subscribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        endpoint: sub.endpoint,
        keys: sub.keys,            // nested {p256dh, auth} — send as-is
        device_label: `${navigator.platform || 'device'} · ${new Date().toLocaleDateString()}`,
      }),
    });

    if (res.status === 409) return { success: false, reason: 'no_officer_record' };
    if (!res.ok) return { success: false, reason: 'server_error' };

    const data = await res.json();
    return { success: true, subscriptionId: data.subscription_id };
  } catch (err) {
    console.error('[SentinelIQ] subscribeToPush failed:', err);
    return { success: false, reason: 'error' };
  }
}

/** Explicit opt-out — unsubscribes locally AND removes the server row. */
export async function unsubscribeFromPush(authFetch) {
  try {
    const reg = await navigator.serviceWorker.ready;
    const subscription = await reg.pushManager.getSubscription();
    if (!subscription) return { success: true };

    const endpoint = subscription.endpoint;
    await subscription.unsubscribe();
    await authFetch(`${API}/push/subscribe`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ endpoint }),
    });
    return { success: true };
  } catch (err) {
    console.error('[SentinelIQ] unsubscribeFromPush failed:', err);
    return { success: false };
  }
}

/** Register the service worker, failing loudly rather than silently. */
export function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    console.warn('[SentinelIQ] Service Worker unsupported — push disabled.');
    return;
  }
  if (!isSecureContextOk()) {
    // The single most common way this feature "mysteriously doesn't work":
    // the phone is pointed at a LAN IP (192.168.x.x), which is NOT a secure
    // context, so registration fails with no visible error anywhere.
    console.warn(
      '[SentinelIQ] Not a secure context (%s). Service Workers and Web Push '
      + 'require HTTPS or localhost — a LAN IP will NOT work. Use `ngrok http 5173` '
      + 'or mkcert, and open the HTTPS URL on the phone.',
      window.location.origin,
    );
    return;
  }
  const swUrl = `${import.meta.env.BASE_URL || '/'}sw.js`.replace(/\/{2,}/g, '/');
  const swScope = import.meta.env.BASE_URL || '/';
  navigator.serviceWorker
    .register(swUrl, { scope: swScope })
    .then((reg) => console.log('[SentinelIQ] SW registered, scope:', reg.scope))
    .catch((err) => console.warn('[SentinelIQ] SW registration notice:', err.message));
}
