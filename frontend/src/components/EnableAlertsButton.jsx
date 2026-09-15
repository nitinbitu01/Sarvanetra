// frontend/src/components/EnableAlertsButton.jsx
//
// Opt-in control for push. Every failure mode gets its own message, because
// "it didn't work" is useless on a phone with no devtools — the officer
// needs to know whether to install the app, change a browser setting, or
// tell someone the server is misconfigured.
import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import {
  isIosSafariNotInstalled, isSecureContextOk, pushSupported,
  subscribeToPush, unsubscribeFromPush,
} from '../utils/push';

const MESSAGES = {
  not_supported: {
    text: 'Push notifications are not supported in this browser.',
    color: 'var(--text-muted)',
  },
  insecure_context: {
    // The LAN-IP trap. Naming the fix is the entire value of this branch.
    text: 'Push requires HTTPS. Open the app over an https:// address '
      + '(e.g. an ngrok tunnel) — a 192.168.x.x address will not work.',
    color: 'var(--accent-yellow)',
  },
  ios_not_installed: {
    text: 'On iPhone: tap Share → Add to Home Screen, then open Sarvanetra AI '
      + 'from the home-screen icon and enable alerts there. Push does not '
      + 'work from a Safari tab.',
    color: '#3b82f6',
  },
  permission_denied: {
    text: 'Notifications are blocked. Re-enable them for this site in your '
      + 'browser settings, then try again.',
    color: 'var(--accent-yellow)',
  },
  no_officer_record: {
    text: 'Your login is not linked to an officer record, so alerts have '
      + 'nowhere to route. Ask an admin to create one.',
    color: 'var(--accent-yellow)',
  },
  server_error: {
    text: 'Subscription failed on the server. Check the backend logs.',
    color: 'var(--accent-red)',
  },
  error: {
    text: 'Could not enable alerts. See the browser console for details.',
    color: 'var(--accent-red)',
  },
};

export default function EnableAlertsButton() {
  const { authFetch } = useAuth();
  const [state, setState] = useState('checking');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!pushSupported()) { setState('not_supported'); return; }
    if (!isSecureContextOk()) { setState('insecure_context'); return; }
    if (isIosSafariNotInstalled()) { setState('ios_not_installed'); return; }
    // Reflect the real current state rather than assuming 'idle': a returning
    // officer who already subscribed should not be told to enable again.
    if (Notification.permission === 'granted') {
      navigator.serviceWorker?.ready
        .then((reg) => reg.pushManager.getSubscription())
        .then((sub) => setState(sub ? 'granted' : 'idle'))
        .catch(() => setState('idle'));
    } else if (Notification.permission === 'denied') {
      setState('permission_denied');
    } else {
      setState('idle');
    }
  }, []);

  // Must run inside the click handler's call stack: requestPermission() away
  // from a user gesture is silently ignored and never resolves.
  const enable = useCallback(async () => {
    setBusy(true);
    const result = await subscribeToPush(authFetch);
    setBusy(false);
    setState(result.success ? 'granted' : result.reason);
  }, [authFetch]);

  const disable = useCallback(async () => {
    setBusy(true);
    await unsubscribeFromPush(authFetch);
    setBusy(false);
    setState('idle');
  }, [authFetch]);

  if (state === 'checking') return null;

  if (state === 'granted') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ color: 'var(--accent-green)', fontSize: 12, fontWeight: 600 }}>
          ✓ Push alerts active on this device
        </span>
        <button
          onClick={disable}
          disabled={busy}
          style={{
            padding: '3px 9px', borderRadius: 6, fontSize: 11,
            border: '1px solid var(--border)', background: 'var(--bg-elevated)',
            color: 'var(--text-muted)', cursor: busy ? 'not-allowed' : 'pointer',
          }}
        >
          {busy ? '…' : 'Disable'}
        </button>
      </div>
    );
  }

  if (state === 'idle') {
    return (
      <button
        onClick={enable}
        disabled={busy}
        style={{
          padding: '7px 14px', borderRadius: 8, fontSize: 13, fontWeight: 700,
          border: 'none', background: busy ? 'var(--bg-elevated)' : 'var(--accent-green)',
          color: busy ? 'var(--text-muted)' : '#fff',
          cursor: busy ? 'not-allowed' : 'pointer',
        }}
      >
        {busy ? 'Registering…' : '🔔 Enable Alerts'}
      </button>
    );
  }

  const msg = MESSAGES[state] || MESSAGES.error;
  return (
    <div style={{ fontSize: 12, color: msg.color, lineHeight: 1.5, maxWidth: 460 }}>
      {msg.text}
    </div>
  );
}
