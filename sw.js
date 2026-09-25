// public/sw.js — Sentinel IQ service worker (Day 15)
//
// MUST live in public/, not src/. Vite copies public/ to the site root, so
// this is served at /sw.js and can claim scope '/'. A service worker bundled
// out of src/ lands at /assets/sw-<hash>.js and can only control /assets/ —
// it would register without error and then never receive a push.
//
// Push handling ONLY. No offline caching: a stale cached shell on a
// surveillance dashboard is worse than a failed load, and cache strategy is
// out of scope for today.

self.addEventListener('install', () => {
  // Take over immediately instead of waiting for every tab to close.
  // Without this, a redeployed SW sits in "waiting" and the phone keeps
  // running the previous version until the PWA is fully killed — which,
  // during a demo, looks exactly like push being broken.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  if (!event.data) return;

  let data;
  try {
    data = event.data.json();
  } catch {
    // A malformed payload must not kill the SW — without this the whole
    // worker throws and no LATER push is handled either.
    return;
  }

  event.waitUntil(
    self.clients
      .matchAll({ type: 'window', includeUncontrolled: true })
      .then((clientList) => {
        // ── Foreground: the app is open and focused ──────────────────────
        // Hand the payload to the page and show NO OS notification. Doing
        // both would stack a tray notification on top of the in-app card
        // the user is already looking at.
        const focused = clientList.find((c) => c.focused);
        if (focused) {
          focused.postMessage({ type: 'CRITICAL_ALERT', payload: data });
          return undefined;
        }

        // ── Background / closed: OS notification ─────────────────────────
        const assigned = data.assigned !== false;
        const title = assigned ? '⚠ Critical Alert — You are assigned'
                               : '⚠ Critical Alert — Unassigned';
        const score = (data.score === null || data.score === undefined)
          ? '' : ` · Score ${data.score}`;

        return self.registration.showNotification(title, {
          body: `${data.camera_name || 'Unknown camera'}`
                + `${data.camera_location ? ' — ' + data.camera_location : ''}`
                + score,
          icon: '/icons/icon-192.png',
          badge: '/icons/icon-192.png',
          vibrate: [200, 100, 200, 100, 200],
          // Same alert delivered twice (retry, or two subscriptions on one
          // device) collapses into one tray entry instead of stacking.
          tag: `alert-${data.alert_id}`,
          renotify: true,
          // Ignored on iOS — documented, not worked around. The notification
          // still appears and the tap still routes correctly.
          requireInteraction: true,
          data: { alert_id: data.alert_id },
        });
      })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const alertId = event.notification.data && event.notification.data.alert_id;
  if (!alertId) return;

  // Deep link as a query param, not a path: this dashboard has no client-side
  // router (App.jsx switches on a `page` state value), so /alert/42 would
  // 404 on reload. App.jsx reads ?alert= on mount and opens the card.
  const url = `/?alert=${alertId}`;

  event.waitUntil(
    self.clients
      .matchAll({ type: 'window', includeUncontrolled: true })
      .then((clientList) => {
        for (const client of clientList) {
          if ('navigate' in client) {
            return client.navigate(url).then((c) => (c ? c.focus() : undefined));
          }
        }
        return self.clients.openWindow(url);
      })
  );
});
