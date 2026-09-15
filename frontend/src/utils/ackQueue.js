// frontend/src/utils/ackQueue.js — Day 17: offline ACK outbox.
//
// THE PROBLEM THIS CLOSES
//   AlertDetailCard and RoutingBadge both call POST /alerts/{id}/ack directly
//   from a click handler. Before this module, a dropped connection meant the
//   fetch() throws, the catch block silently reset the button back to
//   "ACKNOWLEDGE", and the officer's tap simply vanished — no error, no
//   retry, no way to tell it didn't register. That is the single most
//   consequential gap in this app: the one action every other feature (GPS
//   routing, escalation timers, push notifications) exists to lead up to.
//
// WHY THIS IS SAFE TO BUILD CLIENT-ONLY
//   backend/routers/v1/routing.py's ack_alert() is already idempotent by
//   construction — a conditional UPDATE (`AND status IN ('ROUTED',
//   'ESCALATED_ROUTED')`) guarded by rowcount, the same atomic-claim pattern
//   used for officer assignment. Replaying POST /alerts/{id}/ack any number
//   of times for the same alert is a no-op after the first success. That
//   means this queue needs no server-side changes and no new idempotency key
//   of its own — alert_id already is one.
//
// PERSISTENCE
//   Best-effort IndexedDB, feature-detected. The in-memory `items` array is
//   the actual source of truth for every synchronous read and every
//   subscriber notification — IndexedDB is a side-channel written after each
//   state change so a killed-and-reopened tab can recover a pending ack.
//   Without IndexedDB (unsupported browser, or this module's own test
//   environment) the queue still works correctly for as long as the tab
//   stays open; it just does not survive a kill. That is a real, documented
//   limitation — see the Background Sync note below — not a silent gap.
//
// WHAT THIS DELIBERATELY DOES NOT DO
//   No Background Sync API registration. It would help exactly one case —
//   the tab is fully closed while offline — but iOS Safari has zero support
//   for it, and this app's own push-notification work (Day 15) already
//   treats iOS as a first-class target. Building Chrome/Android-only partial
//   coverage for a system that has to work on iOS is a worse trade than
//   shipping a smaller mechanism that is verified to work everywhere: flush
//   on reconnect, on 'online', on visibility change, and on a periodic
//   backstop timer, all of which require the tab to still exist. A fully
//   killed tab losing a queued ack is the one remaining gap; it is narrower
//   than what existed before (nothing was queued at all) and is recorded in
//   BACKLOG.md rather than silently left unstated.

const DB_NAME = 'sentinel_ack_queue';
const STORE = 'items';
const TOKEN_KEY = 'sg_token'; // matches AuthContext.jsx — this module cannot
                               // use the useAuth() hook, it isn't a component.
const API = import.meta.env.VITE_API_URL || '/api/v1';

const BACKOFF_BASE_MS = 2000;
const BACKOFF_CAP_MS = 30000;
const RETRY_TICK_MS = 5000;

/** @typedef {{id:string, alertId:string, createdAt:number, status:'pending'|'syncing'|'synced'|'failed', attempts:number, lastAttemptAt:number|null, lastError:string|null}} AckQueueItem */

/** @type {AckQueueItem[]} */
let items = [];
const listeners = new Set();
let flushing = false;
let retryTimer = null;
let dbPromise = null;
let hydrated = false;

function hasIndexedDB() {
  try {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  } catch {
    return false;
  }
}

function openDB() {
  if (!hasIndexedDB()) return Promise.resolve(null);
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore(STORE, { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(req.result);
      // A DB that fails to open (private-browsing quota denial, corruption)
      // degrades to in-memory-only rather than breaking the queue.
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return dbPromise;
}

async function persistItem(item) {
  const db = await openDB();
  if (!db) return;
  try {
    db.transaction(STORE, 'readwrite').objectStore(STORE).put(item);
  } catch {
    // Best-effort. A write failure here must never break the in-memory
    // queue — it only means this one item won't survive a tab kill.
  }
}

async function removeStoredItem(id) {
  const db = await openDB();
  if (!db) return;
  try {
    db.transaction(STORE, 'readwrite').objectStore(STORE).delete(id);
  } catch {
    // Best-effort, same reasoning as persistItem.
  }
}

async function hydrateFromDB() {
  if (hydrated) return;
  hydrated = true;
  const db = await openDB();
  if (!db) return;
  await new Promise((resolve) => {
    try {
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).getAll();
      req.onsuccess = () => {
        const stored = req.result || [];
        // Merge rather than replace: enqueueAck() may already have added
        // items synchronously (in-memory) before this async hydration
        // resolves. Stored items win on id collision (shouldn't happen —
        // ids are UUIDs — but favors the durable copy if it ever does).
        const byId = new Map(items.map((it) => [it.id, it]));
        for (const it of stored) byId.set(it.id, it);
        items = Array.from(byId.values());
        notify();
        if (items.some((it) => it.status === 'pending')) scheduleFlush();
        resolve();
      };
      req.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

function notify() {
  const snapshot = items.slice();
  for (const fn of Array.from(listeners)) {
    try {
      fn(snapshot);
    } catch (err) {
      console.error('[ackQueue] listener threw:', err);
    }
  }
}

function backoffMs(attempts) {
  return Math.min(BACKOFF_CAP_MS, BACKOFF_BASE_MS * 2 ** attempts);
}

function scheduleFlush() {
  if (retryTimer) return;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    flush();
  }, RETRY_TICK_MS);
}

/** Current items, newest first. Call once on mount before subscribe(). */
export function getSnapshot() {
  return items.slice();
}

/** The active (not yet pruned) queue item for one alert, if any. */
export function getItemForAlert(alertId) {
  return items.find((it) => String(it.alertId) === String(alertId));
}

/** @param {(items: AckQueueItem[]) => void} fn */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * Queue an acknowledgement for alertId. Idempotent by alertId: a second call
 * while one is already pending/syncing returns the existing item rather than
 * creating a duplicate — a double-tap on a stalled connection must not
 * become two outbox entries.
 */
export function enqueueAck(alertId) {
  const existing = items.find(
    (it) => String(it.alertId) === String(alertId)
      && (it.status === 'pending' || it.status === 'syncing'),
  );
  if (existing) return existing;

  const item = {
    id: (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`),
    alertId,
    createdAt: Date.now(),
    status: 'pending',
    attempts: 0,
    lastAttemptAt: null,
    lastError: null,
  };
  items = [item, ...items];
  notify();
  persistItem(item);
  flush();
  return item;
}

/** Clear a terminal (synced/failed) item once the UI has shown its result. */
export function dismiss(id) {
  items = items.filter((it) => it.id !== id);
  notify();
  removeStoredItem(id);
}

/**
 * Attempt every pending item once. Safe to call opportunistically and often
 * — 'online' events, WS reconnect, visibility change, and the internal
 * retry tick all call this; per-item backoff (not a global lock beyond
 * `flushing`) is what prevents hammering a still-down connection.
 */
export async function flush() {
  if (flushing) return;
  flushing = true;
  try {
    const token = (() => {
      try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
    })();
    if (!token) return; // not logged in — nothing to authenticate the ack with

    const now = Date.now();
    const due = items.filter((it) => {
      if (it.status !== 'pending') return false;
      if (!it.lastAttemptAt) return true;
      return now - it.lastAttemptAt >= backoffMs(it.attempts);
    });

    for (const item of due) {
      setStatus(item.id, { status: 'syncing' });
      let outcome;
      try {
        const res = await fetch(`${API}/alerts/${item.alertId}/ack`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          outcome = { status: 'synced', lastAttemptAt: Date.now(), lastError: null };
        } else if (res.status === 409) {
          // A real rejection, not a network problem — the alert's state
          // changed server-side while this was queued (e.g. escalated to
          // someone else). Retrying forever would be wrong; this is terminal.
          const body = await res.json().catch(() => ({}));
          outcome = {
            status: 'failed',
            lastAttemptAt: Date.now(),
            lastError: body.reason || 'Alert can no longer be acknowledged.',
          };
        } else {
          // Transient server-side problem (5xx, etc.) — same treatment as a
          // network failure: stay pending, back off, try again.
          outcome = {
            status: 'pending',
            attempts: item.attempts + 1,
            lastAttemptAt: Date.now(),
            lastError: `Server error (${res.status})`,
          };
        }
      } catch (err) {
        // fetch() threw — the actual "connection is dropped" case this
        // module exists for.
        outcome = {
          status: 'pending',
          attempts: item.attempts + 1,
          lastAttemptAt: Date.now(),
          lastError: err?.message || 'Network error',
        };
      }
      setStatus(item.id, outcome);
      if (outcome.status === 'synced' || outcome.status === 'failed') {
        // Do NOT also prune in this same synchronous stretch. setStatus()
        // above already called notify() once with the item present and
        // terminal — a second notify() right after, with the item gone,
        // would land in the same React batch as the first (this was a real
        // bug, caught by a component test: the intermediate 'synced' state
        // was invisible to any subscriber, because React collapsed both
        // updates into one and only the LAST — item removed — ever
        // rendered). A watching component (AlertDetailCard, RoutingBadge)
        // sees the terminal status and calls dismiss() itself on its own,
        // later event-loop turn. This timeout is only a safety net for a
        // terminal item nothing is watching (e.g. queued while no matching
        // card was mounted) — dismiss() a moment later than any component
        // would, so it never races an explicit one.
        scheduleAutoPrune(item.id, outcome.status);
      }
    }
  } finally {
    flushing = false;
    if (items.some((it) => it.status === 'pending')) scheduleFlush();
  }
}

const AUTO_PRUNE_MS = 15000;

/**
 * Remove a terminal item after a delay, unless it was already dismissed (by
 * a component that saw the terminal status and called dismiss() itself) or
 * re-queued in the meantime (status no longer matches what we scheduled
 * this for).
 */
function scheduleAutoPrune(id, expectedStatus) {
  setTimeout(() => {
    const current = items.find((it) => it.id === id);
    if (current && current.status === expectedStatus) dismiss(id);
  }, AUTO_PRUNE_MS);
}

function setStatus(id, patch) {
  items = items.map((it) => (it.id === id ? { ...it, ...patch } : it));
  notify();
  const updated = items.find((it) => it.id === id);
  if (updated) persistItem(updated);
}

// Fire-and-forget hydration on module load. Consumers don't need to await
// this — enqueueAck()/getSnapshot() work correctly before it resolves, it
// only affects whether items from a previous session appear.
hydrateFromDB();

// Backstops. All three call the same flush(); per-item backoff (above)
// is what keeps this from being a busy-loop against a dead connection.
if (typeof window !== 'undefined') {
  window.addEventListener('online', () => flush());
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') flush();
  });
}
