// frontend/src/utils/__tests__/ackQueue.test.js — Day 17 offline ACK outbox.
//
// ackQueue.js is a singleton module (one shared `items` array, listeners
// registered at load time), so each test gets a genuinely fresh module
// instance via vi.resetModules() + a dynamic import — otherwise a queued
// item from one test would leak into the next.
//
// No IndexedDB polyfill is installed here on purpose: jsdom doesn't provide
// one, and the module is designed to feature-detect that and degrade to
// in-memory-only. Testing against that absence is testing the real
// environment this suite runs in, not working around a gap — see the
// dedicated "no IndexedDB" test below.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const TOKEN = 'test-jwt';

function jsonResponse(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

async function freshModule() {
  vi.resetModules();
  return import('../ackQueue.js');
}

beforeEach(() => {
  localStorage.setItem('sg_token', TOKEN);
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe('ackQueue: enqueue + dedup', () => {
  it('creates a pending item', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {}))); // never resolves
    const q = await freshModule();
    const item = q.enqueueAck('42');
    expect(item.status).toBe('pending');
    expect(item.alertId).toBe('42');
    expect(q.getSnapshot()).toHaveLength(1);
  });

  it('does not create a second item for the same alert while one is pending', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    const q = await freshModule();
    const first = q.enqueueAck('42');
    const second = q.enqueueAck('42');
    expect(second.id).toBe(first.id);
    expect(q.getSnapshot()).toHaveLength(1);
  });

  it('getItemForAlert finds the item by alertId', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    const q = await freshModule();
    q.enqueueAck('99');
    expect(q.getItemForAlert('99')?.alertId).toBe('99');
    expect(q.getItemForAlert('does-not-exist')).toBeUndefined();
  });
});

describe('ackQueue: flush outcomes', () => {
  it('a successful sync transitions to synced (observable) before being pruned', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse(200, { status: 'acknowledged' })));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    const notified = [];
    q.subscribe((items) => notified.push(items.map((i) => i.status)));

    q.enqueueAck('42');
    await vi.waitFor(() => expect(q.getItemForAlert('42')?.status).toBe('synced'));

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/alerts/42/ack'),
      expect.objectContaining({
        method: 'POST',
        headers: { Authorization: `Bearer ${TOKEN}` },
      }),
    );
    // Regression guard: 'synced' must be its own observable notification,
    // not collapsed into the same tick as removal. A subscriber (e.g. a
    // React component batching state updates) that only ever sees the FINAL
    // notify() in a same-tick pair would see the item vanish without ever
    // having observed 'synced' — exactly the bug this asserts against.
    expect(notified.some((snapshot) => snapshot.includes('syncing'))).toBe(true);
    expect(notified.some((snapshot) =>
      snapshot.length === 1 && snapshot[0] === 'synced')).toBe(true);

    // Nothing dismissed it explicitly — the safety-net auto-prune should
    // still remove it eventually so an unwatched terminal item doesn't
    // linger forever.
    await vi.advanceTimersByTimeAsync(20_000);
    expect(q.getSnapshot()).toHaveLength(0);
  });

  it('dismiss()ing a synced item cancels the safety-net auto-prune from double-firing badly', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse(200, {})));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    q.enqueueAck('7');
    await vi.waitFor(() => expect(q.getItemForAlert('7')?.status).toBe('synced'));
    q.dismiss(q.getItemForAlert('7').id);
    expect(q.getSnapshot()).toHaveLength(0);

    // The auto-prune timer set when the item became 'synced' will still
    // fire later; it must find nothing to do rather than throw or affect a
    // later, unrelated item that happens to reuse... nothing, since ids are
    // UUIDs, but it must not error against an already-removed id.
    expect(() => vi.advanceTimersByTime(20_000)).not.toThrow();
  });

  it('a 409 is terminal: marks failed and does NOT retry', async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(jsonResponse(409, { reason: 'alert was reassigned' })));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    q.enqueueAck('42');
    await vi.waitFor(() => {
      expect(q.getItemForAlert('42')?.status).toBe('failed');
    });
    expect(q.getItemForAlert('42').lastError).toBe('alert was reassigned');

    const callsAfterFirstFailure = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    // A terminal 'failed' item must never be retried — retrying a real
    // rejection forever would just be a slower version of the exact bug
    // this queue exists to fix (a tap that silently does nothing useful).
    expect(fetchMock.mock.calls.length).toBe(callsAfterFirstFailure);
  });

  it('a network failure stays pending and is retried, not dropped', async () => {
    const fetchMock = vi.fn(() => Promise.reject(new Error('network drop')));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    q.enqueueAck('42');
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(q.getItemForAlert('42').status).toBe('pending');
    expect(q.getItemForAlert('42').attempts).toBe(1);
    expect(q.getItemForAlert('42').lastError).toContain('network drop');

    // Advance past the backoff window (2s * 2^1 = 4s) plus the outer retry
    // tick so the queue's own timer gets a chance to fire again.
    await vi.advanceTimersByTimeAsync(10_000);
    await vi.waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it('a 5xx is treated like a network failure, not a real rejection', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse(503, {})));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    q.enqueueAck('42');
    await vi.waitFor(() => expect(q.getItemForAlert('42')?.attempts).toBe(1));
    expect(q.getItemForAlert('42').status).toBe('pending');
  });

  it('does nothing if no auth token is present (never sends an unauthenticated ack)', async () => {
    localStorage.clear();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    q.enqueueAck('42');
    await vi.advanceTimersByTimeAsync(100);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(q.getItemForAlert('42').status).toBe('pending');
  });
});

describe('ackQueue: pub/sub and dismiss', () => {
  it('subscribers are notified on every state change', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    const q = await freshModule();
    const calls = [];
    const unsubscribe = q.subscribe((items) => calls.push(items.length));

    q.enqueueAck('1');
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.at(-1)).toBe(1);

    unsubscribe();
    const before = calls.length;
    q.enqueueAck('2');
    // After unsubscribing, this listener must not fire again.
    expect(calls.length).toBe(before);
  });

  it('dismiss() removes a resolved item', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse(409, { reason: 'x' })));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    q.enqueueAck('42');
    await vi.waitFor(() => expect(q.getItemForAlert('42')?.status).toBe('failed'));
    const id = q.getItemForAlert('42').id;
    q.dismiss(id);
    expect(q.getItemForAlert('42')).toBeUndefined();
  });
});

describe('ackQueue: environment without IndexedDB', () => {
  it('does not throw and still queues/flushes correctly', async () => {
    // jsdom provides no IndexedDB implementation by default — this IS that
    // environment, not a simulation of it. If the module crashed here, it
    // would crash for real in any browser without IDB support too.
    expect(typeof indexedDB).toBe('undefined');

    const fetchMock = vi.fn(() => Promise.resolve(jsonResponse(200, {})));
    vi.stubGlobal('fetch', fetchMock);
    const q = await freshModule();

    expect(() => q.enqueueAck('1')).not.toThrow();
    await vi.waitFor(() => expect(q.getItemForAlert('1')?.status).toBe('synced'));
  });
});
