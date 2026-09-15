// frontend/src/components/__tests__/AlertDetailCard.test.jsx — Day 17.
//
// Exercises the REAL utils/ackQueue.js (not mocked) through the REAL
// useAckQueueItem hook — only the alert-detail fetch (authFetch) and the
// network layer the queue itself uses (global fetch) are mocked. This is
// deliberate: a test that mocks ackQueue itself would only prove the
// component calls enqueueAck() by name, not that a dropped connection
// actually results in a working queued-then-synced button. Unrelated child
// components (video player, feedback buttons, learning banner) are stubbed
// so failures here point at the ack flow, not at those.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

const authFetch = vi.fn();
vi.mock('../../context/AuthContext', () => ({
  useAuth: () => ({ authFetch }),
  API: 'http://localhost:8000/api/v1',
}));

let mockConnectionState = 'CONNECTED';
vi.mock('../../context/WebSocketContext', () => ({
  useConnectionState: () => mockConnectionState,
}));

vi.mock('../SafeHlsPlayer', () => ({ default: () => null }));
vi.mock('../FeedbackButtons', () => ({ default: () => null }));
vi.mock('../SystemLearningBanner', () => ({ default: () => null }));
vi.mock('../../utils/tts', () => ({
  speakCriticalAlert: () => {},
  flushPendingTTS: () => {},
}));

import AlertDetailCard from '../AlertDetailCard';
import { enqueueAck, getSnapshot as getQueueSnapshot, dismiss } from '../../utils/ackQueue';

const ok = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });

function alertDetailBody(id, overrides = {}) {
  return {
    id, camera_name: `Cam ${id}`, severity: 'CRITICAL', camera_id: `CAM-${id}`,
    created_at: new Date().toISOString(), routing: {}, ...overrides,
  };
}

beforeEach(() => {
  authFetch.mockReset();
  mockConnectionState = 'CONNECTED';
  localStorage.setItem('sg_token', 'test-jwt');
});

afterEach(() => {
  cleanup();
  // Queue state is a module singleton — drain anything a test left behind
  // so the next test starts clean regardless of unique alert ids used.
  for (const item of getQueueSnapshot()) dismiss(item.id);
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('AlertDetailCard: normal ack (connection healthy)', () => {
  it('shows Acknowledged after a successful POST', async () => {
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/detail')) return ok(alertDetailBody('1'));
      if (String(url).includes('/ack')) return ok({ status: 'acknowledged' });
      return ok({});
    });
    render(<AlertDetailCard alertId="1" onClose={() => {}} />);
    fireEvent.click(await screen.findByText('ACKNOWLEDGE'));
    expect(await screen.findByText('✓ Acknowledged')).toBeTruthy();
  });
});

describe('AlertDetailCard: dropped connection', () => {
  it('queues the tap instead of silently reverting to ACKNOWLEDGE', async () => {
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/detail')) return ok(alertDetailBody('2'));
      if (String(url).includes('/ack')) return Promise.reject(new TypeError('Failed to fetch'));
      return ok({});
    });
    // The queue's own retry uses global fetch directly, independent of
    // authFetch. Reject rather than hang forever — a real dropped
    // connection fails quickly, it doesn't stall indefinitely, and a
    // never-resolving mock here would leave ackQueue's module-level
    // `flushing` guard stuck true (its flush() never reaches `finally`),
    // silently breaking every later test in this file that shares the same
    // module singleton.
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('offline'))));

    render(<AlertDetailCard alertId="2" onClose={() => {}} />);
    fireEvent.click(await screen.findByText('ACKNOWLEDGE'));

    const button = await screen.findByText('⏳ Queued — will send when back online');
    expect(button.closest('button').disabled).toBe(true);
    // The critical negative assertion: this must NOT be the pre-existing bug
    // where a network failure silently reverts to a fresh, re-tappable
    // button with no indication anything happened.
    expect(screen.queryByText('ACKNOWLEDGE')).toBeNull();
  });

  it('transitions to Acknowledged automatically once the queue syncs — no re-tap needed', async () => {
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/detail')) return ok(alertDetailBody('3'));
      if (String(url).includes('/ack')) return Promise.reject(new TypeError('offline'));
      return ok({});
    });
    let resolveFetch;
    vi.stubGlobal('fetch', vi.fn(() => new Promise((res) => { resolveFetch = res; })));

    render(<AlertDetailCard alertId="3" onClose={() => {}} />);
    fireEvent.click(await screen.findByText('ACKNOWLEDGE'));
    await screen.findByText('⏳ Queued — will send when back online');

    // Connection comes back; the queue's in-flight retry (already pending
    // from enqueueAck's own immediate flush() call) resolves successfully.
    resolveFetch({ ok: true, status: 200, json: async () => ({ status: 'acknowledged' }) });

    expect(await screen.findByText('✓ Acknowledged')).toBeTruthy();
    expect(screen.queryByText('⏳ Queued — will send when back online')).toBeNull();
  });

  it('shows a clear terminal failure if the queued ack is rejected on sync (409)', async () => {
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/detail')) return ok(alertDetailBody('4'));
      if (String(url).includes('/ack')) return Promise.reject(new TypeError('offline'));
      return ok({});
    });
    let resolveFetch;
    vi.stubGlobal('fetch', vi.fn(() => new Promise((res) => { resolveFetch = res; })));

    render(<AlertDetailCard alertId="4" onClose={() => {}} />);
    fireEvent.click(await screen.findByText('ACKNOWLEDGE'));
    await screen.findByText('⏳ Queued — will send when back online');

    resolveFetch({
      ok: false, status: 409,
      json: async () => ({ reason: 'alert was reassigned while offline' }),
    });

    expect(await screen.findByText(/Could not sync your acknowledgement/))
      .toBeTruthy();
    expect(screen.getByText(/alert was reassigned while offline/)).toBeTruthy();
    // Not stuck on a disabled "Queued" forever, and not falsely claiming success.
    expect(screen.queryByText('✓ Acknowledged')).toBeNull();
  });

  it('a real-time 409 (connection healthy, server says no) is NOT queued', async () => {
    // Distinguishes a live rejection from a dropped connection — queueing
    // this would retry a decision the server already made, live, in front
    // of the officer.
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/detail')) return ok(alertDetailBody('5'));
      if (String(url).includes('/ack')) {
        return Promise.resolve({
          ok: false, status: 409, json: async () => ({ reason: 'not routed' }),
        });
      }
      return ok({});
    });
    render(<AlertDetailCard alertId="5" onClose={() => {}} />);
    fireEvent.click(await screen.findByText('ACKNOWLEDGE'));
    expect(await screen.findByText('Cannot acknowledge — not routed')).toBeTruthy();
    expect(getQueueSnapshot().find((it) => it.alertId === '5')).toBeUndefined();
  });
});

describe('AlertDetailCard: reopened while a queued ack is still pending', () => {
  it('shows Queued immediately on mount, not a blank re-tappable button', async () => {
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/detail')) return ok(alertDetailBody('6'));
      return ok({});
    });
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('offline'))));

    // Simulate a previous mount having already queued this alert (card was
    // closed and reopened, or the app was reloaded, before it synced).
    enqueueAck('6');

    render(<AlertDetailCard alertId="6" onClose={() => {}} />);
    expect(await screen.findByText('⏳ Queued — will send when back online')).toBeTruthy();
    expect(screen.queryByText('ACKNOWLEDGE')).toBeNull();
  });
});
