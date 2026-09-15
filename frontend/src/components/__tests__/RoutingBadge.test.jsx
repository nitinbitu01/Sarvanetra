// frontend/src/components/__tests__/RoutingBadge.test.jsx — Day 17.
//
// RoutingBadge is the control-room-side ACK surface, sharing the exact same
// utils/ackQueue.js outbox as AlertDetailCard.jsx (the officer-phone-side
// surface, covered in AlertDetailCard.test.jsx). This file exists so a
// dropped connection is caught on BOTH real ACK buttons, not just one —
// before this task, RoutingBadge's ack() had the identical bug: a caught
// network error only logged to console and left the button silently
// re-clickable.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

const authFetch = vi.fn();
vi.mock('../../context/AuthContext', () => ({
  useAuth: () => ({ authFetch }),
  API: 'http://localhost:8000/api/v1',
}));

let mockConnectionState = 'CONNECTED';
vi.mock('../../context/WebSocketContext', () => ({
  useConnectionState: () => mockConnectionState,
}));

import RoutingBadge from '../RoutingBadge';
import { getSnapshot as getQueueSnapshot, dismiss } from '../../utils/ackQueue';

const routedActive = {
  status: 'ROUTED', officer_id: 9, officer_name: 'Officer Patel',
  timeout_seconds: 120, seconds_elapsed: 10, routed_alert_id: 501,
};

beforeEach(() => {
  authFetch.mockReset();
  mockConnectionState = 'CONNECTED';
  localStorage.setItem('sg_token', 'test-jwt');
});

afterEach(() => {
  cleanup();
  for (const item of getQueueSnapshot()) dismiss(item.id);
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('RoutingBadge: dropped connection', () => {
  it('queues the ACK instead of silently reverting to a re-clickable button', async () => {
    authFetch.mockImplementation(() => Promise.reject(new TypeError('offline')));
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('offline'))));

    render(<RoutingBadge alertId="10" routing={routedActive} onRoutingChange={() => {}} />);
    fireEvent.click(screen.getByText('ACK'));

    const btn = await screen.findByText('⏳ Queued');
    expect(btn.closest('button').disabled).toBe(true);
    expect(screen.queryByText('ACK')).toBeNull();
  });

  it('calls onRoutingChange(ACKNOWLEDGED) once the queued item syncs', async () => {
    authFetch.mockImplementation(() => Promise.reject(new TypeError('offline')));
    let resolveFetch;
    vi.stubGlobal('fetch', vi.fn(() => new Promise((res) => { resolveFetch = res; })));

    const onRoutingChange = vi.fn();
    render(<RoutingBadge alertId="11" routing={routedActive} onRoutingChange={onRoutingChange} />);
    fireEvent.click(screen.getByText('ACK'));
    await screen.findByText('⏳ Queued');

    resolveFetch({ ok: true, status: 200, json: async () => ({ status: 'acknowledged' }) });

    await vi.waitFor(() => {
      expect(onRoutingChange).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'ACKNOWLEDGED' }),
      );
    });
  });
});

describe('RoutingBadge: reopened while a queued ack is already pending', () => {
  it('shows Queued on mount rather than a fresh ACK button', async () => {
    // Reject rather than hang — see the comment in AlertDetailCard.test.jsx
    // on why a never-resolving mock here would break later tests sharing
    // the same ackQueue module singleton.
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('offline'))));
    const { enqueueAck } = await import('../../utils/ackQueue');
    enqueueAck('12');

    render(<RoutingBadge alertId="12" routing={routedActive} onRoutingChange={() => {}} />);
    expect(await screen.findByText('⏳ Queued')).toBeTruthy();
  });
});
