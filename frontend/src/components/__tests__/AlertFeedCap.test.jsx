// Alert feed DOM-growth cap (Day 18, checkpoint Test 7).
//
// The failure this guards against is slow and invisible until it isn't: a
// control room left open for a shift receives one alert every ~30s, and
// without a cap every one of them stays in the DOM as an AlertRow carrying
// badges, countdown timers and expandable sub-panels. After eight hours that
// is ~1000 live rows. Nothing warns you; the tab just gets slower and then
// the demo stutters.
//
// Asserting the cap directly is the only way to know it holds — it produces
// no visible symptom until the machine is already struggling.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

class FakeWS {
  static OPEN = 1;
  constructor() { this.readyState = 1; setTimeout(() => this.onopen?.(), 0); }
  send() {} close() {}
}
vi.stubGlobal('WebSocket', FakeWS);

const authFetch = vi.fn();
vi.mock('../../context/AuthContext', () => ({
  useAuth: () => ({ authFetch, token: 't', isAdmin: true, user: { role: 'admin' } }),
  API: 'http://localhost:8000/api/v1',
}));

import { WebSocketProvider } from '../../context/WebSocketContext';
import AlertFeed from '../AlertFeed';

const ok = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });

/** N alerts, newest id first — the shape GET /alerts returns. */
function makeAlerts(n) {
  return Array.from({ length: n }, (_, i) => ({
    id: n - i,
    alert_type: 'LOITERING',
    subject_label: `alert ${n - i}`,
    status: 'new',
    lifecycle_status: 'OPEN',
    created_at: new Date().toISOString(),
    camera_name: 'Cam',
    metadata: {},
  }));
}

beforeEach(() => {
  authFetch.mockReset();
  vi.stubGlobal('fetch', vi.fn(() => ok({})));
});
afterEach(() => cleanup());

function mountWith(alerts) {
  authFetch.mockImplementation((url) => {
    const u = String(url);
    if (u.includes('/alerts?')) return ok(alerts);
    if (u.includes('/zone-incidents')) return ok([]);
    if (u.includes('/routing/')) return ok([]);
    return ok({});
  });
  return render(<WebSocketProvider><AlertFeed /></WebSocketProvider>);
}

describe('AlertFeed DOM cap', () => {
  it('renders every alert when the server returns fewer than the cap', async () => {
    const { container } = mountWith(makeAlerts(12));
    await screen.findByText('alert 12');
    await waitFor(() =>
      expect(container.querySelectorAll('.alert-row-wrap').length).toBe(12));
  });

  it('caps the rendered rows at 200 when the server returns more', async () => {
    // 500 rows is what an over-long window or a busy shift produces. The
    // component must not render all of them.
    const { container } = mountWith(makeAlerts(500));
    await screen.findByText('alert 500');
    await waitFor(() => {
      const rows = container.querySelectorAll('.alert-row-wrap').length;
      expect(rows).toBeLessThanOrEqual(200);
      expect(rows).toBeGreaterThan(0);
    });
  });

  it('keeps the NEWEST alerts when capping, not the oldest', async () => {
    mountWith(makeAlerts(500));
    // Newest (id 500) must survive; oldest (id 1) must be the one dropped.
    // Capping from the wrong end would silently hide the alerts that matter.
    expect(await screen.findByText('alert 500')).toBeTruthy();
    await waitFor(() => expect(screen.queryByText('alert 1')).toBeNull());
  });

  it('stays capped as live alerts stream in over a long run', async () => {
    // Start just under the cap, then push 40 live alerts through the same
    // prop path the WebSocket handler uses. This is the shift-length
    // scenario compressed: the DOM must plateau at the cap, not keep growing.
    authFetch.mockImplementation((url) => {
      const u = String(url);
      if (u.includes('/alerts?')) return ok(makeAlerts(195));
      if (u.includes('/zone-incidents')) return ok([]);
      if (u.includes('/routing/')) return ok([]);
      return ok({});
    });

    const { container, rerender } = render(
      <WebSocketProvider><AlertFeed liveAlert={null} /></WebSocketProvider>,
    );
    await screen.findByText('alert 195');
    await waitFor(() =>
      expect(container.querySelectorAll('.alert-row-wrap').length).toBe(195));

    for (let i = 0; i < 40; i += 1) {
      const live = {
        id: 10_000 + i,
        alert_type: 'LOITERING',
        subject_label: `live ${i}`,
        status: 'new',
        lifecycle_status: 'OPEN',
        created_at: new Date().toISOString(),
        camera_name: 'Cam',
        metadata: {},
      };
      rerender(
        <WebSocketProvider><AlertFeed liveAlert={live} /></WebSocketProvider>,
      );
      // eslint-disable-next-line no-await-in-loop
      await screen.findByText(`live ${i}`);
    }

    const rows = container.querySelectorAll('.alert-row-wrap').length;
    // 195 + 40 = 235 without the cap. The assertion is that it plateaued.
    expect(rows).toBeLessThanOrEqual(200);
    expect(rows).toBeGreaterThanOrEqual(195);
    // Newest survived, oldest was evicted to make room.
    expect(screen.getByText('live 39')).toBeTruthy();
    expect(screen.queryByText('alert 1')).toBeNull();
  }, 15000);
});
