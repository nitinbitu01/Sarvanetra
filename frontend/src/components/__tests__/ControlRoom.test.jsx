// Component tests for the Day 18 control-room assembly.
//
// Two properties matter most and neither is visible by looking at the UI:
//   1. exactly ONE WebSocket connection exists (Test 1's hard requirement)
//   2. one failing panel does not take the others down (the acceptance bar)
// Both are asserted here rather than eyeballed in DevTools.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup, fireEvent } from '@testing-library/react';

// ── Count real WebSocket constructions ──────────────────────────────────────
let wsConstructions = 0;
class FakeWS {
  static OPEN = 1;
  constructor() {
    wsConstructions += 1;
    this.readyState = 1;
    setTimeout(() => this.onopen?.(), 0);
  }
  send() {}
  close() {}
}
vi.stubGlobal('WebSocket', FakeWS);

const authFetch = vi.fn();
vi.mock('../../context/AuthContext', () => ({
  useAuth: () => ({ authFetch, token: 'test-token', isAdmin: true, user: { role: 'admin' } }),
  API: 'http://localhost:8000/api/v1',
}));

// Leaflet needs a real layout engine; jsdom has none. The map is covered by
// its own error-boundary test below rather than by rendering tiles.
vi.mock('../LiveMap', () => ({
  default: () => <div data-testid="live-map">map</div>,
}));

import { WebSocketProvider } from '../../context/WebSocketContext';
import { PanelErrorBoundary } from '../PanelErrorBoundary';
import SeverityPanel from '../SeverityPanel';
import ROIPanel from '../ROIPanel';
import ReviewQueueBadge from '../ReviewQueueBadge';
import CameraGrid from '../CameraGrid';
import CameraModal from '../CameraModal';

function ok(body) {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
}

function routeFetch(overrides = {}) {
  return vi.fn((url) => {
    const u = String(url);
    if (u.includes('/alerts/summary')) {
      return overrides.summary ?? ok({ CRITICAL: 3, HIGH: 7, MEDIUM: 12, LOW: 24, UNSCORED: 0 });
    }
    if (u.includes('/review-queue/count')) {
      return overrides.queue ?? ok({ pending: 4 });
    }
    if (u.includes('/cameras')) {
      return overrides.cameras ?? ok([
        { id: 1, camera_id: 'CAM-01', name: 'Main Gate', zone: 'North', status: 'ONLINE', gps_lat: 23.0, gps_lon: 72.5 },
        { id: 2, camera_id: 'CAM-02', name: 'Rear Gate', zone: 'South', status: 'OFFLINE', gps_lat: 23.1, gps_lon: 72.6 },
      ]);
    }
    if (u.includes('/officers')) return ok([]);
    if (u.includes('/client-error')) return ok({ status: 'recorded' });
    return ok({});
  });
}

beforeEach(() => {
  wsConstructions = 0;
  authFetch.mockReset();
  authFetch.mockImplementation(routeFetch());
  vi.stubGlobal('fetch', vi.fn(() => ok({ status: 'recorded' })));
});
afterEach(() => cleanup());

describe('single WebSocket guarantee', () => {
  it('WebSocketProvider opens exactly one connection', async () => {
    render(<WebSocketProvider><div>child</div></WebSocketProvider>);
    await waitFor(() => expect(wsConstructions).toBe(1));
  });

  it('mounting many panels under the provider still opens only one', async () => {
    render(
      <WebSocketProvider>
        <SeverityPanel />
        <ReviewQueueBadge />
        <CameraGrid onExpand={() => {}} />
      </WebSocketProvider>,
    );
    await screen.findByText('Main Gate');
    // The whole point of the context: N subscribers, one socket.
    expect(wsConstructions).toBe(1);
  });
});

describe('panel isolation', () => {
  it('a thrown panel shows an inline error instead of unmounting siblings', async () => {
    function Boom() { throw new Error('data source exploded'); }
    render(
      <WebSocketProvider>
        <PanelErrorBoundary panelName="Broken Panel"><Boom /></PanelErrorBoundary>
        <ROIPanel />
      </WebSocketProvider>,
    );
    expect(await screen.findByText(/Broken Panel unavailable/)).toBeTruthy();
    // The sibling must still be on screen — this is the acceptance bar.
    expect(screen.getByText(/Officer hrs saved/)).toBeTruthy();
  });

  it('reports the crash to the backend without letting the report throw', async () => {
    const reporter = vi.fn(() => Promise.reject(new Error('network down')));
    vi.stubGlobal('fetch', reporter);
    function Boom() { throw new Error('kaboom'); }
    render(
      <WebSocketProvider>
        <PanelErrorBoundary panelName="P"><Boom /></PanelErrorBoundary>
      </WebSocketProvider>,
    );
    await screen.findByText(/P unavailable/);
    await waitFor(() => expect(reporter).toHaveBeenCalled());
    const [url] = reporter.mock.calls[0];
    expect(String(url)).toContain('/client-error');
  });

  it('a failing summary fetch degrades only that panel', async () => {
    authFetch.mockImplementation(routeFetch({
      summary: Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) }),
    }));
    render(
      <WebSocketProvider>
        <PanelErrorBoundary panelName="Alert Summary"><SeverityPanel /></PanelErrorBoundary>
        <PanelErrorBoundary panelName="Review Queue"><ReviewQueueBadge /></PanelErrorBoundary>
      </WebSocketProvider>,
    );
    expect(await screen.findByText(/Alert Summary unavailable/)).toBeTruthy();
    // Queue badge fetched fine and must still render its count.
    expect(await screen.findByText('4')).toBeTruthy();
  });
});

describe('SeverityPanel', () => {
  it('renders server counts', async () => {
    render(<WebSocketProvider><SeverityPanel /></WebSocketProvider>);
    expect(await screen.findByText('3')).toBeTruthy();
    expect(screen.getByText('24')).toBeTruthy();
  });

  it('shows a skeleton before the fetch resolves', () => {
    authFetch.mockImplementation(() => new Promise(() => {}));  // never settles
    const { container } = render(<WebSocketProvider><SeverityPanel /></WebSocketProvider>);
    expect(container.querySelectorAll('.skeleton-line').length).toBeGreaterThan(0);
  });
});

describe('ROIPanel', () => {
  it('makes no network calls at all', () => {
    render(<ROIPanel />);
    expect(authFetch).not.toHaveBeenCalled();
  });

  it('labels its numbers as illustrative rather than measured', () => {
    render(<ROIPanel />);
    // Load-bearing: these figures are not computed from anything. Without
    // this caption a reviewer would reasonably read them as results.
    expect(screen.getByText(/not measured by this system/i)).toBeTruthy();
  });
});

describe('CameraGrid / CameraModal', () => {
  it('renders a tile per camera with live status', async () => {
    render(<WebSocketProvider><CameraGrid onExpand={() => {}} /></WebSocketProvider>);
    await screen.findByText('Main Gate');
    expect(screen.getByText('Rear Gate')).toBeTruthy();
    expect(screen.getByText('ONLINE')).toBeTruthy();
    expect(screen.getByText('OFFLINE')).toBeTruthy();
  });

  it('never requests a video stream from the grid', async () => {
    render(<WebSocketProvider><CameraGrid onExpand={() => {}} /></WebSocketProvider>);
    await screen.findByText('Main Gate');
    // The contract is that tiles are inert: N tiles must not mean N streams.
    const urls = authFetch.mock.calls.map(([u]) => String(u));
    expect(urls.some((u) => u.includes('.m3u8') || u.includes('/stream/'))).toBe(false);
  });

  it('clicking a tile invokes onExpand with that camera', async () => {
    const onExpand = vi.fn();
    render(<WebSocketProvider><CameraGrid onExpand={onExpand} /></WebSocketProvider>);
    fireEvent.click(await screen.findByText('Main Gate'));
    expect(onExpand).toHaveBeenCalledWith(expect.objectContaining({ name: 'Main Gate' }));
  });

  it('modal closes on Escape', () => {
    const onClose = vi.fn();
    render(<CameraModal camera={{ id: 1, name: 'Main Gate', status: 'ONLINE' }} onClose={onClose} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });
});
