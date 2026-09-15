// LiveDetectionPanel — live AI tracking view.
//
// The properties worth pinning are the ones that are invisible by
// inspection and dangerous if wrong on a surveillance dashboard:
//   1. stale boxes are swept, so a dead camera cannot leave phantom
//      detections frozen on screen looking live
//   2. it consumes the SHARED socket rather than opening its own
//   3. per-camera separation - CAM_01's tracks never render under CAM_02
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup } from '@testing-library/react';

// Capture the handlers the panel subscribes with, so tests can push events
// exactly the way the real WebSocketContext would.
const handlers = {};
vi.mock('../../context/WebSocketContext', () => ({
  useWebSocketEvent: (type, fn) => { handlers[type] = fn; },
}));

// DetectionCanvas draws to a real <canvas>; jsdom has no 2d context, so it
// is stubbed to render the track ids it was handed. That keeps this test
// about panel STATE (what reaches the canvas) rather than pixel output.
vi.mock('../DetectionCanvas', () => ({
  DetectionCanvas: ({ tracks }) => (
    <div data-testid="canvas">{Object.keys(tracks).sort().join(',')}</div>
  ),
}));

import LiveDetectionPanel from '../LiveDetectionPanel';

const evt = (camera_id, track_id) => ({
  camera_id, track_id, bbox: [10, 10, 50, 80], confidence: 0.9,
});

beforeEach(() => {
  for (const k of Object.keys(handlers)) delete handlers[k];
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe('LiveDetectionPanel', () => {
  it('renders an empty state before any detection arrives', () => {
    render(<LiveDetectionPanel />);
    expect(screen.getByText(/No detections received yet/)).toBeTruthy();
  });

  it('renders tracks from a detection_event', () => {
    render(<LiveDetectionPanel />);
    act(() => { handlers.detection_event(evt('CAM_01', 7)); });
    expect(screen.getByTestId('canvas').textContent).toBe('7');
  });

  it('sweeps stale tracks so a dead feed leaves no phantom boxes', () => {
    render(<LiveDetectionPanel />);
    act(() => { handlers.detection_event(evt('CAM_01', 7)); });
    expect(screen.getByTestId('canvas').textContent).toBe('7');

    // No further events: after the TTL the box must disappear on its own,
    // without depending on a track_expired message that may never arrive.
    act(() => { vi.advanceTimersByTime(6000); });
    expect(screen.getByTestId('canvas').textContent).toBe('');
  });

  it('removes a track on an explicit track_expired event', () => {
    render(<LiveDetectionPanel />);
    act(() => {
      handlers.detection_event(evt('CAM_01', 7));
      handlers.detection_event(evt('CAM_01', 8));
    });
    expect(screen.getByTestId('canvas').textContent).toBe('7,8');

    act(() => { handlers.track_expired({ camera_id: 'CAM_01', track_id: 7 }); });
    expect(screen.getByTestId('canvas').textContent).toBe('8');
  });

  it('keeps cameras separate — one camera\'s tracks never leak into another', () => {
    render(<LiveDetectionPanel />);
    act(() => {
      handlers.detection_event(evt('CAM_01', 1));
      handlers.detection_event(evt('CAM_02', 99));
    });
    // Two cameras seen -> selector appears, and the active camera shows only
    // its own track.
    expect(screen.getByTestId('canvas').textContent).toBe('1');
    expect(screen.getByText(/CAM_02 \(1\)/)).toBeTruthy();
  });

  it('subscribes to the shared socket instead of opening its own', () => {
    // If this component ever constructed a WebSocket directly it would be a
    // second connection with its own auth handshake — the exact thing the
    // Day 18 single-socket design exists to prevent.
    const spy = vi.fn();
    vi.stubGlobal('WebSocket', spy);
    render(<LiveDetectionPanel />);
    expect(spy).not.toHaveBeenCalled();
    expect(typeof handlers.detection_event).toBe('function');
    vi.unstubAllGlobals();
  });
});
