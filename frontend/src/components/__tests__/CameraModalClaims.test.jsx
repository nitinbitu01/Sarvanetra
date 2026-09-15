// CameraModal must not claim evidence operations it did not perform.
//
// The modal previously shipped a "Capture Frame" button that popped
// "✓ Forensic Snapshot Captured & Hashed" while doing neither. That collides
// with Day 13, where sealing an evidence clip really does compute a SHA-256
// and emit a chain-of-custody PDF — so an operator who sees that toast has
// every reason to believe the real pipeline ran.
//
// These tests assert the absence of the claim. Absence is exactly the kind of
// property that silently regresses when someone restyles a component, which
// is why it is pinned here rather than left to review.

import { describe, it, expect, afterEach, vi, beforeEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import CameraModal from '../CameraModal';
import { AuthProvider } from '../../context/AuthContext';

const camera = {
  id: 7, camera_id: 'CAM-07', name: 'Gate 7',
  zone: 'Central', status: 'ONLINE', gps_lat: 23.0489, gps_lon: 72.5714,
};

// CameraModal now calls useAuth() for authFetch (the snapshot and telemetry
// polls, added when the fake video/stats were replaced with real ones — see
// the file's own header comment). Real AuthProvider rather than a bespoke
// mock context, so the test exercises the same provider production does.
// Its effects call fetch(); mocked to a rejected promise so a real network
// call is never attempted from a unit test, and the component's own
// try/catch around each poll is exactly what is expected to absorb that.
function renderModal(props = {}) {
  return render(
    <AuthProvider>
      <CameraModal camera={camera} onClose={() => {}} {...props} />
    </AuthProvider>,
  );
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('no network in unit test'))));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('CameraModal evidence claims', () => {
  it('offers no capture/snapshot control', () => {
    renderModal();
    const labels = Array.from(document.querySelectorAll('button'))
      .map((b) => (b.textContent || '').toLowerCase());
    expect(labels.some((t) => t.includes('capture'))).toBe(false);
    // NOTE: not asserting the absence of "snapshot" here — the live-video
    // feature added after this test was written is itself named "snapshot"
    // (GET /cameras/{id}/snapshot). What this test guards against is a
    // fabricated CAPTURE CONTROL, i.e. a button — checked directly above —
    // not the word, which now legitimately appears in real status text.
  });

  it('never renders the words "hashed", "forensic" or "sealed"', () => {
    const { container } = renderModal();
    const text = (container.textContent || '').toLowerCase();
    for (const claim of ['hashed', 'forensic', 'sealed', 'chain of custody']) {
      expect(text).not.toContain(claim);
    }
  });

  it('still renders the camera facts that ARE real', () => {
    // Removing the false claim must not strip the honest content with it.
    renderModal();
    expect(screen.getByText('Gate 7')).toBeTruthy();
    expect(screen.getByText('CAM-07')).toBeTruthy();
    // Real GPS from the record, not the Ahmedabad fallback.
    expect(screen.getByText('23.0489, 72.5714')).toBeTruthy();
  });
});
