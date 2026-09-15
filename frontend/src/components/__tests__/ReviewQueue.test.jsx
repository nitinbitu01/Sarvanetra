// Component tests for ReviewQueue.jsx.
//
// These cover the one part of Day 12 that no Python script can reach. The
// backend verification proves reid_review_created is BROADCAST; it cannot
// prove React acts on it. That gap was previously reported as "needs a
// browser" — but the actual claim under test is narrower than a browser:
// "when this prop changes, does the component refetch?" jsdom answers that
// deterministically, without a dev server, a real socket, or a driver.
//
// What is still NOT covered here: the real browser WebSocket transport
// inside useDashboardSocket (a one-line `if (msg.type === ...) handler(msg)`
// dispatch shared with the alerts and camera flows). This tests the consumer
// of that dispatch, not the socket itself.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

// AuthContext reads localStorage and issues real fetches; stub it so these
// tests exercise ReviewQueue's own logic rather than the auth stack.
const authFetch = vi.fn();
vi.mock('../../context/AuthContext', () => ({
  useAuth: () => ({ authFetch, isAdmin: true, user: { role: 'officer' } }),
  API: 'http://localhost:8000/api/v1',
}));

import ReviewQueue from '../ReviewQueue';

const ITEM = {
  id: 1,
  local_track_id: 42,
  candidate_global_person_id: 7,
  similarity_score: 0.78,
  crop_image_path: 'output/crops/track_42/frame_1.jpg',
  candidate_reference_image_path: null,
  status: 'PENDING',
  time_gap_seconds: 120,
  created_at: '2026-08-21T00:00:00',
};

function jsonOk(body) {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
}

/** Route each call by URL so ordering between the queue fetch and the
 *  per-card journey fetch doesn't matter. */
function routeFetch({ queue = [ITEM], journey = { journey: [] } } = {}) {
  return vi.fn((url) => {
    if (String(url).includes('/candidate-journey')) return jsonOk(journey);
    if (String(url).includes('/review-queue')) return jsonOk(queue);
    return jsonOk({});
  });
}

const queueCalls = () =>
  authFetch.mock.calls.filter(
    ([u]) => String(u).includes('/review-queue') && !String(u).includes('/candidate-journey'),
  ).length;

const journeyCalls = () =>
  authFetch.mock.calls.filter(([u]) => String(u).includes('/candidate-journey')).length;

beforeEach(() => {
  authFetch.mockReset();
});
afterEach(() => {
  cleanup();
});

describe('ReviewQueue — live WS refresh', () => {
  it('loads the queue once on mount', async () => {
    authFetch.mockImplementation(routeFetch());
    render(<ReviewQueue wsEvents={null} />);
    await waitFor(() => expect(queueCalls()).toBe(1));
  });

  it('refetches when a reid_review_created event arrives', async () => {
    authFetch.mockImplementation(routeFetch());
    const { rerender } = render(<ReviewQueue wsEvents={null} />);
    await waitFor(() => expect(queueCalls()).toBe(1));

    // This is the assertion the Python suite could not make: the component
    // reacts to the event resolve_identity() broadcasts.
    rerender(<ReviewQueue wsEvents={{ type: 'reid_review_created', review_item_id: 2 }} />);
    await waitFor(() => expect(queueCalls()).toBe(2));
  });

  it('ignores unrelated WS event types', async () => {
    authFetch.mockImplementation(routeFetch());
    const { rerender } = render(<ReviewQueue wsEvents={null} />);
    await waitFor(() => expect(queueCalls()).toBe(1));

    rerender(<ReviewQueue wsEvents={{ type: 'new_alert', alert: { id: 9 } }} />);
    // Give any erroneous refetch a chance to land before asserting it didn't.
    await new Promise((r) => setTimeout(r, 50));
    expect(queueCalls()).toBe(1);
  });
});

describe('ReviewQueue — journey-so-far', () => {
  it('renders a multi-camera journey for the candidate', async () => {
    authFetch.mockImplementation(
      routeFetch({
        journey: {
          global_person_id: 7,
          total_sightings: 2,
          journey: [
            { journey_id: 1, camera_id: 'CAM-A', seen_at: '2026-08-21T00:00:00', confidence: 0.91, confidence_caveat: null },
            { journey_id: 2, camera_id: 'CAM-B', seen_at: '2026-08-21T00:20:00', confidence: 0.72, confidence_caveat: 'Long gap.' },
          ],
        },
      }),
    );
    render(<ReviewQueue wsEvents={null} />);

    await screen.findByText(/CAM-A/);
    await screen.findByText(/CAM-B/);
    expect(screen.getByText(/2 sightings/)).toBeTruthy();
    // A caveated hop must be visibly flagged — an officer relying on this
    // journey as corroboration needs to know one link was itself a guess.
    expect(screen.getByText(/confidence caveat/i)).toBeTruthy();
  });

  it('degrades to a clear "no journey yet" state for a first-time candidate', async () => {
    authFetch.mockImplementation(routeFetch({ journey: { journey: [] } }));
    render(<ReviewQueue wsEvents={null} />);
    expect(await screen.findByText(/No prior sightings recorded/i)).toBeTruthy();
  });

  it('distinguishes a FAILED journey lookup from "no sightings"', async () => {
    // The dangerous confusion: an officer must never read a failed fetch as
    // "this person has no history" — those are opposite inputs to a merge.
    authFetch.mockImplementation((url) => {
      if (String(url).includes('/candidate-journey')) {
        return Promise.resolve({
          ok: false, status: 500, json: () => Promise.resolve({ detail: 'boom' }),
        });
      }
      return jsonOk([ITEM]);
    });
    render(<ReviewQueue wsEvents={null} />);
    expect(await screen.findByText(/Could not load journey/i)).toBeTruthy();
    expect(screen.queryByText(/No prior sightings recorded/i)).toBeNull();
  });

  it('fetches the journey once per card, not on every render', async () => {
    authFetch.mockImplementation(routeFetch());
    const { rerender } = render(<ReviewQueue wsEvents={null} />);
    await waitFor(() => expect(journeyCalls()).toBe(1));

    // An unrelated re-render must not re-trigger the lazy journey fetch —
    // that would turn one fetch per card into one per render.
    rerender(<ReviewQueue wsEvents={{ type: 'new_alert' }} />);
    await new Promise((r) => setTimeout(r, 50));
    expect(journeyCalls()).toBe(1);
  });

  it('does not fetch journeys as part of the queue list request (no N+1)', async () => {
    authFetch.mockImplementation(routeFetch({ queue: [] }));
    render(<ReviewQueue wsEvents={null} />);
    await waitFor(() => expect(queueCalls()).toBe(1));
    await new Promise((r) => setTimeout(r, 50));
    // Empty queue → zero cards → zero journey fetches.
    expect(journeyCalls()).toBe(0);
  });
});
