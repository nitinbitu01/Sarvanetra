// frontend/src/components/ReviewQueue.jsx
// Review queue for borderline ReID decisions. Available to admin + officer.
// Shows side-by-side crops with similarity score, time gap, and approve/reject.
// Live updates via WebSocket reid_review_created events.

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { mockReviewQueue } from '../data/mockData';

const API = import.meta.env.VITE_API_URL || '/api/v1';

// Backend serves crops from a scoped static mount (backend/main.py:
// app.mount("/media/output", ...)), and stored paths are project-root-relative
// ("output/crops/track_5/frame_30.jpg"). The previous `/crops/${path}` built a
// URL relative to the VITE dev-server origin — wrong host AND wrong prefix, so
// every crop 404'd into the placeholder. Same mapping AlertFeed.jsx uses.
const MEDIA_BASE = API.replace(/\/api\/v1\/?$/, '');

function mediaUrl(path) {
  if (!path) return null;
  const clean = String(path).replace(/\\/g, '/').replace(/^\.?\//, '');
  if (clean.startsWith('output/')) {
    return `${MEDIA_BASE}/media/output/${clean.slice('output/'.length)}`;
  }
  return null;
}

const BAND_LABEL = (score) => {
  if (score >= 0.85) return { text: 'High', color: 'var(--accent-green)' };
  if (score >= 0.65) return { text: 'Uncertain', color: 'var(--accent-yellow)' };
  return { text: 'Low', color: 'var(--accent-red)' };
};

function formatGap(seconds) {
  if (!seconds) return '—';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

// Journey-so-far: the candidate identity's movement history, as context for
// this specific merge decision.
//
// Fetched lazily per card (on mount), NOT as part of GET /review-queue — that
// list endpoint should not carry an N+1 of one journey query per pending item.
//
// Hits …/review-queue/{id}/candidate-journey rather than
// /reid/journeys/{global_person_id}: the latter is the investigative lookup
// and returns 403 for OPERATOR unless the person already has an active alert,
// which review candidates generally don't. See reid_review.py's module
// docstring for why these are two endpoints and not one.
//
// Visual language (confidence badge thresholds, ⚠️ caveat) intentionally
// mirrors JourneyView.jsx rather than importing it — that component owns its
// own data fetching and page-level layout, and changing its signature to make
// it embeddable here would risk the standalone Journeys page it already serves.
function CandidateJourney({ itemId, globalPersonId }) {
  const { authFetch } = useAuth();
  const [state, setState] = useState({ status: 'loading', data: null, error: null });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await authFetch(`${API}/reid/review-queue/${itemId}/candidate-journey`);
        if (cancelled) return;
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          setState({ status: 'error', data: null, error: d.detail || `HTTP ${r.status}` });
          return;
        }
        setState({ status: 'ok', data: await r.json(), error: null });
      } catch {
        if (!cancelled) {
          setState({ status: 'error', data: null, error: 'Network error' });
        }
      }
    })();
    // Cancellation guard: an officer approving/rejecting quickly unmounts this
    // card mid-flight, and setState on an unmounted component is a warning at
    // best and a memory leak at worst.
    return () => { cancelled = true; };
  }, [authFetch, itemId]);

  const wrapStyle = {
    marginBottom: 14, padding: '10px 12px', borderRadius: 8,
    background: 'var(--bg-elevated)', border: '1px solid var(--border)',
  };
  const headerStyle = {
    fontSize: 11, color: 'var(--text-muted)', marginBottom: 8,
    display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
  };

  if (state.status === 'loading') {
    return (
      <div style={wrapStyle}>
        <div style={headerStyle}>Journey so far — Person #{globalPersonId}</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>⏳ Loading journey…</div>
      </div>
    );
  }

  if (state.status === 'error') {
    // Degrade visibly, never silently: an officer must not read "no journey
    // shown" as "this person has no prior sightings" when the truth is the
    // lookup failed. Those are very different inputs to a merge decision.
    return (
      <div style={{ ...wrapStyle, borderColor: 'var(--accent-yellow)55' }}>
        <div style={headerStyle}>Journey so far — Person #{globalPersonId}</div>
        <div style={{ fontSize: 12, color: 'var(--accent-yellow)' }}>
          ⚠️ Could not load journey ({state.error}). This is not the same as
          "no prior sightings" — decide on the crops alone, or retry.
        </div>
      </div>
    );
  }

  const entries = state.data?.journey || [];

  if (entries.length === 0) {
    return (
      <div style={wrapStyle}>
        <div style={headerStyle}>Journey so far — Person #{globalPersonId}</div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          🆕 No prior sightings recorded — this candidate identity has no journey yet.
        </div>
      </div>
    );
  }

  return (
    <div style={wrapStyle}>
      <div style={headerStyle}>
        <span>Journey so far — Person #{globalPersonId}</span>
        <span style={{ color: 'var(--text-secondary)' }}>
          {entries.length} sighting{entries.length === 1 ? '' : 's'}
          {state.data?.total_sightings != null && ` · ${state.data.total_sightings} total`}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'stretch', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
        {entries.map((e, i) => {
          const conf = e.confidence;
          const color = conf == null
            ? 'var(--text-muted)'
            : conf >= 0.85 ? 'var(--accent-green)'
              : conf >= 0.65 ? 'var(--accent-yellow)'
                : 'var(--accent-red)';
          return (
            <div key={e.journey_id ?? i} style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
              {i > 0 && <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>→</span>}
              <div style={{
                border: `1px solid ${color}55`, background: `${color}15`,
                borderRadius: 8, padding: '6px 10px', minWidth: 96,
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 5 }}>
                  📷 {e.camera_id || `Cam #${e.camera_db_id ?? '?'}`}
                  {e.confidence_caveat && (
                    <span title={e.confidence_caveat} style={{ cursor: 'help' }}>⚠️</span>
                  )}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                  {e.seen_at ? new Date(e.seen_at).toLocaleString() : '—'}
                </div>
                {conf != null && (
                  <div style={{ fontSize: 10, color, marginTop: 2, fontWeight: 600 }}>
                    {(conf * 100).toFixed(0)}% match
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {entries.some(e => e.confidence_caveat) && (
        <div style={{ fontSize: 10, color: 'var(--accent-yellow)', marginTop: 8, lineHeight: 1.5 }}>
          ⚠️ One or more sightings carry a confidence caveat (hover the icon).
          Those links were themselves appearance-only matches across a long
          gap — this journey is not independent corroboration of the match
          you are being asked to confirm.
        </div>
      )}
    </div>
  );
}

function ReviewCard({ item, onApprove, onReject, loading }) {
  const band = BAND_LABEL(item.similarity_score);
  const longGap = item.time_gap_seconds && item.time_gap_seconds > 21600;

  return (
    <div className="card" style={{ marginBottom: 16, position: 'relative' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>Review #{item.id}</span>
          <span style={{
            background: `${band.color}22`,
            color: band.color,
            border: `1px solid ${band.color}55`,
            borderRadius: 6,
            padding: '2px 10px',
            fontSize: 12,
            fontWeight: 600,
          }}>
            {band.text} — {(item.similarity_score * 100).toFixed(1)}%
          </span>
          {longGap && (
            <span title="Gap >6h: clothing may have changed" style={{
              background: 'var(--accent-yellow)22',
              color: 'var(--accent-yellow)',
              border: '1px solid var(--accent-yellow)55',
              borderRadius: 6,
              padding: '2px 8px',
              fontSize: 11,
            }}>
              ⚠️ {formatGap(item.time_gap_seconds)} gap
            </span>
          )}
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {item.created_at ? new Date(item.created_at).toLocaleString() : ''}
        </span>
      </div>

      {/* Crops side-by-side */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>New Sighting (Track #{item.local_track_id})</div>
          <CropImage path={item.crop_image_path} label="New sighting" />
        </div>
        <div style={{
          width: 2, background: 'var(--border)', borderRadius: 1,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          position: 'relative',
        }}>
          <span style={{
            position: 'absolute', background: 'var(--bg-elevated)',
            padding: '4px 6px', borderRadius: 6,
            fontSize: 11, color: 'var(--text-muted)',
          }}>vs</span>
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
            Candidate Person #{item.candidate_global_person_id}
          </div>
          <CropImage path={item.candidate_reference_image_path} label="Reference" />
        </div>
      </div>

      {/* Journey-so-far for the candidate identity (lazily fetched per card) */}
      <CandidateJourney
        itemId={item.id}
        globalPersonId={item.candidate_global_person_id}
      />

      {/* Meta row */}
      <div style={{ display: 'flex', gap: 20, marginBottom: 14, fontSize: 12, color: 'var(--text-secondary)' }}>
        <span>Similarity: <strong style={{ color: band.color }}>{(item.similarity_score * 100).toFixed(1)}%</strong></span>
        <span>Time gap: <strong>{formatGap(item.time_gap_seconds)}</strong></span>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 10 }}>
        <button
          id={`approve-${item.id}`}
          disabled={loading === item.id}
          onClick={() => onApprove(item.id)}
          style={{
            flex: 1, padding: '9px 0', borderRadius: 8, border: 'none',
            background: loading === item.id ? 'var(--bg-elevated)' : 'var(--accent-green)',
            color: '#fff', fontWeight: 700, fontSize: 13, cursor: loading === item.id ? 'not-allowed' : 'pointer',
            transition: 'opacity 0.2s',
          }}
        >
          {loading === item.id ? '⏳ Processing...' : '✅ Same Person — Merge'}
        </button>
        <button
          id={`reject-${item.id}`}
          disabled={loading === item.id}
          onClick={() => onReject(item.id)}
          style={{
            flex: 1, padding: '9px 0', borderRadius: 8, border: 'none',
            background: loading === item.id ? 'var(--bg-elevated)' : 'var(--accent-red)',
            color: '#fff', fontWeight: 700, fontSize: 13, cursor: loading === item.id ? 'not-allowed' : 'pointer',
            transition: 'opacity 0.2s',
          }}
        >
          {loading === item.id ? '⏳ Processing...' : '❌ Different Person — New ID'}
        </button>
      </div>
    </div>
  );
}

function CropImage({ path, label }) {
  const [broken, setBroken] = useState(false);
  if (!path || broken) {
    return (
      <div style={{
        height: 200, borderRadius: 8, background: 'var(--bg-elevated)',
        border: '1px dashed var(--border-bright)',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', color: 'var(--text-muted)', gap: 8,
      }}>
        <span style={{ fontSize: 28 }}>👤</span>
        <span style={{ fontSize: 11 }}>{label}</span>
        {path && <span style={{ fontSize: 10 }}>Image unavailable</span>}
      </div>
    );
  }
  const src = mediaUrl(path);
  if (!src) {
    return (
      <div style={{
        height: 200, borderRadius: 8, background: 'var(--bg-elevated)',
        border: '1px dashed var(--border-bright)',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', color: 'var(--text-muted)', gap: 8,
      }}>
        <span style={{ fontSize: 28 }}>👤</span>
        <span style={{ fontSize: 11 }}>{label}</span>
        <span style={{ fontSize: 10 }}>Path outside served media root</span>
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={label}
      onError={() => setBroken(true)}
      style={{ width: '100%', height: 200, objectFit: 'cover', borderRadius: 8, border: '1px solid var(--border)' }}
    />
  );
}

export default function ReviewQueue({ wsEvents }) {
  const { authFetch } = useAuth();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(null);
  const [error, setError] = useState(null);
  const [fetching, setFetching] = useState(false);

  const loadQueue = useCallback(async () => {
    setFetching(true);
    try {
      const r = await authFetch(`${API}/reid/review-queue`);
      if (r.ok) {
        const data = await r.json();
        setItems(Array.isArray(data) ? data : (Array.isArray(data?.items) ? data.items : mockReviewQueue));
        setError(null);
      } else {
        setItems(mockReviewQueue);
        setError(null);
      }
    } catch (e) {
      setItems(mockReviewQueue);
      setError(null);
    } finally {
      setFetching(false);
    }
  }, [authFetch]);

  useEffect(() => { loadQueue(); }, [loadQueue]);

  // Live updates via WS
  useEffect(() => {
    if (wsEvents?.type === 'reid_review_created') {
      loadQueue();
    }
  }, [wsEvents, loadQueue]);

  const handleAction = async (itemId, action) => {
    setLoading(itemId);
    setError(null);
    try {
      const r = await authFetch(`${API}/reid/review-queue/${itemId}/${action}`, {
        method: 'POST',
      });
      if (r.ok) {
        setItems(prev => prev.filter(i => i.id !== itemId));
      } else {
        const d = await r.json().catch(() => ({}));
        setError(d.detail || 'Action failed.');
      }
    } catch {
      setError('Network error.');
    } finally {
      setLoading(null);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div className="card-title">Person ReID Review Queue</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            Borderline similarity matches (65–85%) awaiting officer verification
          </div>
        </div>
        <button
          onClick={loadQueue}
          disabled={fetching}
          style={{
            padding: '7px 14px', borderRadius: 8, border: '1px solid var(--border)',
            background: 'var(--bg-elevated)', color: 'var(--text-primary)',
            cursor: 'pointer', fontSize: 12,
          }}
        >
          {fetching ? '⏳' : '🔄 Refresh'}
        </button>
      </div>

      {error && (
        <div style={{
          padding: '10px 14px', borderRadius: 8, background: 'var(--accent-red)15',
          border: '1px solid var(--accent-red)44', color: 'var(--accent-red)',
          marginBottom: 16, fontSize: 13,
        }}>
          ⚠️ {error}
        </div>
      )}

      {(!Array.isArray(items) || items.length === 0) && !fetching ? (
        <div className="empty-state">
          <span className="empty-state-icon">✅</span>
          <div>No pending review items</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            All borderline matches have been resolved
          </div>
        </div>
      ) : (
        <div>
          {(Array.isArray(items) ? items : []).map(item => (
            <ReviewCard
              key={item.id}
              item={item}
              loading={loading}
              onApprove={(id) => handleAction(id, 'approve')}
              onReject={(id) => handleAction(id, 'reject')}
            />
          ))}
        </div>
      )}
    </div>
  );
}
