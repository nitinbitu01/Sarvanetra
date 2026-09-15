// frontend/src/components/AlertFeed.jsx
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';
import RoutingBadge from './RoutingBadge';
import ProofClipModal from './ProofClipModal';

const TYPE_ICONS = {
  PERSON_MATCH:        { icon: '👤', cls: 'badge-person' },
  VEHICLE_MATCH:       { icon: '🚗', cls: 'badge-vehicle' },
  UNCERTAIN_PLATE:     { icon: '⚠️', cls: 'badge-plate' },
  WATCHLIST_FACE_MATCH:{ icon: '🚨', cls: 'badge-critical' },
  // Previously absent — fell back to a generic 🔔 with no distinguishing
  // badge class, the least visually important treatment in the feed, for
  // what this whole platform is actually evaluated on: a live watchlist hit
  // on a vehicle plate.
  STOLEN_VEHICLE_WATCHLIST_HIT: { icon: '🚔', cls: 'badge-critical' },
  // One registration read at two places no vehicle could cover in the time.
  PLATE_CLONE_SUSPECTED: { icon: '👯', cls: 'badge-critical' },
  LOITERING:           { icon: '🧍', cls: 'badge-loitering' },
  CROWD_ANOMALY:       { icon: '👥', cls: 'badge-crowd' },
  ABANDONED_OBJECT:    { icon: '🎒', cls: 'badge-crowd' },
};

// Alert types that get the acknowledge/dismiss/escalate controls. Dismissing
// is what removes an alert's contribution from the SENTINEL IQ aggregate, so
// every scored non-watchlist type needs the control — ABANDONED_OBJECT was
// missing it, which left its score unclearable from the feed.
const LIFECYCLE_TYPES = new Set(['LOITERING', 'CROWD_ANOMALY', 'ABANDONED_OBJECT']);

// Upper bound on rows held in the DOM. Rows beyond this are dropped from the
// live tail only — the alerts themselves stay in the database.
const MAX_FEED_ALERTS = 200;

// Backend serves evidence/reference images from two scoped static mounts
// (see backend/main.py) — never the whole project root. Any path outside
// those two prefixes is intentionally not resolved to a URL.
const MEDIA_BASE = API.replace(/\/api\/v1\/?$/, '');

function mediaUrl(path) {
  if (!path) return null;
  const clean = String(path).replace(/\\/g, '/').replace(/^\.?\//, '');
  if (clean.startsWith('output/')) return `${MEDIA_BASE}/media/output/${clean.slice('output/'.length)}`;
  if (clean.startsWith('data/prep/')) return `${MEDIA_BASE}/media/watchlist-refs/${clean.slice('data/prep/'.length)}`;
  return null;
}

function EvidenceBtn({ alertId }) {
  const { authFetch } = useAuth();
  const [state, setState] = useState('idle'); // idle | checking | intact | tampered
  const verify = async () => {
    setState('checking');
    try {
      const res = await authFetch(`${API}/alerts/${alertId}/verify`);
      const data = await res.json();
      setState(data.intact === true ? 'intact' : data.intact === false ? 'tampered' : 'idle');
    } catch { setState('idle'); }
  };
  const titles = { idle: 'Verify evidence integrity', checking: 'Checking…', intact: '✅ Evidence intact', tampered: '🚨 Evidence tampered!' };
  const emojis = { idle: '🔒', checking: '⏳', intact: '✅', tampered: '🚨' };
  return (
    <button className={`evidence-btn ${state === 'intact' ? 'intact' : state === 'tampered' ? 'tampered' : ''}`}
      onClick={verify} disabled={state === 'checking'} title={titles[state]}>
      {emojis[state]}
    </button>
  );
}

function CompareImage({ src, label }) {
  const [broken, setBroken] = useState(false);
  if (!src || broken) {
    return (
      <div className="compare-img-placeholder">
        <span style={{ fontSize: 22 }}>👤</span>
        <span style={{ fontSize: 10 }}>{label}</span>
      </div>
    );
  }
  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <img src={src} alt={label} onError={() => setBroken(true)} className="compare-img" />
    </div>
  );
}

function FalsePositiveControl({ alert, onMarked }) {
  const { authFetch } = useAuth();
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  if (alert.status === 'false_positive') {
    return (
      <div className="fp-badge" title={alert.false_positive_reason || ''}>
        ❌ Marked false positive
      </div>
    );
  }

  if (!open) {
    return (
      <button className="btn btn-secondary" style={{ fontSize: 11, padding: '5px 10px' }}
        onClick={() => setOpen(true)}>
        Mark False Positive
      </button>
    );
  }

  const submit = async () => {
    if (!reason.trim()) { setError('Reason is required.'); return; }
    setSubmitting(true);
    setError(null);
    try {
      const res = await authFetch(`${API}/alerts/${alert.id}/mark-false-positive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: reason.trim() }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || 'Failed to mark false positive.');
        return;
      }
      const data = await res.json();
      onMarked(alert.id, data.false_positive_reason);
      setOpen(false);
    } catch {
      setError('Network error.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fp-form">
      <input
        className="form-input"
        style={{ fontSize: 12, padding: '6px 10px' }}
        placeholder="Reason (required)…"
        value={reason}
        onChange={e => setReason(e.target.value)}
        disabled={submitting}
      />
      <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
        <button className="btn btn-danger" style={{ fontSize: 11, padding: '5px 10px' }}
          onClick={submit} disabled={submitting}>
          {submitting ? 'Submitting…' : 'Confirm'}
        </button>
        <button className="btn btn-secondary" style={{ fontSize: 11, padding: '5px 10px' }}
          onClick={() => { setOpen(false); setError(null); }} disabled={submitting}>
          Cancel
        </button>
      </div>
      {error && <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 4 }}>{error}</div>}
    </div>
  );
}

function LifecycleControl({ alert, onLifecycleChanged }) {
  const { authFetch } = useAuth();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const status = alert.lifecycle_status || 'OPEN';
  const badgeCls = {
    OPEN: 'badge-lifecycle-open', ACKNOWLEDGED: 'badge-lifecycle-acknowledged',
    DISMISSED: 'badge-lifecycle-dismissed', ESCALATED: 'badge-lifecycle-escalated',
  }[status] || 'badge-lifecycle-open';

  const act = async (action, body) => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await authFetch(`${API}/alerts/${alert.id}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || `Failed to ${action}.`);
        return;
      }
      const data = await res.json();
      onLifecycleChanged(alert.id, data);
    } catch {
      setError('Network error.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="lifecycle-actions">
        <span className={`badge ${badgeCls}`}>{status}</span>
        {status !== 'ACKNOWLEDGED' && status !== 'DISMISSED' && (
          <button className="btn btn-secondary" disabled={submitting}
            onClick={() => act('acknowledge')}>Acknowledge</button>
        )}
        {status !== 'DISMISSED' && (
          <button className="btn btn-secondary" disabled={submitting}
            onClick={() => act('dismiss', { feedback: 'FALSE_POSITIVE' })}>Dismiss (FP)</button>
        )}
        {status !== 'DISMISSED' && (
          <button className="btn btn-secondary" disabled={submitting}
            onClick={() => act('dismiss', { feedback: 'TRUE_POSITIVE' })}>Dismiss (TP)</button>
        )}
        {status !== 'ESCALATED' && status !== 'DISMISSED' && (
          <button className="btn btn-danger" disabled={submitting}
            onClick={() => act('escalate')}>Escalate</button>
        )}
      </div>
      {error && <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 4 }}>{error}</div>}
    </div>
  );
}

// Day 10: the alert's own SENTINEL IQ contribution, as a badge.
//
// Renders the STORED value only. A null contribution means the alert type has
// no base score (nothing in this system fires it) or the alert predates the
// scoring engine — either way the badge is omitted entirely rather than shown
// as 0, because a 0 next to an alert reads as "scored, and it came out
// harmless". The tooltip carries the same formula trace the score card shows.
function IQBadge({ alert }) {
  if (alert.iq_contribution == null) return null;

  const b = alert.iq_breakdown;
  let title = `Sarvanetra IQ contribution: ${alert.iq_contribution}`;
  if (b) {
    const applied = (b.multipliers || []).filter(m => m.applied);
    const parts = [`${b.alert_type} ${b.base_score}`, ...applied.map(m => `${m.name} ${m.factor}`)];
    title = `${parts.join(' × ')} = ${b.total}`;
    const night = (b.multipliers || []).find(m => m.name === 'Night');
    if (night?.detail?.night_source) title += `\nnight_source: ${night.detail.night_source}`;
  }

  const score = alert.iq_contribution;
  const cls = score >= 7 ? 'badge-high' : score >= 4 ? 'badge-medium' : 'badge-low';

  return (
    <span className={`badge ${cls}`} title={title}>
      🧠 {Number(score).toFixed(2).replace(/\.?0+$/, '')}
    </span>
  );
}

// Surfaces the DRAFT human-in-the-loop procedure at the moment of decision.
//
// This is an affordance, NOT enforcement, and it does not close the open item
// it comes from — see docs/PRODUCTION_READINESS_OPEN_ITEMS.md item 4. Nothing
// here prevents an officer acting before comparing, and nothing records
// whether they did. What it does do is put the constraint in front of them
// instead of in a document they have never read.
function WatchlistUseNotice({ alert }) {
  const [open, setOpen] = useState(false);
  const isStub = alert.metadata?.is_stub === true;

  return (
    <div style={{
      margin: '0 14px 10px', padding: '8px 10px',
      border: '1px solid var(--accent-red)', borderRadius: 'var(--radius-md)',
      background: 'rgba(255,0,0,0.05)',
    }}>
      {isStub && (
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent-red)', marginBottom: 6 }}>
          ⛔ Embedder is in STUB MODE — this match is meaningless. Do not act on it.
        </div>
      )}
      <div style={{ fontSize: 11, fontWeight: 600 }}>
        ⚠️ Unconfirmed match — compare the two images above before acting.
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>
        The percentage is a similarity score at an unvalidated threshold, not a
        probability that this is the same person.
      </div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          background: 'none', border: 'none', padding: '4px 0 0',
          color: 'var(--text-muted)', fontSize: 10, cursor: 'pointer',
          textDecoration: 'underline',
        }}>
        {open ? 'Hide procedure' : 'What am I required to do?'}
      </button>
      {open && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, lineHeight: 1.6 }}>
          <strong>MUST:</strong> visually compare live capture against the reference
          photo before any action; treat the match as unconfirmed until you have;
          treat an inconclusive comparison (poor angle, low resolution, occlusion,
          blur) as <em>not a match</em>; record the outcome before closing.
          <br />
          <strong>MUST NOT:</strong> detain, stop, search, or approach anyone on the
          basis of this alert alone; read the confidence figure as a probability of
          identity.
          <div style={{ marginTop: 6, fontStyle: 'italic' }}>
            Draft procedure, not yet approved or mandated — see
            docs/PRODUCTION_READINESS_OPEN_ITEMS.md item 4.
          </div>
        </div>
      )}
    </div>
  );
}

function ZoneIncidentCard({ incident }) {
  const [expanded, setExpanded] = useState(false);
  const openedTs = incident.opened_at ? new Date(incident.opened_at).toLocaleTimeString() : '';

  return (
    <div className="zone-incident-card">
      <div className="zone-incident-header" onClick={() => setExpanded(!expanded)}>
        <div className="zone-incident-title">
          <span style={{ fontSize: 20 }}>🚨</span>
          <div>
            <div>ZONE INCIDENT — {incident.zone}</div>
            <div className="zone-incident-meta">
              <span>⏱️ Opened at {openedTs}</span>
              <span>📍 {incident.center_lat.toFixed(4)}°N, {incident.center_lon.toFixed(4)}°E</span>
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="zone-incident-badge">{incident.alert_count} ALERTS CLUSTERED</span>
          <button className="btn btn-secondary" style={{ fontSize: 11, padding: '4px 8px' }}>
            {expanded ? '▲ Collapse' : '▼ View Alerts'}
          </button>
        </div>
      </div>

      {expanded && incident.alerts && incident.alerts.length > 0 && (
        <div className="zone-incident-alerts-list">
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Constituent Alerts (Same 100m Radius / 2-min Window):
          </div>
          {incident.alerts.map(a => {
            const { icon } = TYPE_ICONS[a.alert_type] || { icon: '🔔' };
            const aTs = a.created_at ? new Date(a.created_at).toLocaleTimeString() : '';
            return (
              <div key={a.id} className="zone-constituent-row">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>{icon}</span>
                  <span style={{ fontWeight: 600 }}>{a.alert_type.replace(/_/g, ' ')}</span>
                  <span style={{ color: 'var(--text-muted)' }}>— {a.subject_label}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--text-muted)' }}>
                  <span>📷 {a.camera_name || 'Camera'}</span>
                  <span>⏱️ {aTs}</span>
                  {a.confidence != null && <span>{(a.confidence * 100).toFixed(0)}%</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function AlertRow({ alert, isNew, onMarked, onLifecycleChanged, routing, onRoutingChange, onViewProof }) {
  const { authFetch } = useAuth();
  const { icon, cls } = TYPE_ICONS[alert.alert_type] || { icon: '🔔', cls: '' };
  const ts = alert.created_at ? new Date(alert.created_at).toLocaleTimeString() : '';
  const isWatchlistMatch = alert.alert_type === 'WATCHLIST_FACE_MATCH'
    || alert.alert_type === 'STOLEN_VEHICLE_WATCHLIST_HIT';
  const isBehaviorAlert = LIFECYCLE_TYPES.has(alert.alert_type);
  const isUncalibrated = alert.alert_type === 'LOITERING' && alert.metadata?.calibration_method === 'uncalibrated';
  const refPhoto = mediaUrl(alert.metadata?.reference_photo_path);
  const livePhoto = mediaUrl(alert.snapshot_path);

  // Day 11: Expandable merged alerts
  const [expandedMerged, setExpandedMerged] = useState(false);
  const [mergedList, setMergedList] = useState([]);
  const [loadingMerged, setLoadingMerged] = useState(false);

  const toggleMerged = async () => {
    if (!expandedMerged && mergedList.length === 0) {
      setLoadingMerged(true);
      try {
        const res = await authFetch(`${API}/alerts/${alert.id}/merged`);
        if (res.ok) setMergedList(await res.json());
      } catch (err) {
        console.error('Failed to load merged alerts', err);
      } finally {
        setLoadingMerged(false);
      }
    }
    setExpandedMerged(!expandedMerged);
  };

  // Day 14: a red pulsing border marks an alert that timed out without an
  // ACK. Driven by routing status, not by a transient WS flag, so it
  // survives a reload — an escalated alert is still escalated after F5.
  const escalatedClass = routing
    && (routing.status === 'ESCALATED_ROUTED' || routing.status === 'ESCALATED_UNROUTED')
    ? ' escalated' : '';

  return (
    <div className={`alert-row-wrap ${isWatchlistMatch ? 'critical' : ''}${escalatedClass}`}>
      <div className={`alert-row ${isNew ? 'new' : ''}`}>
        <span className="alert-type-icon">{icon}</span>
        <div className="alert-body">
          <div className="alert-subject">
            <span className={`badge ${cls}`} style={{ marginRight: 6 }}>{alert.alert_type.replace(/_/g, ' ')}</span>
            {/* A staged or rehearsal alert must never be mistaken for a real
                detection. The API exposes is_simulated; showing it is what
                makes that flag mean anything to whoever is looking. */}
            {alert.is_simulated && (
              <span
                className="badge"
                style={{ marginRight: 6, background: '#78350f', color: '#fde68a', border: '1px solid #b45309' }}
                title="Staged for demonstration — not a real detection. Raised deliberately so the detector can be shown when no live case is occurring."
              >
                ⚑ STAGED
              </span>
            )}
            {alert.subject_label || 'Unknown subject'}
            {isUncalibrated && (
              <span className="uncalibrated-tag" title="No camera_calibration row for this camera — using the default px-per-meter fallback, not a trustworthy per-camera measurement.">
                ⚠ uncalibrated
              </span>
            )}
            {alert.merged_count > 0 && (
              <button
                className="badge-similar"
                style={{ marginLeft: 8 }}
                onClick={toggleMerged}
                title="Click to view all merged duplicate alerts for this event"
              >
                +{alert.merged_count} similar {expandedMerged ? '▲' : '▼'}
              </button>
            )}
          </div>
          <div className="alert-meta">
            📷 {alert.camera_name || 'Unknown camera'}{alert.camera_zone ? ` · ${alert.camera_zone}` : ''} · {ts}
          </div>
          {alert.description && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4, lineHeight: 1.4, maxWidth: 650 }}>
              {alert.description.length > 120 ? `${alert.description.slice(0, 120)}…` : alert.description}
            </div>
          )}
          {/* Day 14: dispatch state. Rendered only when a routing record
              exists — non-CRITICAL alerts never enter the state machine, so
              they get no badge rather than an empty one. */}
          {routing && (
            <div style={{ marginTop: 7 }}>
              <RoutingBadge
                alertId={alert.id}
                routing={routing}
                onRoutingChange={onRoutingChange}
              />
            </div>
          )}
        </div>
        <div className="alert-actions">
          {alert.confidence != null && (
            <span className="alert-confidence">{(alert.confidence * 100).toFixed(0)}%</span>
          )}
          {alert.danger_score != null && (
            /* Thresholds were 70 and 40, on a field nothing writes above 10:
               the highest danger_score in this database is 9.8. Every alert
               the system has ever raised therefore rendered with the LOW
               badge, stolen-vehicle hits included. Scaled to the 0-10 the
               producers actually use. */
            <span className={`badge ${alert.danger_score >= 7 ? 'badge-high' : alert.danger_score >= 4 ? 'badge-medium' : 'badge-low'}`}
              title="Severity 0-10, as written by the alert producer. Separate from the Sarvanetra IQ contribution beside it.">
              ⚡ {Number(alert.danger_score).toFixed(1)}
            </span>
          )}
          <IQBadge alert={alert} />
          <button
            className="proof-badge-btn"
            onClick={() => onViewProof && onViewProof(alert)}
            title="Click to view authentic CCTV video proof clip"
            style={{
              background: 'rgba(59, 130, 246, 0.18)',
              color: '#60a5fa',
              border: '1px solid rgba(59, 130, 246, 0.4)',
              borderRadius: 6,
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <span>🎬</span> View Proof
          </button>
          <EvidenceBtn alertId={alert.id} />
        </div>
      </div>

      {/* Day 11: Expanded duplicate merged detections */}
      {expandedMerged && (
        <div className="merged-alerts-panel">
          <div className="merged-alert-header">
            Merged Duplicate Detections (Same Global ID & Alert Type — Display Grouping Only):
          </div>
          {loadingMerged ? (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>⏳ Loading merged records…</div>
          ) : mergedList.length === 0 ? (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No additional records found.</div>
          ) : (
            mergedList.map(m => {
              const mTs = m.created_at ? new Date(m.created_at).toLocaleTimeString() : '';
              return (
                <div key={m.id} className="merged-alert-item">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: 'var(--text-muted)' }}>↳ Alert #{m.id}</span>
                    <span>📷 {m.camera_name || 'Camera'}{m.camera_zone ? ` (${m.camera_zone})` : ''}</span>
                    {m.confidence != null && (
                      <span className="alert-confidence">Conf: {(m.confidence * 100).toFixed(0)}%</span>
                    )}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{mTs}</span>
                    <EvidenceBtn alertId={m.id} />
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Auto-fired CRITICAL alert — let an officer visually confirm at a glance,
          even though the system doesn't wait for that confirmation before alerting. */}
      {isWatchlistMatch && (
        <div className="watchlist-compare">
          <div className="compare-images">
            <CompareImage src={refPhoto} label="Watchlist reference" />
            <div className="compare-vs">vs</div>
            <CompareImage src={livePhoto} label="Live capture" />
          </div>
          <WatchlistUseNotice alert={alert} />
          <div className="watchlist-compare-footer">
            <FalsePositiveControl alert={alert} onMarked={onMarked} />
          </div>
        </div>
      )}

      {/* Day 8: general alert lifecycle — acknowledge/dismiss/escalate,
          live-synced across every connected operator's tab. */}
      {isBehaviorAlert && (
        <div className="watchlist-compare-footer" style={{ padding: '0 14px 10px' }}>
          <LifecycleControl alert={alert} onLifecycleChanged={onLifecycleChanged} />
        </div>
      )}
    </div>
  );
}

export default function AlertFeed({ liveAlert, statusEvent, mergedEvent, zoneIncidentEvent,
                                    routingEvent }) {
  const { authFetch } = useAuth();
  const [alerts, setAlerts] = useState([]);
  const [zoneIncidents, setZoneIncidents] = useState([]);
  const [newIds, setNewIds] = useState(new Set());
  const [loading, setLoading] = useState(true);
  // alert_id → routing record. Seeded from the server so a reload resumes
  // countdowns at the right position instead of restarting them.
  const [routingByAlert, setRoutingByAlert] = useState({});

  const fetchRouting = useCallback(async () => {
    try {
      const [activeRes, unroutedRes] = await Promise.all([
        authFetch(`${API}/routing/active`),
        authFetch(`${API}/routing/unrouted`),
      ]);
      const next = {};
      // Both lists are needed: /routing/active excludes terminal states, so
      // an UNROUTED alert would otherwise render with no badge at all —
      // indistinguishable from a non-CRITICAL alert that was never routed.
      for (const res of [activeRes, unroutedRes]) {
        if (res.ok) {
          for (const r of await res.json()) next[r.alert_id] = r;
        }
      }
      setRoutingByAlert(prev => ({ ...prev, ...next }));
    } catch {
      /* routing badges are additive — a failed poll must not blank the feed */
    }
  }, [authFetch]);

  const fetchAlerts = useCallback(async () => {
    try {
      const [alertsRes, incidentsRes] = await Promise.all([
        authFetch(`${API}/alerts?limit=50`).catch(() => null),
        authFetch(`${API}/zone-incidents?expand_alerts=true`).catch(() => null),
      ]);
      if (alertsRes && alertsRes.ok) {
        const aData = await alertsRes.json();
        setAlerts(Array.isArray(aData) ? aData.slice(0, MAX_FEED_ALERTS) : []);
      }
      if (incidentsRes && incidentsRes.ok) {
        const iData = await incidentsRes.json();
        setZoneIncidents(Array.isArray(iData) ? iData : []);
      }
    } catch {
      // transient error, will retry
    } finally {
      setLoading(false);
    }
  }, [authFetch]);

  useEffect(() => {
    fetchAlerts();
    fetchRouting();
    const interval = setInterval(() => {
      fetchAlerts();
      fetchRouting();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchAlerts, fetchRouting]);

  // Day 14: routing events. alert.routed / .escalated / .unrouted change
  // which officer owns an alert and restart the clock, so they re-seed from
  // the server rather than being patched locally — the server's
  // seconds_elapsed is the only trustworthy origin for the countdown.
  // alert.ack is patched inline: it is terminal, needs no timer, and the
  // badge should turn green instantly rather than after a round-trip.
  useEffect(() => {
    if (!routingEvent) return;
    if (routingEvent.type === 'alert.ack') {
      setRoutingByAlert(prev => {
        const cur = prev[routingEvent.alert_id];
        if (!cur) return prev;
        return {
          ...prev,
          [routingEvent.alert_id]: {
            ...cur, status: 'ACKNOWLEDGED', ack_at: routingEvent.ack_at,
          },
        };
      });
    } else if (routingEvent.type !== 'officer.status') {
      fetchRouting();
    }
  }, [routingEvent, fetchRouting]);

  const handleRoutingChange = useCallback((updated) => {
    if (!updated?.alert_id) return;
    setRoutingByAlert(prev => ({ ...prev, [updated.alert_id]: updated }));
  }, []);

  // Prepend live alert from WebSocket (unless it was merged into an existing primary)
  useEffect(() => {
    if (!liveAlert) return;
    if (liveAlert.merged_into_alert_id) {
      setAlerts(prev => prev.map(a =>
        a.id === liveAlert.merged_into_alert_id
          ? { ...a, merged_count: (a.merged_count || 0) + 1 }
          : a
      ));
      return;
    }
    // Cap the in-memory feed. Without this, a control room left open for a
    // shift accumulates one row per alert forever — at 30s intervals that is
    // ~1000 AlertRows after eight hours, each with its own badges, timers and
    // expandable sub-panels. Dropped rows remain in the database and are
    // still reachable via Refresh; only the live tail is bounded.
    setAlerts(prev => [liveAlert, ...prev].slice(0, MAX_FEED_ALERTS));
    setNewIds(ids => new Set([...ids, liveAlert.id]));
    setTimeout(() => setNewIds(ids => { const n = new Set(ids); n.delete(liveAlert.id); return n; }), 2000);
  }, [liveAlert]);

  // Day 8: live lifecycle sync
  useEffect(() => {
    if (!statusEvent) return;
    setAlerts(prev => prev.map(a =>
      a.id === statusEvent.alert_id
        ? { ...a, lifecycle_status: statusEvent.lifecycle_status, feedback: statusEvent.feedback }
        : a
    ));
  }, [statusEvent]);

  // Day 11: Alert merged live event
  useEffect(() => {
    if (!mergedEvent) return;
    setAlerts(prev => {
      const updated = prev.filter(a => a.id !== mergedEvent.merged_alert_id);
      return updated.map(a =>
        a.id === mergedEvent.primary_alert_id
          ? { ...a, merged_count: (a.merged_count || 0) + 1 }
          : a
      );
    });
  }, [mergedEvent]);

  // Day 11: Zone incident live event (opened or updated)
  useEffect(() => {
    if (!zoneIncidentEvent) return;
    setZoneIncidents(prev => {
      const exists = prev.some(inc => inc.id === zoneIncidentEvent.incident_id);
      if (exists) {
        return prev.map(inc =>
          inc.id === zoneIncidentEvent.incident_id
            ? {
                ...inc,
                alert_count: (zoneIncidentEvent.alert_ids || []).length,
                alert_ids: zoneIncidentEvent.alert_ids,
              }
            : inc
        );
      } else {
        return [
          {
            id: zoneIncidentEvent.incident_id,
            zone: zoneIncidentEvent.zone,
            center_lat: zoneIncidentEvent.center_lat,
            center_lon: zoneIncidentEvent.center_lon,
            opened_at: zoneIncidentEvent.opened_at,
            alert_count: (zoneIncidentEvent.alert_ids || []).length,
            alert_ids: zoneIncidentEvent.alert_ids,
          },
          ...prev,
        ];
      }
    });
  }, [zoneIncidentEvent]);

  const handleMarked = useCallback((alertId, reason) => {
    setAlerts(prev => prev.map(a =>
      a.id === alertId ? { ...a, status: 'false_positive', false_positive_reason: reason } : a
    ));
  }, []);

  const handleLifecycleChanged = useCallback((alertId, data) => {
    setAlerts(prev => prev.map(a =>
      a.id === alertId ? { ...a, lifecycle_status: data.lifecycle_status, feedback: data.feedback } : a
    ));
  }, []);

  const [proofAlert, setProofAlert] = useState(null);

  if (loading) return <div className="empty-state"><span className="empty-state-icon">⏳</span>Loading alerts…</div>;

  return (
    <div>
      {/* Day 11: Active Zone Incidents Section */}
      {zoneIncidents.length > 0 && (
        <div className="zone-incidents-container">
          <div className="card-title" style={{ color: 'var(--accent-red)' }}>
            🚨 Active Zone Incidents ({zoneIncidents.length})
          </div>
          {zoneIncidents.map(inc => (
            <ZoneIncidentCard key={inc.id} incident={inc} />
          ))}
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="card-title">🔔 Alert Feed <span style={{ color: 'var(--text-muted)', fontWeight: 400, fontSize: 12 }}>({alerts.length})</span></div>
        <button className="btn btn-secondary" onClick={fetchAlerts} style={{ fontSize: 12, padding: '5px 10px' }}>↻ Refresh</button>
      </div>
      {alerts.length === 0 ? (
        <div className="empty-state"><span className="empty-state-icon">🔕</span>No alerts yet</div>
      ) : (
        <div className="alert-list">
          {alerts.map(a => (
            <AlertRow key={a.id} alert={a} isNew={newIds.has(a.id)}
              onMarked={handleMarked} onLifecycleChanged={handleLifecycleChanged}
              routing={routingByAlert[a.id]} onRoutingChange={handleRoutingChange}
              onViewProof={setProofAlert} />
          ))}
        </div>
      )}

      {/* Proof Video Modal */}
      {proofAlert && (
        <ProofClipModal alert={proofAlert} onClose={() => setProofAlert(null)} />
      )}
    </div>
  );
}

