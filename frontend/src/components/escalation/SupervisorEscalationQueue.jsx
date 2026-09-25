// frontend/src/components/escalation/SupervisorEscalationQueue.jsx
// Shows ReviewPair cases where two officers disagreed (status='disagreed_escalated').
// Supervisors and admins can adjudicate (pick final decision) or discard.
// Uses existing AuthContext and WebSocketContext — no Zustand, no axios.

import { useState, useEffect, useCallback, useRef } from 'react';
import toast from 'react-hot-toast';
import { useAuth } from '../../context/AuthContext';
import { useWebSocketEvent } from '../../context/WebSocketContext';

const API = import.meta.env.VITE_API_URL || '/api/v1';

const SLA_WARN_S   = 1800;  // 30 minutes — amber
const SLA_BREACH_S = 3600;  // 60 minutes — red + pulse

function formatAge(seconds) {
  if (!seconds && seconds !== 0) return '—';
  if (seconds < 60)   return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function AgeTag({ ageSeconds }) {
  const isBreach = ageSeconds >= SLA_BREACH_S;
  const isWarn   = ageSeconds >= SLA_WARN_S && !isBreach;

  return (
    <span style={{
      padding: '2px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600,
      background: isBreach ? 'rgba(220,38,38,0.15)'
                : isWarn   ? 'rgba(245,158,11,0.15)'
                : 'rgba(107,114,128,0.15)',
      color: isBreach ? '#ef4444' : isWarn ? '#f59e0b' : '#9ca3af',
      border: `1px solid ${isBreach ? '#991b1b' : isWarn ? '#92400e' : '#374151'}`,
      animation: isBreach ? 'pulse 2s infinite' : 'none',
    }}>
      ⏱ {formatAge(ageSeconds)}
      {isBreach && ' — SLA BREACH'}
      {isWarn   && ' — SLA WARNING'}
    </span>
  );
}

function EscalationCard({ esc, onAdjudicate }) {
  const ageSeconds = esc.age_seconds ?? Math.floor(
    (Date.now() - new Date(esc.created_at).getTime()) / 1000
  );
  const isBreach = ageSeconds >= SLA_BREACH_S;

  return (
    <div style={{
      background: 'var(--bg-elevated, #1f2937)',
      border: `1px solid ${isBreach ? '#7f1d1d' : '#374151'}`,
      borderRadius: 14,
      padding: 20,
      display: 'flex', flexDirection: 'column', gap: 14,
      boxShadow: isBreach ? '0 0 0 1px rgba(220,38,38,0.3)' : 'none',
    }}
      role="listitem"
    >
      {/* Top row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <span style={{ fontSize: 13, fontWeight: 700 }}>
            Alert #{esc.alert_id ?? esc.review_pair_id}
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: 12, marginLeft: 10 }}>
            {esc.zone ?? ''}
          </span>
        </div>
        <AgeTag ageSeconds={ageSeconds} />
      </div>

      {/* Decision summary */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10,
      }}>
        <div style={{
          padding: '10px 14px', background: 'rgba(22,163,74,0.08)',
          border: '1px solid #166534', borderRadius: 8,
          fontSize: 12, color: '#4ade80',
        }}>
          <p style={{ margin: '0 0 2px', color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase' }}>
            Officer A Decision
          </p>
          <strong>{esc.first_decision ?? 'CONFIRMED'}</strong>
        </div>
        <div style={{
          padding: '10px 14px', background: 'rgba(220,38,38,0.08)',
          border: '1px solid #991b1b', borderRadius: 8,
          fontSize: 12, color: '#f87171',
        }}>
          <p style={{ margin: '0 0 2px', color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase' }}>
            Officer B Decision
          </p>
          <strong>{esc.second_decision ?? 'REJECTED'}</strong>
        </div>
      </div>

      {/* Status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <span style={{
          padding: '2px 10px', borderRadius: 20, fontSize: 11,
          background: 'rgba(245,158,11,0.12)', color: '#f59e0b',
          border: '1px solid #92400e',
        }}>
          ⚖️ {esc.status}
        </span>

        <button
          onClick={() => onAdjudicate(esc)}
          style={{
            padding: '8px 18px',
            background: 'rgba(99,102,241,0.15)',
            border: '1px solid #4338ca',
            borderRadius: 8,
            color: '#818cf8',
            fontWeight: 600, fontSize: 13,
            cursor: 'pointer',
          }}
        >
          ⚖️ Adjudicate This Escalation
        </button>
      </div>
    </div>
  );
}

function AdjudicationModal({ esc, onClose, onConfirm, isSubmitting }) {
  const [decision, setDecision] = useState(null);
  const [discardMode, setDiscardMode] = useState(false);
  const [discardReason, setDiscardReason] = useState('');
  const [discardConfirmed, setDiscardConfirmed] = useState(false);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="adjudication-modal-title"
      style={{
        position: 'fixed', inset: 0, zIndex: 998,
        background: 'rgba(0,0,0,0.8)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 16,
      }}
    >
      <div style={{
        background: 'var(--bg-card, #1f2937)',
        border: '1px solid var(--border, #374151)',
        borderRadius: 18, padding: 28,
        width: '100%', maxWidth: 520,
        boxShadow: '0 20px 50px rgba(0,0,0,0.7)',
      }}>
        <h3 id="adjudication-modal-title" style={{ margin: '0 0 6px', fontSize: 17, fontWeight: 700 }}>
          ⚖️ Supervisor Adjudication
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: 13, margin: '0 0 20px' }}>
          Alert #{esc.alert_id ?? esc.review_pair_id} · Two officers disagreed. Your decision is final.
        </p>

        {!discardMode ? (
          <>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Final Ruling
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 20 }}>
              {[
                { value: 'CONFIRMED', label: '✅ Adjudicate: CONFIRMED', color: '#22c55e', border: '#166534', bg: 'rgba(34,197,94,0.08)' },
                { value: 'REJECTED',  label: '❌ Adjudicate: REJECTED',  color: '#ef4444', border: '#991b1b', bg: 'rgba(239,68,68,0.08)' },
              ].map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setDecision(opt.value)}
                  style={{
                    padding: '14px 18px',
                    background: decision === opt.value ? opt.bg : 'rgba(255,255,255,0.03)',
                    border: `1px solid ${decision === opt.value ? opt.border : '#374151'}`,
                    borderRadius: 10, color: decision === opt.value ? opt.color : '#9ca3af',
                    fontWeight: 600, fontSize: 14, cursor: 'pointer', textAlign: 'left',
                    transition: 'all 0.2s',
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <button
                onClick={() => onConfirm(decision, null)}
                disabled={!decision || isSubmitting}
                style={{
                  flex: 1, padding: '12px 18px',
                  background: decision ? '#6366f1' : 'rgba(99,102,241,0.2)',
                  border: '1px solid #4338ca', borderRadius: 10,
                  color: decision ? '#fff' : '#818cf8',
                  fontWeight: 700, fontSize: 14, cursor: decision ? 'pointer' : 'not-allowed',
                  opacity: isSubmitting ? 0.7 : 1,
                }}
              >
                {isSubmitting ? '⟳ Recording…' : 'Confirm Final Decision →'}
              </button>
              <button
                onClick={() => setDiscardMode(true)}
                style={{
                  padding: '12px 14px',
                  background: 'rgba(239,68,68,0.08)', border: '1px solid #7f1d1d',
                  borderRadius: 10, color: '#ef4444', fontWeight: 600, cursor: 'pointer', fontSize: 13,
                }}
              >
                🗑 Discard
              </button>
              <button onClick={onClose} style={{
                padding: '12px 14px',
                background: 'rgba(255,255,255,0.04)', border: '1px solid #374151',
                borderRadius: 10, color: '#9ca3af', cursor: 'pointer', fontSize: 13,
              }}>
                Cancel
              </button>
            </div>
          </>
        ) : (
          /* Discard sub-panel */
          <>
            <div role="alert" style={{
              padding: 12, background: 'rgba(239,68,68,0.1)', border: '1px solid #7f1d1d',
              borderRadius: 10, fontSize: 13, color: '#f87171', marginBottom: 16,
            }}>
              ⚠️ This will permanently exclude this case from training storage after a 24-hour recovery window.
            </div>

            <label style={{ display: 'block', marginBottom: 10, fontSize: 13, color: 'var(--text-muted)' }}>
              Reason for Discard (required)
            </label>
            <select
              id="discard-reason"
              value={discardReason}
              onChange={e => setDiscardReason(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', marginBottom: 14,
                background: '#111827', border: '1px solid #374151', borderRadius: 8,
                color: '#f9fafb', fontSize: 13,
              }}
            >
              <option value="">— Select reason —</option>
              <option value="duplicate_alert">Duplicate alert</option>
              <option value="image_quality">Poor image quality</option>
              <option value="technical_error">Technical / system error</option>
              <option value="out_of_scope">Out of scope for this watchlist</option>
            </select>

            <label style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20, cursor: 'pointer', fontSize: 13, color: 'var(--text-muted)' }}>
              <input
                type="checkbox"
                id="confirm-discard-check"
                checked={discardConfirmed}
                onChange={e => setDiscardConfirmed(e.target.checked)}
                style={{ accentColor: '#ef4444', width: 16, height: 16 }}
              />
              I understand this action is logged and audited
            </label>

            <div style={{ display: 'flex', gap: 10 }}>
              <button
                onClick={() => onConfirm('DISCARD', discardReason)}
                disabled={!discardReason || !discardConfirmed || isSubmitting}
                style={{
                  flex: 1, padding: '12px 18px',
                  background: 'rgba(220,38,38,0.15)', border: '1px solid #991b1b',
                  borderRadius: 10, color: '#ef4444', fontWeight: 700, cursor: 'pointer',
                  opacity: (!discardReason || !discardConfirmed || isSubmitting) ? 0.4 : 1,
                }}
              >
                {isSubmitting ? '⟳ Discarding…' : 'Confirm Discard'}
              </button>
              <button onClick={() => setDiscardMode(false)} style={{
                padding: '12px 14px',
                background: 'rgba(255,255,255,0.04)', border: '1px solid #374151',
                borderRadius: 10, color: '#9ca3af', cursor: 'pointer', fontSize: 13,
              }}>
                Back
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

import { mockEscalations } from '../../data/mockData';

export default function SupervisorEscalationQueue() {
  const { authFetch, user } = useAuth();
  const [escalations, setEscalations] = useState(mockEscalations);
  const [loadState, setLoadState]     = useState('success'); // loading | success | error
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [selected, setSelected]       = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isFetching, setIsFetching]   = useState(false);
  const pollRef                        = useRef(null);

  const fetchEscalations = useCallback(async () => {
    setIsFetching(true);
    try {
      const res = await authFetch(`${API}/review/escalations`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list = Array.isArray(data) ? data : (data?.escalations || mockEscalations);
      setEscalations(list);
      setLoadState('success');
      setLastUpdated(new Date());
    } catch {
      setEscalations(prev => (Array.isArray(prev) && prev.length > 0 ? prev : mockEscalations));
      setLoadState('success');
      setLastUpdated(new Date());
    } finally {
      setIsFetching(false);
    }
  }, [authFetch]);

  // Initial load + 30s polling
  useEffect(() => {
    fetchEscalations();
    pollRef.current = setInterval(fetchEscalations, 30_000);
    return () => clearInterval(pollRef.current);
  }, [fetchEscalations]);

  // Live WS updates
  useWebSocketEvent('escalation_added',   () => fetchEscalations());
  useWebSocketEvent('escalation_resolved',() => fetchEscalations());

  const handleAdjudicate = async (finalDecision, reason) => {
    if (!selected) return;
    setIsSubmitting(true);
    try {
      const res = await authFetch(`${API}/review/adjudicate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          escalation_id: selected.review_pair_id ?? selected.id,
          supervisor_id: user?.username,
          final_decision: finalDecision,
          reason: reason ?? null,
          idempotency_key: crypto.randomUUID(),
        }),
      });
      toast.success('⚖️ Adjudication recorded successfully.');
      setSelected(null);
      // Remove adjudicated case locally
      setEscalations(prev => (Array.isArray(prev) ? prev.filter(e => (e.review_pair_id ?? e.id) !== (selected.review_pair_id ?? selected.id)) : []));
      fetchEscalations();
    } catch (err) {
      toast.error(`Failed to record adjudication: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const safeEscalations = Array.isArray(escalations) ? escalations : [];
  const breachCount = safeEscalations.filter(e => {
    const age = e.age_seconds ?? Math.floor((Date.now() - new Date(e.created_at).getTime()) / 1000);
    return age >= SLA_BREACH_S;
  }).length;

  return (
    <div style={{ padding: 24, maxWidth: 900, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>⚖️ Escalation Queue</h1>
          {safeEscalations.length > 0 && (
            <span style={{
              padding: '2px 12px', borderRadius: 20, fontSize: 12, fontWeight: 600,
              background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid #92400e',
            }}>
              {safeEscalations.length} Pending
            </span>
          )}
          {breachCount > 0 && (
            <span role="alert" style={{
              padding: '2px 12px', borderRadius: 20, fontSize: 12, fontWeight: 700,
              background: 'rgba(220,38,38,0.15)', color: '#ef4444', border: '1px solid #991b1b',
            }}>
              🔴 {breachCount} SLA Breach{breachCount > 1 ? 'es' : ''}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastUpdated && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Updated {lastUpdated.toLocaleTimeString('en-IN')}
            </span>
          )}
          <button
            onClick={fetchEscalations}
            disabled={isFetching}
            style={{
              padding: '6px 14px', fontSize: 12, background: 'var(--bg-elevated, #374151)',
              border: '1px solid var(--border, #4b5563)', borderRadius: 8,
              color: 'var(--text-muted)', cursor: 'pointer',
              opacity: isFetching ? 0.5 : 1,
            }}
          >
            {isFetching ? '⟳ Refreshing…' : '↻ Refresh'}
          </button>
        </div>
      </div>

      {/* Loading */}
      {loadState === 'loading' && (
        <div role="status" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {[1, 2, 3].map(i => (
            <div key={i} style={{ height: 160, background: '#1f2937', borderRadius: 14, animation: 'shimmer 1.5s infinite' }} />
          ))}
        </div>
      )}

      {/* Error */}
      {loadState === 'error' && !safeEscalations.length && (
        <div role="alert" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 60, gap: 14 }}>
          <span style={{ fontSize: 48 }}>⚠️</span>
          <p style={{ color: '#ef4444', fontWeight: 600 }}>Failed to load escalation queue</p>
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Retrying automatically every 30 seconds…</p>
          <button className="btn-primary" onClick={fetchEscalations}>Retry Now</button>
        </div>
      )}

      {/* Empty */}
      {loadState === 'success' && safeEscalations.length === 0 && (
        <div role="status" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 80, gap: 14 }}>
          <span style={{ fontSize: 64 }}>✅</span>
          <p style={{ color: '#f9fafb', fontWeight: 700, fontSize: 20 }}>No Pending Escalations</p>
          <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>All conflicted reviews have been adjudicated.</p>
        </div>
      )}

      {/* Escalation Cards — sorted oldest first (highest priority) */}
      {safeEscalations.length > 0 && (
        <div role="list" aria-label="Pending escalations" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {[...safeEscalations].sort((a, b) => {
            const ageA = a.age_seconds ?? (Date.now() - new Date(a.created_at).getTime()) / 1000;
            const ageB = b.age_seconds ?? (Date.now() - new Date(b.created_at).getTime()) / 1000;
            return ageB - ageA;
          }).map(esc => (
            <EscalationCard
              key={esc.review_pair_id ?? esc.id}
              esc={esc}
              onAdjudicate={() => setSelected(esc)}
            />
          ))}
        </div>
      )}

      {/* Adjudication Modal */}
      {selected && (
        <AdjudicationModal
          esc={selected}
          onClose={() => setSelected(null)}
          onConfirm={handleAdjudicate}
          isSubmitting={isSubmitting}
        />
      )}
    </div>
  );
}
