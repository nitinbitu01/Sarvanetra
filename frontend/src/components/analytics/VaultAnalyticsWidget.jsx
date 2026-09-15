// frontend/src/components/analytics/VaultAnalyticsWidget.jsx
// Active Learning Vault analytics panel showing:
//   - Total samples, hard positives, hard negatives, frozen anchors
//   - SVG bar chart: sample counts per Gujarat weather condition
//   - Retraining readiness progress bar
//   - Rubber-stamp officer detection alert
// Uses existing AuthContext and WebSocketContext. No external chart library.

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useWebSocketEvent } from '../../context/WebSocketContext';

const API = import.meta.env.VITE_API_URL || '/api/v1';
const POLL_INTERVAL_MS = 60_000;

// Presentation only. Conditions the classifier emits but this map does not
// name still appear, using the fallback below — the chart renders what was
// measured rather than a fixed set of rows.
//
// It used to iterate this object instead of the response, so a condition the
// backend reported and this file did not list was dropped from the chart
// (WET, 5,293 entries), while conditions absent from the footage were drawn
// as a confident zero (MONSOON, DUST_STORM). A zero bar and a missing bar
// mean different things, and neither was correct.
const ENV_CONFIG = {
  DAY:         { label: '☀️ Day',         color: '#f59e0b' },
  NIGHT:       { label: '🌙 Night',       color: '#6366f1' },
  NIGHT_GLARE: { label: '💡 Night Glare', color: '#f97316' },
  WET:         { label: '🌧️ Wet road',   color: '#3b82f6' },
  MONSOON:     { label: '🌧️ Monsoon',    color: '#2563eb' },
  DUST_STORM:  { label: '🌪️ Dust Storm', color: '#a16207' },
};

const ENV_FALLBACK_COLOR = '#94a3b8';

function envLabel(key) {
  return ENV_CONFIG[key]?.label
    ?? key.replace(/_/g, ' ').toLowerCase().replace(/^./, (c) => c.toUpperCase());
}

function MetricCard({ icon, label, value, color }) {
  return (
    <div style={{
      padding: '14px 16px',
      background: 'var(--bg-elevated, #111827)',
      border: '1px solid var(--border, #374151)',
      borderRadius: 12,
      display: 'flex', flexDirection: 'column', gap: 4,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 18 }} aria-hidden="true">{icon}</span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color: color ?? '#f9fafb' }}>
        {value ?? '—'}
      </div>
    </div>
  );
}

function EnvironmentalBars({ distribution }) {
  // Driven by the response, largest first.
  const entries = Object.entries(distribution || {})
    .filter(([, n]) => Number.isFinite(n))
    .sort((a, b) => b[1] - a[1]);

  if (entries.length === 0) {
    return (
      <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0, fontStyle: 'italic' }}>
        No conditions classified yet.
      </p>
    );
  }

  const maxCount = Math.max(...entries.map(([, n]) => n), 1);
  const totalCount = entries.reduce((sum, [, n]) => sum + n, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }} role="list" aria-label="Environmental diversity distribution">
      {entries.map(([key, count]) => {
        const cfg = { label: envLabel(key), color: ENV_CONFIG[key]?.color ?? ENV_FALLBACK_COLOR };
        const pct = Math.round((count / totalCount) * 100) || 0;
        const barWidth = maxCount > 0 ? (count / maxCount) * 100 : 0;

        return (
          <div key={key} role="listitem" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ width: 110, fontSize: 13, color: 'var(--text-muted)', flexShrink: 0 }}>
              {cfg.label}
            </span>
            <div
              style={{ flex: 1, height: 14, background: '#1f2937', borderRadius: 7, overflow: 'hidden' }}
              role="progressbar"
              aria-label={`${cfg.label}: ${count} samples (${pct}%)`}
              aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
            >
              <div style={{
                height: '100%', width: `${barWidth}%`,
                background: cfg.color, borderRadius: 7,
                transition: 'width 0.8s ease',
              }} />
            </div>
            <span style={{ width: 72, fontSize: 12, textAlign: 'right', color: cfg.color, flexShrink: 0 }}>
              {count} ({pct}%)
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function VaultAnalyticsWidget() {
  const { authFetch } = useAuth();
  const [data, setData]           = useState(null);
  const [loadState, setLoadState] = useState('loading');
  const [isFetching, setIsFetching] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const pollRef                   = useRef(null);

  const fetchStats = useCallback(async () => {
    setIsFetching(true);
    try {
      const res = await authFetch(`${API}/vault/stats`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setLoadState('success');
      setLastUpdated(new Date());
    } catch (err) {
      if (loadState !== 'success') setLoadState('error');
      // Keep showing stale data if we have it
    } finally {
      setIsFetching(false);
    }
  }, [authFetch]);

  useEffect(() => {
    fetchStats();
    pollRef.current = setInterval(fetchStats, POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [fetchStats]);

  // Invalidate on vault updates via WebSocket
  useWebSocketEvent('vault_updated', () => fetchStats());

  const isStale = lastUpdated && (Date.now() - lastUpdated.getTime() > 5 * 60_000);

  const retrainPct = data
    ? Math.min((data.retrain_eligible_count / (data.retrain_threshold || 1)) * 100, 100)
    : 0;

  const rubberStampOfficers = data?.officer_throughput?.filter(o => o.rubber_stamp_flag) ?? [];

  // ── Skeleton ──────────────────────────────────────────────────────────────────
  if (loadState === 'loading' && !data) {
    return (
      <section style={{ background: 'var(--bg-elevated, #1f2937)', border: '1px solid var(--border, #374151)', borderRadius: 16, padding: 20 }}
               aria-label="Vault analytics loading">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }} role="status">
          <div style={{ height: 24, width: 200, background: '#374151', borderRadius: 6, animation: 'pulse 1.5s infinite' }} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {[1,2,3,4].map(i => <div key={i} style={{ height: 80, background: '#374151', borderRadius: 10, animation: 'pulse 1.5s infinite' }} />)}
          </div>
          <div style={{ height: 140, background: '#374151', borderRadius: 10, animation: 'pulse 1.5s infinite' }} />
        </div>
      </section>
    );
  }

  // ── Hard error (no stale data) ────────────────────────────────────────────────
  if (loadState === 'error' && !data) {
    return (
      <section style={{ background: 'var(--bg-elevated, #1f2937)', border: '1px solid #7f1d1d', borderRadius: 16, padding: 20 }}
               role="alert" aria-label="Vault analytics error">
        <p style={{ color: '#ef4444', fontWeight: 600 }}>⚠️ Vault stats unavailable</p>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Service may be busy during model retraining.</p>
        <button onClick={fetchStats} style={{ marginTop: 10, padding: '8px 16px', background: '#374151', border: '1px solid #4b5563', borderRadius: 8, color: '#f9fafb', cursor: 'pointer', fontSize: 13 }}>
          Retry
        </button>
      </section>
    );
  }

  return (
    <section
      style={{ background: 'var(--bg-elevated, #1f2937)', border: '1px solid var(--border, #374151)', borderRadius: 16, padding: 20, display: 'flex', flexDirection: 'column', gap: 18 }}
      aria-label="Active Learning Vault Analytics"
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>🧠 Active Learning Vault</h2>
          {isStale && (
            <span role="alert" style={{ fontSize: 11, padding: '2px 8px', background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid #92400e', borderRadius: 10 }}>
              ⚠️ Stale Data
            </span>
          )}
          {loadState === 'error' && data && (
            <span role="alert" style={{ fontSize: 11, padding: '2px 8px', background: 'rgba(239,68,68,0.12)', color: '#f87171', border: '1px solid #7f1d1d', borderRadius: 10 }}>
              Using Cached Data
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {lastUpdated && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {lastUpdated.toLocaleTimeString('en-IN')}
            </span>
          )}
          <button
            onClick={fetchStats}
            disabled={isFetching}
            aria-label="Refresh vault analytics"
            style={{ padding: '4px 10px', fontSize: 12, background: 'rgba(255,255,255,0.04)', border: '1px solid #374151', borderRadius: 6, color: 'var(--text-muted)', cursor: 'pointer', opacity: isFetching ? 0.5 : 1 }}
          >
            {isFetching ? '⟳' : '↻'}
          </button>
        </div>
      </div>

      {/* Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <MetricCard icon="🗄️" label="Total Samples" value={data?.total_samples} color="#f9fafb" />
        <MetricCard icon="🎯" label="Hard Positives" value={data?.hard_positives} color="#22c55e" />
        <MetricCard icon="🛡️" label="Hard Negatives" value={data?.hard_negatives} color="#ef4444" />
        <MetricCard icon="⚓" label="Frozen Anchors" value={data?.frozen_anchors} color="#818cf8" />
      </div>

      {/* Environmental Distribution Bar Chart */}
      <div>
        <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', margin: '0 0 12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          🌍 Environmental Diversity
        </h3>
        <EnvironmentalBars distribution={data?.environmental_distribution ?? {}} />
      </div>

      {/* Retraining Readiness */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexWrap: 'wrap', gap: 6 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', margin: 0, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            🔄 Retraining Readiness
          </h3>
          <span style={{ fontSize: 13, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
            {data?.retrain_eligible_count ?? 0} / {data?.retrain_threshold ?? '—'}
          </span>
        </div>
        <div
          style={{ width: '100%', height: 22, background: '#111827', borderRadius: 11, overflow: 'hidden' }}
          role="progressbar"
          aria-label={`Retraining readiness: ${Math.round(retrainPct)}%`}
          aria-valuenow={Math.round(retrainPct)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div style={{
            height: '100%', width: `${retrainPct}%`,
            background: retrainPct >= 100 ? '#22c55e' : retrainPct >= 75 ? '#f59e0b' : '#3b82f6',
            borderRadius: 11,
            transition: 'width 0.8s ease',
            display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 8,
            animation: retrainPct >= 100 ? 'pulse 2s infinite' : 'none',
          }}>
            {retrainPct >= 20 && (
              <span style={{ fontSize: 11, fontWeight: 700, color: '#fff' }}>
                {Math.round(retrainPct)}%
              </span>
            )}
          </div>
        </div>
        {retrainPct >= 100 && (
          <p role="status" style={{ color: '#22c55e', fontSize: 13, margin: '8px 0 0', fontWeight: 600 }}>
            ✅ Threshold reached — ready for retraining pipeline trigger
          </p>
        )}
        {data?.last_retrain_at && (
          <p style={{ color: 'var(--text-muted)', fontSize: 11, margin: '4px 0 0' }}>
            Last retrain: {new Date(data.last_retrain_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
          </p>
        )}
      </div>

      {/* Rubber-stamp officer alert */}
      {rubberStampOfficers.length > 0 && (
        <div role="alert" style={{
          padding: 16, background: 'rgba(220,38,38,0.08)', border: '1px solid #7f1d1d', borderRadius: 12,
        }}>
          <p style={{ color: '#ef4444', fontWeight: 700, fontSize: 13, margin: '0 0 10px' }}>
            ⚠️ Potential Rubber-Stamping Detected
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {rubberStampOfficers.map(o => (
              <div key={o.officer_id} style={{
                display: 'flex', justifyContent: 'space-between',
                padding: '6px 10px', background: 'rgba(255,255,255,0.03)',
                borderRadius: 8, fontSize: 12, color: 'var(--text-muted)',
              }}>
                <code style={{ color: '#f9fafb' }}>{o.officer_id}</code>
                <span>{(o.reviews_per_hour ?? 0).toFixed(1)} reviews/hr · avg {(o.avg_review_duration_s ?? 0).toFixed(1)}s/review</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
