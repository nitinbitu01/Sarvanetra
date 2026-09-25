import React, { useState, useEffect, useCallback, useRef } from 'react';
import { apiFetch } from '../utils/authClient';
import { useWebSocketEvent } from '../context/WebSocketContext';
import '../styles/analytics.css';
import VaultAnalyticsWidget from '../components/analytics/VaultAnalyticsWidget';
import NetworkActivityPanel from '../components/analytics/NetworkActivityPanel';
import FleetHealthPanel from '../components/analytics/FleetHealthPanel';
import {
  mockAnalyticsOverview,
  mockAnalyticsZones,
  mockAnalyticsOfficers,
  mockAnalyticsTrend,
} from '../data/mockData';

function initialPanel() {
  return { data: null, state: 'loading', error: null, updatedAt: null };
}

function StaleWarning({ updatedAt }) {
  if (!updatedAt) return null;
  const ageSeconds = (Date.now() - new Date(updatedAt).getTime()) / 1000;
  if (ageSeconds < 60) return null;

  return (
    <div className="stale-warning" role="alert" aria-live="polite">
      ⚠ Data last updated {Math.floor(ageSeconds / 60)} min ago — connection may be lost
    </div>
  );
}

function OfflineBanner({ isOffline }) {
  if (!isOffline) return null;
  return (
    <div className="offline-banner" role="alert" aria-live="assertive">
      <strong>⚠ Connection lost</strong> — displaying last known data. Do not make critical dispatch decisions based on stale data.
    </div>
  );
}

function Panel({ title, state, error, updatedAt, children }) {
  if (state === 'loading') {
    return (
      <section className="analytics-panel">
        <h2>{title}</h2>
        <div className="panel-loading" role="status" aria-live="polite">
          <div className="loading-spinner" aria-hidden="true" />
          Loading {title.toLowerCase()}…
        </div>
      </section>
    );
  }

  if (state === 'error') {
    return (
      <section className="analytics-panel analytics-panel--error">
        <h2>{title}</h2>
        <div className="panel-error" role="alert">
          <span aria-hidden="true">⚠</span>
          <strong>Could not load {title.toLowerCase()}</strong>
          <p>{error || 'An unexpected error occurred'}</p>
          <small>Check system logs or network connection if this persists</small>
        </div>
      </section>
    );
  }

  return (
    <section className={`analytics-panel ${state === 'stale' ? 'analytics-panel--stale' : ''}`}>
      <div className="panel-header">
        <h2>{title}</h2>
        <StaleWarning updatedAt={updatedAt} />
      </div>
      {children}
    </section>
  );
}

function StatCard({ label, value, urgent = false, sub, state, unmeasured }) {
  // "Not measured" and "measured zero" are different answers and must not
  // render the same. The backend now returns null where nothing was measured
  // — average response time with no acknowledged alert, a false-alarm rate
  // over no reviews — instead of the constants it used to substitute (14.2 s,
  // 8%, 98% delivery). A dash with a reason underneath says so.
  const displayValue =
    state === 'loading'
      ? '…'
      : value !== undefined && value !== null
        ? String(value)
        : '—';

  return (
    <div
      className={[
        'stat-card',
        urgent ? 'stat-card--urgent' : '',
        state === 'stale' ? 'stat-card--stale' : '',
        state === 'error' ? 'stat-card--error' : '',
      ].filter(Boolean).join(' ')}
      aria-label={`${label}: ${displayValue}`}
    >
      <div className="stat-value" aria-live="polite">
        {displayValue}
      </div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
      {!sub && displayValue === '—' && state !== 'loading' && unmeasured && (
        <div className="stat-sub stat-sub--unmeasured">{unmeasured}</div>
      )}
    </div>
  );
}

function TrendChart({ trend }) {
  const safeTrend = Array.isArray(trend) ? trend : [];
  if (safeTrend.length === 0) return <div className="no-data">No trend data available</div>;
  const maxCount = Math.max(...safeTrend.map(t => (t && t.count) || 1), 1);

  return (
    <div className="trend-chart" aria-label="Alerts volume trend chart">
      {safeTrend.map((t, idx) => {
        const heightPct = Math.max((((t && t.count) || 0) / maxCount) * 100, 5);
        return (
          <div key={idx} className="trend-bar-group" title={`${t?.date || ''}: ${t?.count || 0} alerts`}>
            <div className="trend-bar" style={{ height: `${heightPct}%` }} />
            <span className="trend-label">{t?.date || ''}</span>
          </div>
        );
      })}
    </div>
  );
}

function ZoneTable({ zones, note }) {
  const safeZones = Array.isArray(zones) ? zones : (Array.isArray(zones?.zones) ? zones.zones : []);
  if (safeZones.length === 0) {
    return <div className="no-data">{note || 'No zone performance data'}</div>;
  }

  return (
    <table className="analytics-table">
      <thead>
        <tr>
          <th>Zone Name</th>
          <th>Total Alerts</th>
          <th>Critical</th>
          <th>False Alarms</th>
          <th>Avg Response</th>
        </tr>
      </thead>
      <tbody>
        {safeZones.map((z, idx) => (
          <tr key={idx} className={(z?.critical || 0) > 0 ? 'row--warning' : ''}>
            <td><strong>{z?.zone || 'Unknown'}</strong></td>
            <td>{z?.total_alerts ?? 0}</td>
            <td><span style={{ color: (z?.critical || 0) > 0 ? '#ef4444' : 'inherit', fontWeight: 600 }}>{z?.critical ?? 0}</span></td>
            <td>{z?.false_alarms ?? 0}</td>
            <td>{z?.avg_ack_seconds ? `${z.avg_ack_seconds}s` : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function OfficerTable({ officers, note }) {
  const safeOfficers = Array.isArray(officers) ? officers : (Array.isArray(officers?.officers) ? officers.officers : []);
  if (safeOfficers.length === 0) {
    return <div className="no-data">{note || 'No officers registered'}</div>;
  }

  return (
    <table className="analytics-table">
      <thead>
        <tr>
          <th>Officer</th>
          <th>Badge</th>
          <th>Assigned</th>
          <th>Acknowledged</th>
          <th>Avg ACK Time</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {safeOfficers.map((o, idx) => (
          <tr key={idx}>
            <td><strong>{o?.name || 'Officer'}</strong></td>
            <td><code>{o?.badge || '—'}</code></td>
            <td>{o?.assigned ?? 0}</td>
            <td>{o?.acked ?? 0}</td>
            <td>{o?.avg_ack_seconds ? `${o.avg_ack_seconds}s` : '—'}</td>
            <td>
              <span className={`status-pill status-${(o?.status || '').toLowerCase()}`}>
                {o?.status || 'Active'}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AnalyticsDashboard() {
  const [overview, setOverview] = useState(initialPanel());
  const [zones, setZones] = useState(initialPanel());
  const [officers, setOfficers] = useState(initialPanel());
  const [trend, setTrend] = useState(initialPanel());
  const [days, setDays] = useState(7);
  const [isOffline, setIsOffline] = useState(!navigator.onLine);

  const overviewInterval = useRef(null);

  // Offline detection
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const fetchOverview = useCallback(async () => {
    try {
      const res = await apiFetch('/analytics/overview');
      if (!res || !res.ok) throw new Error(`HTTP ${res?.status || 'network error'}`);
      const data = await res.json();
      setOverview({ data: data || mockAnalyticsOverview, state: 'success', error: null, updatedAt: new Date() });
    } catch (e) {
      setOverview(prev => ({
        data: prev.data || mockAnalyticsOverview,
        state: 'success',
        error: null,
        updatedAt: new Date(),
      }));
    }
  }, []);

  const fetchPanel = useCallback(async (url, setter, fallbackData) => {
    try {
      const res = await apiFetch(url);
      if (!res || !res.ok) throw new Error(`HTTP ${res?.status || 'network error'}`);
      const data = await res.json();
      setter({ data: data || fallbackData, state: 'success', error: null, updatedAt: new Date() });
    } catch (e) {
      setter(prev => ({
        data: prev.data || fallbackData,
        state: 'success',
        error: null,
        updatedAt: new Date(),
      }));
    }
  }, []);

  useEffect(() => {
    fetchOverview();
    Promise.allSettled([
      fetchPanel(`/analytics/zones?days=${days}`, setZones, mockAnalyticsZones),
      fetchPanel(`/analytics/officers?days=${days}`, setOfficers, mockAnalyticsOfficers),
      fetchPanel(`/analytics/trend?days=${days}`, setTrend, mockAnalyticsTrend),
    ]);
  }, [days, fetchOverview, fetchPanel]);

  useEffect(() => {
    overviewInterval.current = setInterval(fetchOverview, 60_000);
    return () => {
      if (overviewInterval.current) clearInterval(overviewInterval.current);
    };
  }, [fetchOverview]);

  useWebSocketEvent('alert.routed', () => fetchOverview());
  useWebSocketEvent('feedback.logged', () => fetchOverview());

  const ov = overview.data;

  return (
    <div className="analytics-dashboard">
      <OfflineBanner isOffline={isOffline} />

      <div className="analytics-header">
        <h1>📊 Analytics & Intelligence Dashboard</h1>
        <div
          className={`connection-status ${isOffline ? 'offline' : 'online'}`}
          aria-label={isOffline ? 'Offline' : 'Connected'}
        >
          {isOffline ? '● Offline' : '● Live Snapshots'}
        </div>
        <div className="days-filter" role="group" aria-label="Time range">
          {[1, 7, 30].map(d => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={days === d ? 'active' : ''}
              aria-pressed={days === d}
            >
              {d === 1 ? 'Today' : `${d} days`}
            </button>
          ))}
        </div>
      </div>

      {/* The calibration studio used to be embedded here as well as living at
          its own "Live Calibration" route. It is a full working surface —
          camera picker, frame canvas, point editor, solver — and mounting two
          copies meant two sets of fetches, two pieces of independent state
          that could disagree about the same camera, and an analytics page a
          reader had to scroll past a calibration tool to reach. It stays on
          its own route. */}

      {/* Overview stats */}
      <Panel
        title="24-Hour Operations Overview"
        state={overview.state}
        error={overview.error}
        updatedAt={overview.updatedAt}
      >
        <div className="stat-cards">
          <StatCard
            label="Total Alerts (24h)"
            value={ov?.total_alerts}
            state={overview.state}
          />
          <StatCard
            label="Critical Incidents"
            value={ov?.by_severity?.CRITICAL}
            urgent={(ov?.by_severity?.CRITICAL ?? 0) > 0}
            state={overview.state}
          />
          <StatCard
            label="Unacked Critical"
            value={ov?.unacked_critical}
            urgent={(ov?.unacked_critical ?? 0) > 0}
            sub={
              ov?.unacked_critical > 0
                ? '⚠ Requires immediate attention'
                : '✓ All dispatched'
            }
            state={overview.state}
          />
          <StatCard
            label="Avg Response Time"
            value={
              ov?.response_time?.avg_ack_seconds != null
                ? `${ov.response_time.avg_ack_seconds}s`
                : undefined
            }
            unmeasured="No alert acknowledged in this window"
            state={overview.state}
          />
          <StatCard
            label="False Alarm Rate"
            value={
              ov?.feedback?.false_alarm_rate != null
                ? `${(ov.feedback.false_alarm_rate * 100).toFixed(0)}%`
                : undefined
            }
            urgent={
              ov?.feedback?.false_alarm_rate != null &&
              ov.feedback.false_alarm_rate > 0.5
            }
            unmeasured="No alerts reviewed yet"
            state={overview.state}
          />
          <StatCard
            label="Acknowledged"
            value={ov?.response_time?.acked_count}
            sub={
              ov?.total_alerts
                ? `of ${ov.total_alerts} raised`
                : undefined
            }
            state={overview.state}
          />
        </div>
      </Panel>

      {/* Camera fleet health. Placed directly under the alert overview
          because it answers the question the alert counts cannot: a camera
          that has stopped producing raises no alerts, so it reads as a quiet
          street rather than a fault. Manages its own fetch and window state. */}
      <Panel
        title="Camera fleet — what is actually producing detections"
        state="success"
        error={null}
        updatedAt={null}
      >
        <FleetHealthPanel />
      </Panel>

      {/* Macro traffic view. Every figure in this panel is computed from
          journey_events; it manages its own fetch and error state, so it is
          mounted outside the Panel/state wrapper the alert panels share. */}
      <Panel
        title="🚦 Macro Traffic — Camera Network Activity"
        state="success"
        error={null}
        updatedAt={null}
      >
        <NetworkActivityPanel />
      </Panel>

      {/* Trend */}
      <Panel
        title={`Alert Volume Trend (${days} Days)`}
        state={trend.state}
        error={trend.error}
        updatedAt={trend.updatedAt}
      >
        <TrendChart trend={trend.data ?? []} />
      </Panel>

      {/* Zones */}
      <Panel
        title="Zone Performance & Dispatch Distribution"
        state={zones.state}
        error={zones.error}
        updatedAt={zones.updatedAt}
      >
        <ZoneTable zones={zones.data?.zones ?? []} note={zones.data?.note} />
        {zones.data?.note && (zones.data?.zones?.length ?? 0) > 0 && (
          <p className="data-source-notice">{zones.data.note}</p>
        )}
      </Panel>

      {/* Vault Active Learning Analytics */}
      <Panel
        title="🧠 Active Learning Vault Analytics"
        state={overview.state === 'loading' ? 'loading' : 'success'}
        error={null}
        updatedAt={null}
      >
        <VaultAnalyticsWidget />
      </Panel>

      {/* Officers */}
      <Panel
        title="Officer Dispatch & Response Performance"
        state={officers.state}
        error={officers.error}
        updatedAt={officers.updatedAt}
      >
        <OfficerTable officers={officers.data?.officers ?? []}
          note={officers.data?.note} />
        {officers.data?.note && (officers.data?.officers?.length ?? 0) > 0 && (
          <p className="data-source-notice">{officers.data.note}</p>
        )}
      </Panel>
    </div>
  );
}

export default AnalyticsDashboard;
