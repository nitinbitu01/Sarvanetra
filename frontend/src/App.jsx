// frontend/src/App.jsx â€” Sarvanetra AI Dashboard Shell
import './index.css';
import { useState, useCallback } from 'react';
import { useAuth } from './context/AuthContext';
import LoginForm from './components/LoginForm';
import CameraMap from './components/CameraMap';
import CameraOnboardingForm from './components/CameraOnboardingForm';
import BulkCameraImport from './components/BulkCameraImport';
import GapAnalysisPanel from './components/GapAnalysisPanel';
import WatchlistManager from './components/WatchlistManager';
import EvaluationMode from './components/EvaluationMode';
import AlertFeed from './components/AlertFeed';
import AuditLogPanel from './components/AuditLogPanel';
import SystemHealthBadge from './components/SystemHealthBadge';
import ReviewQueue from './components/ReviewQueue';
import JourneyView from './components/JourneyView';
import ValidationReportViewer from './components/ValidationReportViewer';
import SentinelIQCard from './components/SentinelIQCard';
import UnroutedBanner from './components/UnroutedBanner';
import OfficerPanel from './components/OfficerPanel';
import EnableAlertsButton from './components/EnableAlertsButton';
import AlertDetailCard from './components/AlertDetailCard';
import { warmVoiceList } from './utils/tts';
import { useWebSocketEvent } from './context/WebSocketContext';
import ControlRoom from './components/ControlRoom';
import { useEffect } from 'react';
import { useAuth as useAuthCtx, API } from './context/AuthContext';

import OfficerApp from './components/OfficerApp';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import ActiveReviewPage from './pages/ActiveReviewPage';
import SupervisorEscalationQueue from './components/escalation/SupervisorEscalationQueue';
import LiveCalibrationStudio from './components/analytics/LiveCalibrationStudio';
import TrafficHeatmap from './components/analytics/TrafficHeatmap';
import EdgeNodesPanel from './components/EdgeNodesPanel';
import FleetOperations from './components/FleetOperations';
import ReIDProofPanel from './components/ReIDProofPanel';
import TrajectoryEnginePanel from './components/TrajectoryEnginePanel';
import TrajectoryProofPanel from './components/TrajectoryProofPanel';
import Icon, { BrandMark } from './components/Icon';
import Top10CommandCentre from './pages/Top10CommandCentre';

// â”€â”€ Navigation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Grouped by what the operator is trying to DO. Sixteen items in one flat list
// is unscannable, and under demo pressure "which of these sixteen was the
// watchlist again" is a real cost. Order within a group is by frequency of use.
const NAV_GROUPS = [
  {
    section: 'MONITOR',
    items: [
      { id: 'top10',        icon: 'grid',           label: 'Top-10 Command Centre' },
      { id: 'control',       icon: 'monitor',        label: 'Control Room' },
      { id: 'map',           icon: 'map',            label: 'Camera Map' },
      { id: 'heatmap',       icon: 'flame',          label: 'Traffic Heatmap' },
      { id: 'live-studio',   icon: 'scan',           label: 'Live Calibration' },
    ],
  },
  {
    section: 'INVESTIGATE',
    items: [
      { id: 'evaluation',    icon: 'crosshair',      label: 'Trace Vehicle' },
      { id: 'journeys',      icon: 'route',          label: 'Journeys' },
      { id: 'reid-proof',    icon: 'searchCheck',    label: 'Cross-Camera Re-ID' },
      { id: 'trajectory-proof',  icon: 'route',       label: 'Trajectory Proof' },
      { id: 'validation',    icon: 'clipboardCheck', label: 'Validation', adminOnly: true },
    ],
  },
  {
    section: 'INTELLIGENCE',
    items: [
      { id: 'alerts',        icon: 'bell',           label: 'Alert Feed', badge: '3', badgeClass: 'critical' },
      { id: 'watchlist',     icon: 'shieldAlert',    label: 'Watchlist', badge: '147', adminOnly: true },
      { id: 'iq',            icon: 'cpu',            label: 'Sarvanetra IQ' },
      { id: 'escalations',   icon: 'triangleAlert',  label: 'Escalations', badge: '1', badgeClass: 'warning', adminOnly: true },
    ],
  },
  {
    section: 'OPERATIONS',
    items: [
      { id: 'reid-queue',    icon: 'searchCheck',    label: 'Review Queue', badge: '0' },
      { id: 'active-review', icon: 'scale',          label: 'Active Review' },
      { id: 'officer-pwa',   icon: 'phone',          label: 'Officer App' },
      { id: 'cameras',       icon: 'cameraAdd',      label: 'Add Camera', adminOnly: true },
    ],
  },
  {
    section: 'SYSTEM',
    items: [
      { id: 'fleet',         icon: 'monitor',        label: 'Fleet Operations' },
      { id: 'analytics',     icon: 'chart',          label: 'Analytics' },
      { id: 'edge',          icon: 'cpu',            label: 'Edge Inference' },
      { id: 'audit',         icon: 'list',           label: 'Audit Log', adminOnly: true },
    ],
  },
];

const NAV_COLLAPSED_KEY = 'sarvanetra.nav.collapsed';

// â”€â”€ Inner dashboard (shown after login) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function Dashboard() {
  const { user, logout, token, isAdmin, authFetch } = useAuth();
  const [page, setPage]           = useState('control');
  const [cameras, setCameras]     = useState([]);
  const [liveAlert, setLiveAlert] = useState(null);
  const [wsEvent, setWsEvent]     = useState(null); // last WS event (for ReID live updates)

  const refetchCameras = useCallback(() => {
    return authFetch(`${API}/cameras`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data) return;
        const list = Array.isArray(data) ? data : (data.cameras || []);
        if (list.length > 0) setCameras(list);
      })
      .catch(() => {});
  }, [authFetch]);

  // Load cameras on mount and periodically retry
  useEffect(() => {
    refetchCameras();
    const interval = setInterval(refetchCameras, 15000);
    return () => clearInterval(interval);
  }, [refetchCameras]);

  const onCameraAdded = useCallback((cam) => {
    setCameras(prev => [...prev, cam]);
  }, []);

  // Bulk/CSV import adds many rows at once and does not broadcast a
  // per-camera WS event the way single-camera create does â€” a full refetch
  // is simpler and no slower than reconciling a partial-success response
  // (some rows may have failed) against local state by hand.
  const onCamerasBulkImported = useCallback(() => { refetchCameras(); }, [refetchCameras]);

  const onToggleMaintenance = useCallback(async (cam) => {
    const turningOn = !cam.maintenance_mode;
    const note = turningOn
      ? (window.prompt('Maintenance note (optional):', '') ?? '')
      : null;
    try {
      const res = await authFetch(`${API}/cameras/${cam.id}/maintenance`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ maintenance_mode: turningOn, note: note || null }),
      });
      if (!res.ok) return;
      const data = await res.json();
      setCameras(prev => prev.map(c => c.id === cam.id
        ? { ...c, maintenance_mode: data.maintenance_mode, status: data.status,
            maintenance_note: data.maintenance_note }
        : c));
    } catch {
      // Non-critical UI action; the button simply stays in its prior state.
    }
  }, [authFetch]);

  const onWsEvent = useCallback((alertPayload) => {
    // useDashboardSocket only invokes this for `new_alert` messages, passing
    // msg.alert (the alert row itself, which has no `.type` field) â€” always
    // treat it as a live alert rather than re-checking a `.type` that isn't there.
    setWsEvent(alertPayload);
    setLiveAlert(alertPayload);
  }, []);

  // Day 7: camera heartbeat pushes ONLINE/OFFLINE transitions â€” patch the
  // matching camera in place so the map pin updates without a refetch.
  const onCameraStatusChanged = useCallback((evt) => {
    setCameras(prev => prev.map(c =>
      c.id === evt.camera_id
        ? { ...c, status: evt.status, last_heartbeat_at: evt.timestamp }
        : c
    ));
  }, []);

  // Day 8: alert lifecycle transitions (acknowledge/dismiss/escalate) from
  // ANY connected operator â€” surfaced to AlertFeed so every tab's list
  // reflects who already handled an alert, live.
  const [alertStatusEvent, setAlertStatusEvent] = useState(null);
  const onAlertStatusUpdate = useCallback((evt) => {
    setAlertStatusEvent(evt);
  }, []);

  // Day 11: Alert fatigue management (dedup + zone incidents)
  const [mergedEvent, setMergedEvent] = useState(null);
  const onAlertMerged = useCallback((evt) => {
    setMergedEvent(evt);
  }, []);

  const [zoneIncidentEvent, setZoneIncidentEvent] = useState(null);
  const onZoneIncidentOpened = useCallback((evt) => {
    setZoneIncidentEvent(evt);
  }, []);
  const onZoneIncidentUpdated = useCallback((evt) => {
    setZoneIncidentEvent(evt);
  }, []);

  // Day 14: one signal for all five routing event types. Bumping a counter
  // alongside the payload guarantees a state change even when two identical
  // events arrive back to back (e.g. two officer.status updates with the
  // same body) â€” a bare setState with an equal object would not re-render.
  const [routingEvent, setRoutingEvent] = useState(null);
  const onRoutingEvent = useCallback((evt) => {
    setRoutingEvent({ ...evt, _seq: Date.now() });
  }, []);

  // â”€â”€ Day 15: full-screen critical alert card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Opened two ways, both of which land here rather than on a route: this
  // app has no client-side router, so /alert/42 would 404 on the cold start
  // a notification tap produces.
  //   1. Notification tap  â†’ SW opens /?alert=42 â†’ read on mount below.
  //   2. Foreground push   â†’ SW postMessage â†’ opened without any tap.
  const [detailAlertId, setDetailAlertId] = useState(null);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('alert');
    if (id && /^\d+$/.test(id)) {
      setDetailAlertId(Number(id));
      // Strip the param so a later refresh doesn't reopen a card the officer
      // already dismissed.
      window.history.replaceState({}, '', window.location.pathname);
    }
    // getVoices() is empty on Chrome's first call; warm it now so the Hindi
    // voice is available by the time an alert actually arrives.
    warmVoiceList();
  }, []);

  useEffect(() => {
    if (!('serviceWorker' in navigator)) return undefined;
    // Foreground path: the SW sends postMessage instead of showing an OS
    // notification when a window is focused, so the card must open here with
    // no tap involved.
    function onSWMessage(event) {
      if (event.data?.type === 'CRITICAL_ALERT' && event.data.payload?.alert_id) {
        setDetailAlertId(event.data.payload.alert_id);
      }
    }
    navigator.serviceWorker.addEventListener('message', onSWMessage);
    return () => navigator.serviceWorker.removeEventListener('message', onSWMessage);
  }, []);

  // Day 18: App no longer opens its own socket.
  //
  // WebSocketProvider (main.jsx) owns the ONE connection via
  // useDashboardSocket and re-broadcasts to subscribers. Calling
  // useDashboardSocket here as well would open a SECOND WebSocket with its
  // own auth handshake and reconnect loop â€” visible as two entries in
  // DevTools â†’ Network â†’ WS, and every event handled twice.
  useWebSocketEvent('camera_added', onCameraAdded);
  useWebSocketEvent('new_alert', onWsEvent);
  useWebSocketEvent('camera_status_changed', onCameraStatusChanged);
  useWebSocketEvent('alert_status_update', onAlertStatusUpdate);
  useWebSocketEvent('alert_merged', onAlertMerged);
  useWebSocketEvent('zone_incident_opened', onZoneIncidentOpened);
  useWebSocketEvent('zone_incident_updated', onZoneIncidentUpdated);
  useWebSocketEvent('alert.routed', onRoutingEvent);
  useWebSocketEvent('alert.ack', onRoutingEvent);
  useWebSocketEvent('alert.escalated', onRoutingEvent);
  useWebSocketEvent('alert.unrouted', onRoutingEvent);
  useWebSocketEvent('officer.status', onRoutingEvent);

  // Resizable sidebar state with localStorage persistence (200px - 360px)
  const SIDEBAR_WIDTH_KEY = 'sarvanetra.c2.sidebar_width';
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    try {
      const saved = localStorage.getItem(SIDEBAR_WIDTH_KEY);
      return saved ? Math.max(200, Math.min(360, parseInt(saved, 10))) : 232;
    } catch {
      return 232;
    }
  });
  const [isDraggingNav, setIsDraggingNav] = useState(false);

  // Collapsed rail persists per operator
  const [navCollapsed, setNavCollapsed] = useState(() => {
    try { return localStorage.getItem(NAV_COLLAPSED_KEY) === '1'; } catch { return false; }
  });

  const toggleNav = useCallback(() => {
    setNavCollapsed(prev => {
      const next = !prev;
      try { localStorage.setItem(NAV_COLLAPSED_KEY, next ? '1' : '0'); } catch { /* private mode */ }
      return next;
    });
  }, []);

  const startResizeNav = useCallback((e) => {
    e.preventDefault();
    setIsDraggingNav(true);
    const startX = e.clientX;
    const startW = sidebarWidth;

    const onMouseMove = (moveEvent) => {
      const delta = moveEvent.clientX - startX;
      const newW = Math.max(200, Math.min(360, startW + delta));
      setSidebarWidth(newW);
    };

    const onMouseUp = () => {
      setIsDraggingNav(false);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      try {
        localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
      } catch {}
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }, [sidebarWidth]);

  useEffect(() => {
    if (!isDraggingNav) {
      try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth)); } catch {}
    }
  }, [sidebarWidth, isDraggingNav]);

  // Global Keyboard Shortcuts (âŒ˜K for search, âŒ˜B for sidebar toggle)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('global-search-input')?.focus();
      } else if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault();
        toggleNav();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleNav]);

  const navGroups = NAV_GROUPS
    .map(g => ({ ...g, items: g.items.filter(i => isAdmin || !i.adminOnly) }))
    .filter(g => g.items.length > 0);

  const onlineCount = cameras.filter(c => c.status === 'ONLINE').length;

  return (
    <div className="app-shell">
      {detailAlertId && (
        <AlertDetailCard
          alertId={detailAlertId}
          onClose={() => setDetailAlertId(null)}
        />
      )}

      {/* â”€â”€ Fixed 52px C2 Top Nav â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <header className="topnav">
        <div className="topnav-left">
          <div className="topnav-logo" onClick={() => setPage('control')} title="Sarvanetra Control Room">
            <BrandMark size={24} />
            <div className="topnav-logo-name">Sarvanetra</div>
          </div>
          <div className="topnav-logo-divider" />
          <div className="topnav-platform-tag">CCTV INTELLIGENCE PLATFORM</div>
        </div>

        {/* 480px Global Search Bar with âŒ˜K Focus Ring */}
        <div className="c2-search-container">
          <Icon name="search" size={14} style={{ color: 'var(--text-muted)' }} />
          <input
            id="global-search-input"
            type="text"
            className="c2-search-input"
            placeholder="Search cameras, plates, alerts, officers..."
            aria-label="Global search"
          />
          <kbd className="c2-search-kbd">âŒ˜K</kbd>
        </div>

        {/* Right Telemetry & Session Actions */}
        <div className="topnav-actions">
          <div className="c2-status-pill" title="Network Telemetry Status">
            <span className="c2-status-dot" />
            <span>System OK</span>
          </div>

          <button
            className="c2-bell-btn"
            title="3 critical alerts pending review"
            onClick={() => setPage('alerts')}
            aria-label="Alerts"
          >
            <Icon name="bell" size={16} />
            <span className="c2-badge-count">3</span>
          </button>

          <div className="c2-push-toggle" title="Push alert distribution active">
            <Icon name="shield" size={13} style={{ color: 'var(--status-success)' }} />
            <span>Push alerts: ON</span>
          </div>

          <div className="topnav-logo-divider" />

          <div className="c2-user-chip" title={`Operator: ${user?.username || 'admin'}`}>
            <div className="c2-user-avatar">
              <Icon name="user" size={13} />
            </div>
            <span className="c2-user-name">{user?.username || 'admin'}</span>
            <span className="c2-level-tag">LEVEL 3</span>
          </div>

          <button className="btn-logout" onClick={logout} id="btn-sign-out" title="Sign out of current terminal session">
            <Icon name="logout" size={13} />Sign Out
          </button>
        </div>
      </header>

      {/* â”€â”€ Body â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <div className={`dashboard-body ${navCollapsed ? 'nav-collapsed' : ''}`}>
        {/* Resizable Sidebar (200px - 360px) */}
        <nav
          className="sidenav"
          aria-label="Primary"
          style={{ width: navCollapsed ? 56 : sidebarWidth }}
        >
          {!navCollapsed && (
            <div
              className={`sidenav-resizer ${isDraggingNav ? 'is-dragging' : ''}`}
              onMouseDown={startResizeNav}
              title="Drag to resize sidebar (200px - 360px)"
            />
          )}

          {navGroups.map((group, gi) => (
            <div className="sidenav-group" key={group.section}>
              {navCollapsed
                ? (gi > 0 && <div className="sidenav-rule" />)
                : <div className="sidenav-label">{group.section}</div>}
              {group.items.map(item => (
                <button
                  key={item.id}
                  id={`nav-${item.id}`}
                  className={`sidenav-item ${page === item.id ? 'active' : ''}`}
                  onClick={() => setPage(item.id)}
                  title={item.label}
                  aria-current={page === item.id ? 'page' : undefined}
                >
                  <div className="sidenav-item-left">
                    <Icon name={item.icon} size={16} />
                    <span className="sidenav-item-label">{item.label}</span>
                  </div>
                  {item.badge && !navCollapsed && (
                    <span className={`sidenav-item-badge ${item.badgeClass || ''}`}>
                      {item.badge}
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}

          <button
            className="sidenav-toggle"
            onClick={toggleNav}
            title={navCollapsed ? 'Expand navigation' : 'Collapse navigation (âŒ˜B)'}
            aria-expanded={!navCollapsed}
          >
            <Icon name="panelLeft" size={15} />
            <span className="sidenav-item-label">Collapse</span>
          </button>
        </nav>

        {/* Main */}
        <main className={`main-content ${(page === 'control' || page === 'map') ? 'c2-control-main' : ''}`}>
          {page === 'map' && (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <div className="c2-panel-header" style={{ padding: '0 14px', height: 36 }}>
                <div className="c2-panel-header-left">
                  <Icon name="map" size={15} style={{ color: 'var(--accent-primary, #2563EB)' }} />
                  <span className="c2-panel-title" style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.6px' }}>GUJARAT SURVEILLANCE FLEET Â· GIS MAPPING</span>
                  <span className="c2-panel-badge">TACTICAL COMMAND</span>
                </div>
              </div>
              <CameraMap cameras={cameras} />
            </div>
          )}

          {page === 'cameras' && isAdmin && (
            <div>
              <div className="page-title"><Icon name="cameraAdd" size={20} />Camera Registry</div>
              <CameraOnboardingForm onCameraAdded={onCameraAdded} />
              <BulkCameraImport onImported={onCamerasBulkImported} />
              <GapAnalysisPanel />
              <div style={{ marginTop: 24 }}>
                <div className="card-title" style={{ marginBottom: 12 }}>Registered Cameras ({cameras.length})</div>
                {cameras.length === 0 ? (
                  <div className="empty-state">
                    <Icon name="camera" size={34} className="empty-state-icon" />
                    No cameras added yet
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {cameras.map(cam => (
                      <div key={cam.id} style={{
                        display: 'flex', alignItems: 'center', gap: 12,
                        padding: '10px 14px', borderRadius: 'var(--radius-md)',
                        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                        boxShadow: 'var(--ring-top)'
                      }}>
                        <Icon name="camera" size={16} style={{ color: 'var(--text-muted)' }} />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 600 }}>{cam.name}</div>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            {cam.zone || 'No zone'} Â· {cam.camera_type || 'Fixed'} Â· {cam.protocol} Â· {cam.ip_address || 'No IP'}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{ fontSize: 11, padding: '3px 8px' }}
                          onClick={() => onToggleMaintenance(cam)}
                        >
                          {cam.maintenance_mode ? 'âœ… Clear maintenance' : 'ðŸ”§ Maintenance'}
                        </button>
                        <span className={`badge badge-${cam.status?.toLowerCase()}`}>{cam.status}</span>
                        <span className={`badge badge-${cam.risk_level?.toLowerCase()}`}>{cam.risk_level}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {page === 'top10' && (
            <div style={{ height: '100%', overflow: 'auto' }}>
              <Top10CommandCentre />
            </div>
          )}

          {page === 'evaluation' && (
            <div>
              <div className="page-title">
                <Icon name="crosshair" size={20} />Trace Designated Vehicle
              </div>
              <EvaluationMode />
            </div>
          )}

          {page === 'watchlist' && isAdmin && (
            <div>
              <div className="page-title"><Icon name="shieldAlert" size={20} />Vehicle Watchlist</div>
              <WatchlistManager />
            </div>
          )}

          {page === 'heatmap' && (
            <div>
              <div className="page-title">
                <Icon name="flame" size={20} />City Traffic Heatmap
              </div>
              <div className="page-subtitle">
                Vehicle volume per camera node across the network, from
                one-minute rollups.
              </div>
              <TrafficHeatmap />
            </div>
          )}

          {page === 'live-studio' && (
            <div style={{ height: '100%', overflowY: 'auto', padding: '16px 20px' }}>
              <LiveCalibrationStudio />
            </div>
          )}

          {page === 'reid-proof' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <div className="page-title">
                <Icon name="searchCheck" size={20} />Cross-Camera Re-ID
              </div>
              <div className="page-subtitle">
                One motorcycle, filmed passing four points on four phones. Play
                the clips: the identity on the rider does not change.
              </div>
              <div className="card" style={{ maxWidth: 1100 }}>
                <ReIDProofPanel />
              </div>
            </div>
          )}

          {page === 'trajectory-proof' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <div className="page-title">
                <Icon name="route" size={20} />Trajectory Proof
              </div>
              <div className="page-subtitle">
                The route on a map, and under every pin the frame that camera
                actually recorded. Same vehicle, four places, in time order.
              </div>
              <div className="card" style={{ maxWidth: 1150 }}>
                <TrajectoryProofPanel />
              </div>
            </div>
          )}

          {page === 'trajectory-engine' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <div className="page-title">
                <Icon name="scan" size={20} />Trajectory Engine
              </div>
              <div className="page-subtitle">
                The six stages that turn a plate into a route, each reporting
                what it can do against the live database right now â€” including
                the stages that do not yet have the data to claim a number.
              </div>
              <div className="card" style={{ maxWidth: 1100 }}>
                <TrajectoryEnginePanel />
              </div>
            </div>
          )}

          {page === 'edge' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <div className="page-title">
                <Icon name="cpu" size={20} />Edge Inference
              </div>
              <div className="page-subtitle">
                Nodes that run detection, tracking and plate reading beside the
                camera and send only the result. The figures below are counted
                from the requests those nodes actually made.
              </div>
              <div className="card" style={{ maxWidth: 720 }}>
                <EdgeNodesPanel height="auto" />
              </div>
            </div>
          )}

          {page === 'fleet' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <FleetOperations />
            </div>
          )}

          {page === 'analytics' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <AnalyticsDashboard />
            </div>
          )}

          {page === 'control' && (
            // Full-height, no page-title strip: the control room is meant to
            // fill the viewport, and a heading would push the bottom row of
            // the grid below the fold on a 1080p screen.
            <div style={{ height: '100%', minHeight: 0 }}>
              <ControlRoom
                liveAlert={liveAlert}
                statusEvent={alertStatusEvent}
                mergedEvent={mergedEvent}
                zoneIncidentEvent={zoneIncidentEvent}
                routingEvent={routingEvent}
                onNavigate={setPage}
              />
            </div>
          )}

          {page === 'officer-pwa' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <OfficerApp />
            </div>
          )}

          {page === 'alerts' && (
            <div>
              <div className="page-title"><Icon name="bell" size={20} />Alert Feed</div>
              <UnroutedBanner refreshSignal={routingEvent} />
              <OfficerPanel refreshSignal={routingEvent} />
              <div className="card">
                <AlertFeed
                  liveAlert={liveAlert}
                  statusEvent={alertStatusEvent}
                  mergedEvent={mergedEvent}
                  zoneIncidentEvent={zoneIncidentEvent}
                  routingEvent={routingEvent}
                />
              </div>
            </div>
          )}

          {page === 'iq' && (
            <div>
              <div className="page-title"><Icon name="cpu" size={20} />Sarvanetra IQ â€” Threat Scoring</div>
              <div className="card">
                <SentinelIQCard liveAlert={liveAlert} statusEvent={alertStatusEvent} />
              </div>
            </div>
          )}

          {page === 'reid-queue' && (
            <div>
              <div className="page-title"><Icon name="searchCheck" size={20} />Person ReID Review Queue</div>
              <div className="card">
                <ReviewQueue wsEvents={wsEvent} />
              </div>
            </div>
          )}

          {/* â”€â”€ Gap 3: Active Learning Operator UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          {/* Officer HITL Review â€” server-authoritative 4s gate, confidence    */}
          {/* hidden until after decision, multi-tab lock, PII blur on alt+tab. */}
          {page === 'active-review' && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <ActiveReviewPage />
            </div>
          )}

          {/* Supervisor Escalation Queue â€” cases where two officers disagreed. */}
          {/* Admins only. SLA age tracking with amber/red colour coding.       */}
          {page === 'escalations' && isAdmin && (
            <div style={{ height: '100%', overflowY: 'auto' }}>
              <SupervisorEscalationQueue />
            </div>
          )}

          {page === 'journeys' && (
            <div style={{ height: '100%', minHeight: 0 }}>
              <JourneyView />
            </div>
          )}

          {page === 'validation' && isAdmin && (
            <div>
              <div className="page-title"><Icon name="clipboardCheck" size={20} />ReID Validation Report</div>
              <div className="card">
                <ValidationReportViewer />
              </div>
            </div>
          )}

          {page === 'audit' && isAdmin && (
            <div>
              <div className="page-title"><Icon name="list" size={20} />Audit Log</div>
              <div className="card">
                <AuditLogPanel />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

// â”€â”€ Root: auth guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function App() {
  const { token } = useAuth();
  return token ? <Dashboard /> : <LoginForm />;
}
