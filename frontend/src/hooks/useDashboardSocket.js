// frontend/src/hooks/useDashboardSocket.js
import { useEffect, useRef, useCallback, useState } from 'react';

const getWsUrl = (path) => {
  if (import.meta.env.VITE_WS_URL) return `${import.meta.env.VITE_WS_URL}${path}`;
  if (typeof window === 'undefined') return `ws://localhost:8000${path}`;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
};

const WS_URL = getWsUrl('/ws/dashboard');

export function useDashboardSocket(token, {
  onCameraAdded,
  onNewAlert,
  onCameraStatusChanged,
  onAlertStatusUpdate,
  onAlertMerged,
  onZoneIncidentOpened,
  onZoneIncidentUpdated,
  onRoutingEvent,
} = {}) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);

  const connect = useCallback(() => {
    if (!token) return;
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ token }));
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'authenticated') setConnected(true);
        if (msg.type === 'camera_added' && onCameraAdded) onCameraAdded(msg.camera);
        if (msg.type === 'new_alert' && onNewAlert) onNewAlert(msg.alert);
        // Day 7: camera heartbeat status transitions (ONLINE/OFFLINE), pushed
        // only on actual change — see backend/services/camera_heartbeat.py
        if (msg.type === 'camera_status_changed' && onCameraStatusChanged) onCameraStatusChanged(msg);
        // Day 8: alert lifecycle transitions (acknowledge/dismiss/escalate) —
        // pushed to every connected operator so two people don't both act on
        // the same alert. See backend/routers/v1/alerts.py.
        if (msg.type === 'alert_status_update' && onAlertStatusUpdate) onAlertStatusUpdate(msg);
        // Day 11: Alert fatigue management (dedup + zone incidents)
        if (msg.type === 'alert_merged' && onAlertMerged) onAlertMerged(msg);
        if (msg.type === 'zone_incident_opened' && onZoneIncidentOpened) onZoneIncidentOpened(msg);
        if (msg.type === 'zone_incident_updated' && onZoneIncidentUpdated) onZoneIncidentUpdated(msg);
        // Day 14: alert routing. All five routing event types funnel through
        // one handler — the dashboard reacts to them in the same way (patch
        // per-alert routing state, refresh the officer roster and the
        // unrouted banner), so five near-identical callbacks would be five
        // places to forget to wire up.
        if (msg.type === 'alert.routed' || msg.type === 'alert.ack'
            || msg.type === 'alert.escalated' || msg.type === 'alert.unrouted'
            || msg.type === 'officer.status') {
          onRoutingEvent?.(msg);
        }
      } catch { /* ignore malformed */ }
    };

    ws.onclose = (ev) => {
      setConnected(false);
      // Reconnect if not a deliberate auth failure
      if (ev.code !== 4001 && token) {
        setTimeout(connect, 3000);
      }
    };

    ws.onerror = () => ws.close();
  }, [
    token, onCameraAdded, onNewAlert, onCameraStatusChanged,
    onAlertStatusUpdate, onAlertMerged, onZoneIncidentOpened, onZoneIncidentUpdated,
    onRoutingEvent,
  ]);

  useEffect(() => {
    connect();
    return () => { wsRef.current?.close(); };
  }, [connect]);

  return { connected };
}
