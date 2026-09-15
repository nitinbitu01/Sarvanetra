// frontend/src/context/WebSocketContext.jsx
//
// A pub/sub layer over the ONE existing dashboard WebSocket.
//
// WHY THIS EXISTS
//   Before Day 18 every consumer of WS data had to be passed a callback prop
//   from App.jsx (onNewAlert, onRoutingEvent, onCameraStatusChanged …), and
//   App.jsx had to thread the resulting state down to each panel. That works
//   for five consumers; the control-room layout has twelve, several of them
//   three levels deep. The alternative to a context here is prop-drilling a
//   dozen callbacks through the grid.
//
//   This deliberately does NOT open a socket. useDashboardSocket already owns
//   the single connection, its auth handshake, and its reconnect loop —
//   duplicating that would give the app two sockets and two reconnect timers.
//   The provider subscribes once and re-broadcasts to local listeners.
//
// EVENT NAMES ARE NOT UNIFORM
//   This backend emits two conventions, because they were added on different
//   days: dotted (`alert.routed`, `officer.status`, from Day 14/15) and
//   underscored (`new_alert`, `camera_status_changed`, `alert_merged`, from
//   Days 5-11). Nothing renames them at the boundary — a translation layer
//   would mean the string in a component no longer matches the string in the
//   backend, and every future debugging session starts by discovering that.
//   Subscribe with the name the server actually sends.
import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from 'react';
import { useAuth } from './AuthContext';
import { useDashboardSocket } from '../hooks/useDashboardSocket';

export const WebSocketContext = createContext({
  connectionState: 'CONNECTING',
  subscribe: () => () => {},
  lastEventAt: null,
});

export function WebSocketProvider({ children }) {
  const { token } = useAuth();

  // Map<eventType, Set<handler>>. A ref, not state: adding a subscriber must
  // not re-render every other subscriber.
  const listenersRef = useRef(new Map());
  const [connectionState, setConnectionState] = useState('CONNECTING');
  const [lastEventAt, setLastEventAt] = useState(null);

  const emit = useCallback((type, payload) => {
    setLastEventAt(Date.now());
    const set = listenersRef.current.get(type);
    if (!set) return;
    for (const fn of Array.from(set)) {
      try {
        fn(payload);
      } catch (err) {
        // One panel's bad handler must not stop the other panels from
        // receiving the same event.
        console.error(`[WS] handler for "${type}" threw:`, err);
      }
    }
  }, []);

  const subscribe = useCallback((type, handler) => {
    const map = listenersRef.current;
    if (!map.has(type)) map.set(type, new Set());
    map.get(type).add(handler);
    return () => {
      const set = map.get(type);
      if (!set) return;
      set.delete(handler);
      if (set.size === 0) map.delete(type);
    };
  }, []);

  // Bridge every callback the existing hook exposes into the pub/sub bus,
  // preserving the exact server-side event names.
  const onNewAlert = useCallback((alert) => emit('new_alert', alert), [emit]);
  const onCameraAdded = useCallback((c) => emit('camera_added', c), [emit]);
  const onCameraStatusChanged = useCallback((e) => emit('camera_status_changed', e), [emit]);
  const onAlertStatusUpdate = useCallback((e) => emit('alert_status_update', e), [emit]);
  const onAlertMerged = useCallback((e) => emit('alert_merged', e), [emit]);
  const onZoneIncidentOpened = useCallback((e) => emit('zone_incident_opened', e), [emit]);
  const onZoneIncidentUpdated = useCallback((e) => emit('zone_incident_updated', e), [emit]);
  const onRoutingEvent = useCallback((e) => emit(e.type, e), [emit]);

  const { connected } = useDashboardSocket(token, {
    onCameraAdded,
    onNewAlert,
    onCameraStatusChanged,
    onAlertStatusUpdate,
    onAlertMerged,
    onZoneIncidentOpened,
    onZoneIncidentUpdated,
    onRoutingEvent,
  });

  useEffect(() => {
    setConnectionState((prev) => {
      if (connected) return 'CONNECTED';
      // Distinguish "never connected" from "dropped": the header should say
      // Connecting… on first load and Reconnecting… after a drop, because
      // those mean different things to an operator.
      return prev === 'CONNECTED' || prev === 'RECONNECTING'
        ? 'RECONNECTING' : 'CONNECTING';
    });
  }, [connected]);

  const value = useMemo(
    () => ({ connectionState, subscribe, lastEventAt }),
    [connectionState, subscribe, lastEventAt],
  );

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

/**
 * Subscribe to one server event type for the lifetime of a component.
 *
 * The handler is held in a ref so a caller passing an inline arrow function
 * does not resubscribe on every render — which would otherwise churn the
 * listener set dozens of times a second under load.
 */
export function useWebSocketEvent(eventType, handler) {
  const { subscribe } = useContext(WebSocketContext);
  const ref = useRef(handler);
  ref.current = handler;

  useEffect(() => {
    if (!eventType) return undefined;
    return subscribe(eventType, (payload) => ref.current?.(payload));
  }, [eventType, subscribe]);
}

export function useConnectionState() {
  return useContext(WebSocketContext).connectionState;
}
