/**
 * useWebSocket.js — WebSocket hook for Sentinel Gujarat dashboard.
 *
 * Connects to ws://localhost:8000/ws/detections and handles:
 *   - "snapshot"        → initialises tracks from current backend state
 *   - "detection_event" → upserts a track by track_id
 *   - "track_expired"   → removes a track by track_id
 *
 * Auto-reconnect: on close, waits RECONNECT_DELAY_MS then reconnects.
 * Shows "reconnecting" state to the UI during the gap.
 *
 * Returns:
 *   { tracks, wsState, cameraId }
 *   tracks   : object keyed by track_id, value = {track_id, bbox, confidence, frame_number, timestamp}
 *   wsState  : "connecting" | "connected" | "reconnecting" | "disconnected"
 *   cameraId : string from the last snapshot/event, or null
 */

import { useEffect, useRef, useState, useCallback } from 'react';

const getWsUrl = (path) => {
  if (import.meta.env.VITE_WS_URL) return `${import.meta.env.VITE_WS_URL}${path}`;
  if (typeof window === 'undefined') return `ws://localhost:8000${path}`;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
};

const WS_URL = getWsUrl('/ws/detections');
const RECONNECT_DELAY_MS = 2000;

export function useWebSocket() {
  const [tracks, setTracks] = useState({});       // { [track_id]: event_data }
  const [wsState, setWsState] = useState('connecting');
  const [cameraId, setCameraId] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const unmountedRef = useRef(false);

  const connect = useCallback(() => {
    if (unmountedRef.current) return;

    setWsState('connecting');

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmountedRef.current) return;
      setWsState('connected');
      console.log('[WS] Connected to', WS_URL);
    };

    ws.onmessage = (evt) => {
      if (unmountedRef.current) return;
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        console.warn('[WS] Non-JSON message:', evt.data);
        return;
      }

      switch (msg.type) {
        case 'snapshot': {
          // Initialise from backend's current state on connect
          if (msg.camera_id) setCameraId(msg.camera_id);
          const initial = {};
          (msg.tracks || []).forEach(t => { initial[t.track_id] = t; });
          setTracks(initial);
          console.log(`[WS] Snapshot received: ${msg.tracks?.length ?? 0} active tracks`);
          break;
        }

        case 'detection_event': {
          if (msg.camera_id) setCameraId(msg.camera_id);
          setTracks(prev => ({
            ...prev,
            [msg.track_id]: {
              track_id:    msg.track_id,
              bbox:        msg.bbox,
              confidence:  msg.confidence,
              frame_number: msg.frame_number,
              timestamp:   msg.timestamp,
            },
          }));
          break;
        }

        case 'track_expired': {
          setTracks(prev => {
            const next = { ...prev };
            delete next[msg.track_id];
            return next;
          });
          console.log(`[WS] Track ${msg.track_id} expired — removed from canvas`);
          break;
        }

        default:
          console.debug('[WS] Unknown message type:', msg.type);
      }
    };

    ws.onerror = (err) => {
      console.warn('[WS] Error:', err);
    };

    ws.onclose = (evt) => {
      if (unmountedRef.current) return;
      console.log(`[WS] Closed (code=${evt.code}). Reconnecting in ${RECONNECT_DELAY_MS}ms...`);
      setWsState('reconnecting');

      // Clear tracks on disconnect so stale boxes don't linger
      // (server's expiry mechanism handles this during normal operation,
      //  but on full disconnect we clear immediately)
      setTracks({});

      reconnectTimer.current = setTimeout(() => {
        if (!unmountedRef.current) connect();
      }, RECONNECT_DELAY_MS);
    };
  }, []);  // stable reference — no deps

  useEffect(() => {
    unmountedRef.current = false;
    connect();

    return () => {
      unmountedRef.current = true;
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;  // prevent reconnect on intentional unmount
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { tracks, wsState, cameraId };
}
