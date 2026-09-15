// frontend/src/components/LiveStreamIngestBar.jsx
//
// Sentinel Gujarat — Live RTSP / Network Stream Ingestion Bar
// Allows operators and hackathon jury to enter any live RTSP / HTTP CCTV link
// and immediately start live AI detection, tracking, and physical speed analysis.

import React, { useState, useEffect } from 'react';
import { apiFetch } from '../utils/authClient';

const PRESET_STREAMS = [
  { id: 'CAM_04', name: 'Paldi Circle (Ahmedabad)', type: 'RTSP/CCTV' },
  { id: 'CAM_01', name: 'Chimanbhai Bridge (Ahmedabad)', type: 'RTSP/CCTV' },
  { id: 'CAM_02', name: 'Janpath Junction (Ahmedabad)', type: 'RTSP/CCTV' },
  { id: 'CAM_08', name: 'Majewadi Gate (Junagadh)', type: 'RTSP/CCTV' },
  { id: 'CAM_10', name: 'Char Chowk (Junagadh)', type: 'RTSP/CCTV' },
];

export default function LiveStreamIngestBar({
  selectedCameraId = 'CAM_04',
  onCameraChange,
  onStreamConnected,
}) {
  const [targetCamId, setTargetCamId] = useState(selectedCameraId);
  const [streamUrl, setStreamUrl] = useState('');
  const [cameraName, setCameraName] = useState('');
  const [connecting, setConnecting] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);
  // Short-lived, camera-scoped token for the <img> video URLs.
  const [streamToken, setStreamToken] = useState(null);
  const [streamTokenError, setStreamTokenError] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [showLiveFeed, setShowLiveFeed] = useState(false);

  // Sync with parent selection
  useEffect(() => {
    if (selectedCameraId) {
      setTargetCamId(selectedCameraId);
    }
  }, [selectedCameraId]);

  // Fetch telemetry for selected camera
  useEffect(() => {
    if (!targetCamId) return;
    let active = true;

    const fetchTelemetry = async () => {
      try {
        const res = await apiFetch(`/analytics/live/stream-telemetry/${targetCamId}`);
        if (res.ok && active) {
          const data = await res.json();
          setTelemetry(data);
        }
      } catch {
        // quiet fallback
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 3000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [targetCamId]);

  // Exchange the bearer token for a short-lived, camera-scoped stream token.
  //
  // The video endpoints require authentication now, and an <img> tag cannot
  // send an Authorization header — so the token travels in the URL instead.
  // It is refreshed before it expires; the server issues them for 10 minutes
  // and this renews at 8, so the picture never drops mid-view.
  useEffect(() => {
    if (!targetCamId) return;
    let active = true;

    const fetchToken = async () => {
      try {
        const res = await apiFetch(`/analytics/live/stream-token/${targetCamId}`);
        if (!active) return;
        if (!res.ok) {
          setStreamTokenError(
            res.status === 401 || res.status === 403
              ? 'Not authorised to view this camera'
              : 'Live feed unavailable',
          );
          setStreamToken(null);
          return;
        }
        const data = await res.json();
        setStreamToken(data.token);
        setStreamTokenError(null);
      } catch {
        if (active) setStreamTokenError('Live feed unavailable');
      }
    };

    fetchToken();
    const renew = setInterval(fetchToken, 8 * 60 * 1000);
    return () => {
      active = false;
      clearInterval(renew);
    };
  }, [targetCamId]);

  const handleConnect = async (e) => {
    if (e) e.preventDefault();
    if (!targetCamId) {
      setStatusMsg({ type: 'error', text: 'Please select or enter a Camera ID' });
      return;
    }

    setConnecting(true);
    setStatusMsg(null);

    try {
      const payload = {
        camera_id: targetCamId,
        stream_url: streamUrl.trim() || `data/clips/${targetCamId}/${targetCamId}_0100.mp4`,
        name: cameraName.trim() || undefined,
      };

      const res = await apiFetch('/analytics/live/connect-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to connect stream');
      }

      const data = await res.json();
      setStatusMsg({
        type: 'success',
        text: `✅ Connected! Ingesting live feed: ${data.stream_url_masked} (${data.source_type.toUpperCase()})`,
      });

      if (onStreamConnected) onStreamConnected(data);
      if (onCameraChange) onCameraChange(targetCamId);
      setShowLiveFeed(true);
    } catch (err) {
      setStatusMsg({ type: 'error', text: `❌ ${err.message}` });
    } finally {
      setConnecting(false);
    }
  };

  const handlePresetClick = (presetId) => {
    setTargetCamId(presetId);
    setStreamUrl('');
    if (onCameraChange) onCameraChange(presetId);
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%)',
      border: '1px solid rgba(56, 189, 248, 0.25)',
      borderRadius: 14,
      padding: '16px 20px',
      boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.4)',
      backdropFilter: 'blur(12px)',
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
      width: '100%',
    }}>
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 20 }}>🔴</span>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.01em' }}>
              Live CCTV Stream Ingestion &amp; Real-time AI Connector
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>
              Enter any live RTSP / ONVIF / HTTP network CCTV stream to start instant vehicle detection &amp; speed analysis
            </div>
          </div>
        </div>

        {/* Status badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {telemetry && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: telemetry.is_connected ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              border: `1px solid ${telemetry.is_connected ? '#10b981' : '#ef4444'}`,
              borderRadius: 20,
              padding: '4px 12px',
              fontSize: 11,
              fontWeight: 700,
              color: telemetry.is_connected ? '#34d399' : '#f87171',
            }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: telemetry.is_connected ? '#10b981' : '#ef4444',
                boxShadow: telemetry.is_connected ? '0 0 8px #10b981' : 'none',
              }} />
              {telemetry.is_connected ? `ONLINE (${(telemetry.source_type || 'RTSP').toUpperCase()})` : 'OFFLINE'}
            </div>
          )}

          <button
            onClick={() => setShowLiveFeed(!showLiveFeed)}
            style={{
              background: showLiveFeed ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)',
              border: `1px solid ${showLiveFeed ? '#38bdf8' : 'rgba(255, 255, 255, 0.1)'}`,
              borderRadius: 8,
              padding: '5px 12px',
              fontSize: 12,
              fontWeight: 600,
              color: showLiveFeed ? '#38bdf8' : '#cbd5e1',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              transition: 'all 0.2s',
            }}
          >
            <span>📹</span>
            {showLiveFeed ? 'Hide Live Stream' : 'View Live Stream Output'}
          </button>
        </div>
      </div>

      {/* Ingestion Input Form */}
      <form onSubmit={handleConnect} style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        {/* Camera Selector */}
        <select
          value={targetCamId}
          onChange={(e) => {
            setTargetCamId(e.target.value);
            if (onCameraChange) onCameraChange(e.target.value);
          }}
          style={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: 8,
            color: '#f8fafc',
            padding: '8px 12px',
            fontSize: 13,
            fontWeight: 600,
            outline: 'none',
            minWidth: 160,
          }}
        >
          {PRESET_STREAMS.map(p => (
            <option key={p.id} value={p.id}>
              {p.id} — {p.name}
            </option>
          ))}
          <option value="CAM_CUSTOM">+ Custom Jury Camera</option>
        </select>

        {/* RTSP / HTTP URL Input Field */}
        <div style={{ flex: 1, minWidth: 260, position: 'relative' }}>
          <input
            type="text"
            placeholder="Paste Live RTSP / HTTP link e.g. rtsp://admin:pass@192.168.1.50:554/live or http://..."
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            style={{
              width: '100%',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: 8,
              color: '#f8fafc',
              padding: '8px 14px',
              fontSize: 13,
              fontFamily: 'monospace',
              outline: 'none',
              transition: 'border-color 0.2s',
            }}
            onFocus={(e) => (e.target.style.borderColor = '#38bdf8')}
            onBlur={(e) => (e.target.style.borderColor = 'rgba(255, 255, 255, 0.15)')}
          />
        </div>

        {/* Connect Action Button */}
        <button
          type="submit"
          disabled={connecting}
          style={{
            background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
            border: '1px solid #38bdf8',
            borderRadius: 8,
            color: '#ffffff',
            padding: '8px 18px',
            fontSize: 13,
            fontWeight: 700,
            cursor: connecting ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)',
            transition: 'all 0.2s',
            opacity: connecting ? 0.7 : 1,
          }}
        >
          {connecting ? (
            <>
              <span className="spinner" style={{ width: 14, height: 14, border: '2px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 1s linear infinite' }} />
              Connecting...
            </>
          ) : (
            <>
              <span>⚡</span>
              Start Live AI Analysis
            </>
          )}
        </button>
      </form>

      {/* Quick Presets Strip */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', fontSize: 12 }}>
        <span style={{ color: '#64748b', fontWeight: 600 }}>Quick Presets:</span>
        {PRESET_STREAMS.map(p => (
          <button
            key={p.id}
            onClick={() => handlePresetClick(p.id)}
            style={{
              background: targetCamId === p.id ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255, 255, 255, 0.03)',
              border: `1px solid ${targetCamId === p.id ? '#38bdf8' : 'rgba(255, 255, 255, 0.08)'}`,
              borderRadius: 6,
              padding: '3px 8px',
              fontSize: 11,
              color: targetCamId === p.id ? '#38bdf8' : '#94a3b8',
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {p.name.split(' ')[0]} ({p.id})
          </button>
        ))}
      </div>

      {/* Status Feedback Message */}
      {statusMsg && (
        <div style={{
          padding: '8px 12px',
          borderRadius: 6,
          fontSize: 12,
          fontWeight: 600,
          background: statusMsg.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          border: `1px solid ${statusMsg.type === 'success' ? '#10b981' : '#ef4444'}`,
          color: statusMsg.type === 'success' ? '#34d399' : '#f87171',
        }}>
          {statusMsg.text}
        </div>
      )}

      {/* Live Stream MJPEG Visualizer (When toggled open) */}
      {showLiveFeed && (
        <div style={{
          marginTop: 6,
          borderRadius: 10,
          overflow: 'hidden',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          background: '#000',
          position: 'relative',
        }}>
          <div style={{
            position: 'absolute',
            top: 10,
            left: 10,
            background: 'rgba(0,0,0,0.75)',
            padding: '4px 10px',
            borderRadius: 6,
            fontSize: 11,
            color: '#38bdf8',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            zIndex: 10,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 6px #10b981' }} />
            LIVE CCTV STREAM · {targetCamId} · REAL-TIME YOLOv8 DETECTION OVERLAY
          </div>

          {/* The video routes now require authentication. An <img> cannot send
            * an Authorization header, so the component first exchanges its
            * bearer token for a short-lived token scoped to this one camera
            * and puts that in the URL. Until it arrives, streamToken is null
            * and no request is made — rendering the bare URL would just draw a
            * broken image. */}
          {streamToken ? (
            <img
              src={`/api/v1/analytics/live/stream/${targetCamId}?token=${encodeURIComponent(streamToken)}`}
              alt={`Live Stream ${targetCamId}`}
              style={{ width: '100%', maxHeight: 420, objectFit: 'contain', display: 'block' }}
              onError={(e) => {
                // fallback to frame snapshot, same token
                e.target.src = `/api/v1/analytics/live/frame/${targetCamId}`
                  + `?token=${encodeURIComponent(streamToken)}&t=${Date.now()}`;
              }}
            />
          ) : (
            <div style={{
              height: 240, display: 'flex', alignItems: 'center',
              justifyContent: 'center', color: '#64748b', fontSize: 13,
            }}>
              {streamTokenError || 'Authorising live feed…'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
