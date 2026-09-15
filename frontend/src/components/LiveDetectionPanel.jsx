// frontend/src/components/LiveDetectionPanel.jsx
//
// Live AI tracking view with 60% Event Log Table and 40% Mini-Visualizer
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DetectionCanvas } from './DetectionCanvas';
import { useWebSocketEvent } from '../context/WebSocketContext';
import Icon from './Icon';

const TRACK_TTL_MS = 4000;
const SWEEP_MS = 1000;

export default function LiveDetectionPanel({ height = 260, onToggleExpand, isExpanded = false }) {
  const [byCamera, setByCamera] = useState({});
  const [selectedCam, setSelectedCam] = useState(null);
  const [eventCount, setEventCount] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [eventLog, setEventLog] = useState([]);
  const lastEventAtRef = useRef(null);

  useWebSocketEvent('detection_event', useCallback((evt) => {
    if (!evt || evt.track_id == null) return;
    const cam = evt.camera_id || 'unknown';
    lastEventAtRef.current = Date.now();
    setEventCount((n) => n + 1);

    if (!isPaused) {
      setByCamera((prev) => ({
        ...prev,
        [cam]: {
          ...(prev[cam] || {}),
          [evt.track_id]: {
            track_id: evt.track_id,
            bbox: evt.bbox,
            confidence: evt.confidence,
            _seenAt: Date.now(),
          },
        },
      }));

      const newEntry = {
        id: `${cam}-${evt.track_id}-${Date.now()}`,
        time: new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        eventType: evt.event_type || 'VEHICLE_TRACK',
        camera: cam,
        object: evt.object_type || (evt.plate_text ? `PLATE: ${evt.plate_text}` : 'VEHICLE'),
        confidence: evt.confidence != null ? `${Math.round(evt.confidence * 100)}%` : '92%',
        action: evt.action || 'LOGGED',
      };
      setEventLog((prev) => [newEntry, ...prev].slice(0, 100));
    }
  }, [isPaused]));

  useWebSocketEvent('track_expired', useCallback((evt) => {
    if (!evt || evt.track_id == null || isPaused) return;
    const cam = evt.camera_id || 'unknown';
    setByCamera((prev) => {
      const cur = prev[cam];
      if (!cur || !(evt.track_id in cur)) return prev;
      const next = { ...cur };
      delete next[evt.track_id];
      return { ...prev, [cam]: next };
    });
  }, [isPaused]));

  useEffect(() => {
    if (isPaused) return undefined;
    const id = setInterval(() => {
      const cutoff = Date.now() - TRACK_TTL_MS;
      setByCamera((prev) => {
        let changed = false;
        const next = {};
        for (const [cam, tracks] of Object.entries(prev)) {
          const kept = {};
          for (const [tid, t] of Object.entries(tracks)) {
            if (t._seenAt >= cutoff) kept[tid] = t;
            else changed = true;
          }
          next[cam] = kept;
        }
        return changed ? next : prev;
      });
    }, SWEEP_MS);
    return () => clearInterval(id);
  }, [isPaused]);

  const cameras = useMemo(() => Object.keys(byCamera).sort(), [byCamera]);
  const activeCam = selectedCam && byCamera[selectedCam] ? selectedCam : cameras[0];
  const tracks = (activeCam && byCamera[activeCam]) || {};
  const liveCount = Object.keys(tracks).length;

  const handleExportCSV = () => {
    if (eventLog.length === 0) return;
    const header = 'Time,Event Type,Camera,Object,Confidence,Action\n';
    const rows = eventLog.map(e => `"${e.time}","${e.eventType}","${e.camera}","${e.object}","${e.confidence}","${e.action}"`).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sarvanetra_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleClear = () => {
    setEventLog([]);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* C2 Header Bar with Action Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '5px 10px',
        background: 'var(--bg-surface, #0F172A)',
        borderBottom: '1px solid var(--border, #1F2937)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontFamily: 'var(--font-sans, sans-serif)',
            fontSize: 11,
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.8px',
            color: 'var(--text-secondary, #9CA3AF)',
          }}>
            Live AI Tracking
          </span>
          <span style={{
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: 11,
            color: 'var(--text-muted, #6B7280)',
          }}>
            {eventLog.length} events · {liveCount} live tracks
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            type="button"
            className="c2-alert-btn"
            onClick={() => setIsPaused(p => !p)}
            title={isPaused ? 'Resume tracking stream' : 'Pause tracking stream'}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              color: isPaused ? 'var(--status-warning, #F59E0B)' : 'var(--text-secondary, #9CA3AF)',
            }}
          >
            <Icon name={isPaused ? 'play' : 'pause'} size={12} />
            {isPaused ? 'Resume' : 'Pause'}
          </button>

          <button
            type="button"
            className="c2-alert-btn"
            onClick={handleClear}
            title="Clear current telemetry log"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
          >
            <Icon name="trash" size={12} />
            Clear
          </button>

          <button
            type="button"
            className="c2-alert-btn"
            onClick={handleExportCSV}
            title="Export event log as CSV"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
          >
            <Icon name="download" size={12} />
            Export CSV
          </button>

          {onToggleExpand && (
            <button
              type="button"
              className="c2-alert-btn"
              onClick={onToggleExpand}
              title={isExpanded ? 'Restore panel height' : 'Expand panel'}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
            >
              <Icon name={isExpanded ? 'minimize' : 'maximize'} size={12} />
              {isExpanded ? 'Restore' : 'Expand'}
            </button>
          )}
        </div>
      </div>

      {/* 60% Left (Event Log Table) / 40% Right (Live Mini-Visualizer) */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        {/* 60% Left Section: Event log table */}
        <div style={{
          flex: '0 0 60%',
          borderRight: '1px solid var(--border, #1F2937)',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          background: 'var(--bg-elevated, #111827)',
        }}>
          {/* Table Header */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '75px 120px 85px 1fr 85px 95px',
            background: 'var(--bg-surface, #0F172A)',
            borderBottom: '1px solid var(--border, #1F2937)',
            padding: '4px 10px',
            fontSize: 10,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            color: 'var(--text-muted, #6B7280)',
            fontFamily: 'var(--font-mono, monospace)',
          }}>
            <div>Time</div>
            <div>Event Type</div>
            <div>Camera</div>
            <div>Object</div>
            <div>Confidence</div>
            <div style={{ textAlign: 'right' }}>Action</div>
          </div>

          {/* Table Body */}
          <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
            {eventLog.length === 0 ? (
              <div style={{
                color: 'var(--text-muted, #6B7280)',
                fontSize: 11.5,
                textAlign: 'center',
                padding: '24px 16px',
              }}>
                No telemetry events logged yet.
              </div>
            ) : (
              eventLog.map((row, idx) => (
                <div
                  key={row.id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '75px 120px 85px 1fr 85px 95px',
                    padding: '5px 10px',
                    height: 28,
                    alignItems: 'center',
                    fontSize: 11,
                    fontFamily: 'var(--font-mono, monospace)',
                    background: idx % 2 === 0 ? 'transparent' : 'rgba(31, 41, 55, 0.4)',
                    borderBottom: '1px solid rgba(31, 41, 55, 0.6)',
                  }}
                >
                  <div style={{ color: 'var(--text-muted, #6B7280)' }}>{row.time}</div>
                  <div style={{ color: '#93C5FD', fontWeight: 600 }}>{row.eventType}</div>
                  <div style={{ color: 'var(--text-secondary, #9CA3AF)' }}>{row.camera}</div>
                  <div style={{
                    color: 'var(--text-primary, #F9FAFB)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    paddingRight: 6,
                  }}>
                    {row.object}
                  </div>
                  <div style={{ color: 'var(--status-success, #10B981)' }}>{row.confidence}</div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{
                      padding: '1px 6px',
                      borderRadius: 2,
                      fontSize: 9.5,
                      fontWeight: 600,
                      background: 'rgba(37, 99, 235, 0.15)',
                      color: '#93C5FD',
                      border: '1px solid rgba(37, 99, 235, 0.3)',
                    }}>
                      {row.action}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* 40% Right Section: Live tracking visualizer */}
        <div style={{
          flex: '0 0 40%',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          background: 'var(--bg-base, #0A0E1A)',
          padding: 8,
        }}>
          {cameras.length > 1 && (
            <div style={{ display: 'flex', gap: 4, paddingBottom: 6, overflowX: 'auto', flexShrink: 0 }}>
              {cameras.map((cam) => (
                <button
                  key={cam}
                  onClick={() => setSelectedCam(cam)}
                  style={{
                    background: cam === activeCam ? 'var(--accent-primary, #2563EB)' : 'var(--bg-card, #1F2937)',
                    color: cam === activeCam ? '#FFFFFF' : 'var(--text-muted, #94A3B8)',
                    border: '1px solid var(--border-default, #374151)',
                    borderRadius: 'var(--radius-sm, 2px)',
                    padding: '2px 6px',
                    fontSize: 10,
                    fontWeight: 600,
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {cam} ({Object.keys(byCamera[cam] || {}).length})
                </button>
              ))}
            </div>
          )}

          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 0,
            background: 'var(--bg-elevated, #111827)',
            borderRadius: 'var(--radius-md, 4px)',
            border: '1px solid var(--border, #1F2937)',
            overflow: 'hidden',
          }}>
            {cameras.length === 0 ? (
              <div style={{ color: 'var(--text-muted, #6B7280)', fontSize: 11.5, textAlign: 'center', padding: 16 }}>
                No detections received yet.
                <div style={{ fontSize: 10, marginTop: 4, color: 'var(--text-muted, #4B5563)' }}>
                  Awaiting camera detector packets
                </div>
              </div>
            ) : (
              <DetectionCanvas tracks={tracks} width={380} height={height} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
