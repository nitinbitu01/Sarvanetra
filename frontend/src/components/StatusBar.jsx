/**
 * StatusBar.jsx — Connection status and stats for the sidebar.
 *
 * Props:
 *   wsState   : "connecting" | "connected" | "reconnecting" | "disconnected"
 *   trackCount: number of currently active tracks
 *   cameraId  : string camera identifier from backend
 */

export function StatusBar({ wsState, trackCount, cameraId }) {
  const badge = {
    connected:    { label: 'LIVE',         cls: 'connected',    pulse: true  },
    connecting:   { label: 'Connecting…',  cls: 'reconnecting', pulse: true  },
    reconnecting: { label: 'Reconnecting', cls: 'reconnecting', pulse: true  },
    disconnected: { label: 'Offline',      cls: 'disconnected', pulse: false },
  }[wsState] ?? { label: wsState, cls: 'disconnected', pulse: false };

  return (
    <>
      {/* Connection status */}
      <div className="sidebar-section">
        <div className="sidebar-title">Connection</div>
        <div className={`connection-badge ${badge.cls}`}>
          <span className={`ws-dot ${badge.pulse ? 'pulse' : ''}`} />
          {badge.label}
        </div>
      </div>

      {/* Stats */}
      <div className="sidebar-section">
        <div className="sidebar-title">Session Stats</div>
        <div className="status-grid">
          <div className="stat-card">
            <div className="stat-label">Active Tracks</div>
            <div className={`stat-value ${trackCount > 0 ? 'blue' : 'muted'}`}>
              {trackCount}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Camera</div>
            <div className="stat-value amber" style={{ fontSize: 13, paddingTop: 4 }}>
              {cameraId ?? '—'}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
