/**
 * TrackList.jsx — Sidebar list of currently active confirmed tracks.
 *
 * Props:
 *   tracks : object from useWebSocket — { [track_id]: {track_id, confidence, ...} }
 */

const TRACK_PALETTE = [
  '#3b82f6', '#06b6d4', '#22c55e', '#f59e0b',
  '#a855f7', '#ec4899', '#f97316', '#14b8a6',
];

function trackColor(trackId) {
  return TRACK_PALETTE[trackId % TRACK_PALETTE.length];
}

export function TrackList({ tracks }) {
  const trackList = Object.values(tracks).sort((a, b) => a.track_id - b.track_id);

  return (
    <div className="sidebar-section">
      <div className="sidebar-title">Active Tracks</div>

      {trackList.length === 0 ? (
        <div className="track-empty">
          No confirmed tracks<br />
          <span style={{ fontSize: 10 }}>
            Tracks appear after 3 consecutive detections
          </span>
        </div>
      ) : (
        <div className="track-list">
          {trackList.map(t => (
            <div key={t.track_id} className="track-item">
              <span
                className="track-color-dot"
                style={{ background: trackColor(t.track_id) }}
              />
              <span className="track-id-label">Track {t.track_id}</span>
              <span className="track-conf">
                {(t.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
