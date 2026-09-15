// frontend/src/components/SafeHlsPlayer.jsx
import React from 'react';
import { HlsPlayer } from './HlsPlayer';

class HlsErrorBoundary extends React.Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('[HlsPlayer] Render error — stream unavailable:', error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          className="stream-error"
          role="alert"
          style={{
            padding: 16,
            background: 'var(--bg-elevated)',
            border: '1px dashed var(--border)',
            borderRadius: 8,
            textAlign: 'center',
            color: 'var(--text-muted)',
          }}
        >
          <p style={{ margin: 0, fontWeight: 600, color: 'var(--accent-yellow)' }}>⚠ Stream unavailable</p>
          <small style={{ fontSize: 11 }}>Camera feed could not be loaded. Check ffmpeg and stream_url.</small>
        </div>
      );
    }
    return this.props.children;
  }
}

export function SafeHlsPlayer(props) {
  return (
    <HlsErrorBoundary>
      <HlsPlayer {...props} />
    </HlsErrorBoundary>
  );
}

export default SafeHlsPlayer;
