// frontend/src/components/PanelErrorBoundary.jsx
//
// Contains a crash to ONE panel. The acceptance bar for the control room is
// "no broken panels": a failing data source shows an inline message and every
// other panel stays live. Without a boundary per panel, one component
// throwing during render unmounts the entire dashboard tree — which during a
// demo looks like the whole product died.
import React from 'react';
import { API } from '../context/AuthContext';

export class PanelErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error(
      `[PanelErrorBoundary] ${this.props.panelName} failed:`,
      error?.message,
      '\ncomponent stack:', info?.componentStack?.slice(0, 400),
    );

    // Unauthenticated by design on the server: a panel can crash before auth
    // settles, and a reporter that needs a session cannot report that case.
    fetch(`${API}/client-error`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        panel: this.props.panelName,
        error: error?.message?.slice(0, 300),
        stack: error?.stack?.slice(0, 500),
        component: info?.componentStack?.slice(0, 300),
        timestamp: new Date().toISOString(),
      }),
    }).catch(() => {
      // Swallowed deliberately: an error reporter that can itself throw turns
      // one broken panel into a broken app.
    });
  }

  handleRetry = () => this.setState({ hasError: false, error: null });

  render() {
    if (this.state.hasError) {
      return (
        <div className="panel-error" title={this.state.error?.message}>
          <div>⚠ {this.props.panelName} unavailable</div>
          {import.meta.env.DEV && this.state.error?.message && (
            <div style={{ fontSize: 11, color: 'var(--accent-red, #ef4444)' }}>
              {this.state.error.message}
            </div>
          )}
          {/* Retry rather than requiring a full reload: most panel failures
              here are a transient fetch, and reloading the page would also
              drop every other panel's live state. */}
          <button
            onClick={this.handleRetry}
            style={{
              alignSelf: 'flex-start', marginTop: 2, padding: '3px 9px',
              fontSize: 11, borderRadius: 6, cursor: 'pointer',
              border: '1px solid var(--border, #334155)',
              background: 'var(--bg-elevated, #1e293b)',
              color: 'var(--text-muted, #94a3b8)',
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export function PanelSkeleton({ lines = 4 }) {
  return (
    <div className="panel-content">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="skeleton-line" style={{ width: `${100 - i * 8}%` }} />
      ))}
    </div>
  );
}

export default PanelErrorBoundary;
