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

  componentDidUpdate(prevProps) {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, error: null });
    }
  }

  handleRetry = () => this.setState({ hasError: false, error: null });

  render() {
    if (this.state.hasError) {
      return (
        <div className="panel-error" style={{
          padding: 24, margin: '16px 0',
          background: 'var(--bg-elevated, #1e293b)',
          border: '1px solid var(--border, #334155)',
          borderRadius: 12,
        }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-red, #ef4444)', marginBottom: 8 }}>
            ⚠️ {this.props.panelName || 'Section'} encountered an error
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-muted, #94a3b8)', marginBottom: 16 }}>
            {this.state.error?.message || 'An unexpected rendering error occurred in this view.'}
          </div>
          <button
            onClick={this.handleRetry}
            style={{
              padding: '8px 16px',
              fontSize: 12, borderRadius: 6, cursor: 'pointer',
              border: '1px solid var(--border, #334155)',
              background: '#3b82f6',
              color: '#fff', fontWeight: 600,
            }}
          >
            🔄 Reload Section
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
