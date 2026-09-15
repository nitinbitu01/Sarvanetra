// frontend/src/components/SystemHealthBadge.jsx
import { useEffect, useState } from 'react';

const API = import.meta.env.VITE_API_URL || '/api/v1';

export default function SystemHealthBadge() {
  const [health, setHealth] = useState(null);

  const poll = async () => {
    try {
      const res = await fetch(`${API}/health`);
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth({ status: 'degraded' });
    }
  };

  useEffect(() => {
    poll();
    const id = setInterval(poll, 30_000);
    return () => clearInterval(id);
  }, []);

  const cls = health ? (health.status === 'ok' ? 'ok' : 'degraded') : 'unknown';
  const label = health
    ? health.status === 'ok' ? 'System OK' : 'Degraded'
    : 'Checking…';

  return (
    <div
      title={health ? `DB: ${health.db ?? '?'} | WS: ${health.websocket ?? '?'} | ENV: ${health.env ?? '?'}` : ''}
      style={{ display: 'flex', alignItems: 'center', fontSize: 12, color: 'var(--text-secondary)', cursor: 'default' }}
    >
      <span className={`health-dot ${cls}`} />
      {label}
    </div>
  );
}
