// frontend/src/components/WatchlistManager.jsx
//
// The searchable watchlist database itself — add, list, retire vehicle
// plates. This is the core hackathon requirement's missing control surface:
// live CCTV feeds are matched against watchlist_plates
// (backend/services/watchlist_service.py, called from the live ANPR
// pipeline) on a live poll, but until this component existed there was no
// way to add an entry except hand-editing seed data — no UI, no API. A
// department (or a hackathon judge's "designated vehicle") needs to add a
// plate and have the live pipeline see it within seconds, through the app.
import { useState, useEffect, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';

const CATEGORIES = ['stolen', 'wanted', 'suspect', 'blacklisted', 'other'];

export default function WatchlistManager() {
  const { authFetch } = useAuth();
  const [entries, setEntries] = useState(null);
  const [includeInactive, setIncludeInactive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [form, setForm] = useState({ plate: '', reason: '', category: 'stolen' });
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await authFetch(
        `${API}/plate-search/watchlist?include_inactive=${includeInactive}`);
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      setEntries(data.entries || []);
    } catch {
      setError('Could not load the watchlist.');
    } finally {
      setLoading(false);
    }
  }, [authFetch, includeInactive]);

  useEffect(() => { load(); }, [load]);

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!form.plate.trim()) return;
    setSaving(true);
    setSaveMsg('');
    try {
      const res = await authFetch(`${API}/plate-search/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Failed (${res.status})`);
      setSaveMsg(`✅ ${data.plate} is now on the watchlist — the live pipeline will start matching it within a few seconds.`);
      setForm({ plate: '', reason: '', category: 'stolen' });
      load();
    } catch (err) {
      setSaveMsg(`❌ ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const setActive = async (plate, active) => {
    try {
      const res = await authFetch(`${API}/plate-search/watchlist/${plate}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active }),
      });
      if (res.ok) load();
    } catch {
      // load() simply won't refresh; entry stays in its last-known state.
    }
  };

  const deleteEntry = async (plate) => {
    if (!window.confirm(`Permanently delete ${plate} from the watchlist? Use "Deactivate" instead if this case may return.`)) return;
    try {
      const res = await authFetch(`${API}/plate-search/watchlist/${plate}`, { method: 'DELETE' });
      if (res.ok || res.status === 204) load();
    } catch {
      // load() simply won't refresh; entry stays visible until the next successful call.
    }
  };

  return (
    <div className="card" style={{ maxWidth: 760 }}>
      <div className="card-header">
        <div className="card-title">🚨 Vehicle Watchlist</div>
      </div>

      <form onSubmit={handleAdd} style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16, alignItems: 'flex-end' }}>
        <div className="form-group" style={{ flex: '1 1 160px' }}>
          <label className="form-label">Plate Number *</label>
          <input className="form-input" placeholder="GJ05AB1234" value={form.plate}
            onChange={(e) => setForm(f => ({ ...f, plate: e.target.value.toUpperCase() }))}
            style={{ fontFamily: 'monospace' }} required />
        </div>
        <div className="form-group" style={{ flex: '2 1 240px' }}>
          <label className="form-label">Reason</label>
          <input className="form-input" placeholder="e.g. Stolen — FIR #2026/891" value={form.reason}
            onChange={(e) => setForm(f => ({ ...f, reason: e.target.value }))} />
        </div>
        <div className="form-group" style={{ flex: '0 1 150px' }}>
          <label className="form-label">Category</label>
          <select className="form-select" value={form.category}
            onChange={(e) => setForm(f => ({ ...f, category: e.target.value }))}>
            {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? '⏳ Adding…' : '+ Add to Watchlist'}
        </button>
      </form>

      {saveMsg && (
        <div className={`test-result ${saveMsg.startsWith('✅') ? 'success' : 'error'}`} style={{ marginBottom: 14 }}>
          {saveMsg}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" checked={includeInactive}
            onChange={(e) => setIncludeInactive(e.target.checked)} />
          Show retired entries
        </label>
        <button type="button" className="btn btn-secondary" onClick={load} disabled={loading}>
          {loading ? '⏳' : '↻'} Refresh
        </button>
      </div>

      {error && <div className="test-result error">{error}</div>}

      {entries && (
        entries.length === 0 ? (
          <div className="empty-state"><span className="empty-state-icon">🔍</span>No watchlist entries</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {entries.map((e) => (
              <div key={e.plate} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 12px', borderRadius: 6,
                background: e.active ? 'var(--bg-elevated)' : 'rgba(100,116,139,0.08)',
                border: '1px solid var(--border)', opacity: e.active ? 1 : 0.6,
              }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 700, minWidth: 100 }}>{e.plate}</span>
                <span style={{ flex: 1, fontSize: 12, color: 'var(--text-muted)' }}>{e.reason || '—'}</span>
                <span className={`badge badge-${e.category === 'stolen' ? 'high' : 'medium'}`}>{e.category}</span>
                {!e.active && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>retired</span>}
                {e.active ? (
                  <button type="button" className="btn btn-secondary" style={{ fontSize: 11, padding: '3px 8px' }}
                    onClick={() => setActive(e.plate, false)}>Deactivate</button>
                ) : (
                  <button type="button" className="btn btn-secondary" style={{ fontSize: 11, padding: '3px 8px' }}
                    onClick={() => setActive(e.plate, true)}>Reactivate</button>
                )}
                <button type="button" className="btn btn-secondary" style={{ fontSize: 11, padding: '3px 8px', color: 'var(--accent-red)' }}
                  onClick={() => deleteEntry(e.plate)}>Delete</button>
              </div>
            ))}
          </div>
        )
      )}
    </div>
  );
}
