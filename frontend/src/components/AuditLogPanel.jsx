// frontend/src/components/AuditLogPanel.jsx
import { useEffect, useState, useCallback } from 'react';
import { useAuth, API } from '../context/AuthContext';

const ACTION_COLORS = {
  LOGIN: '#10b981',
  CAMERA_ADD: '#3b82f6',
  CAMERA_TEST: '#06b6d4',
  CAMERA_SOFT_DELETE: '#ef4444',
  ALERT_VIEW: '#8b5cf6',
  EVIDENCE_VERIFY: '#f59e0b',
  FOUND: '#10b981',
  NOT_FOUND: '#94a3b8',
  FORBIDDEN: '#ef4444',
};

export default function AuditLogPanel() {
  const { authFetch } = useAuth();
  const [activeTab, setActiveTab] = useState('searches'); // 'searches' | 'system'
  
  // Search query log state (Part A & C)
  const [searchLogs, setSearchLogs] = useState([]);
  const [searchTotal, setSearchTotal] = useState(0);
  const [loadingSearches, setLoadingSearches] = useState(true);
  
  // Filters for search log
  const [filterPlate, setFilterPlate] = useState('');
  const [filterOutcome, setFilterOutcome] = useState('ALL');
  const [filterOfficer, setFilterOfficer] = useState('ALL');
  const [filterStartDate, setFilterStartDate] = useState('');
  const [filterEndDate, setFilterEndDate] = useState('');

  // Officers seen in the audit trail, for the officer filter dropdown.
  // Fetched once, unfiltered, so the dropdown always lists everyone who has
  // ever searched - not just the officers visible under the current filter.
  const [officerOptions, setOfficerOptions] = useState([]);

  // System audit log state
  const [systemRows, setSystemRows] = useState([]);
  const [loadingSystem, setLoadingSystem] = useState(false);

  // Fetch journey query logs
  const fetchSearchLogs = useCallback(async () => {
    setLoadingSearches(true);
    try {
      let url = `${API}/audit/journey-queries?limit=100`;
      if (filterPlate.trim()) url += `&plate=${encodeURIComponent(filterPlate.trim())}`;
      if (filterOutcome !== 'ALL') url += `&outcome=${encodeURIComponent(filterOutcome)}`;
      if (filterOfficer !== 'ALL') url += `&user_id=${encodeURIComponent(filterOfficer)}`;
      if (filterStartDate) url += `&start_date=${encodeURIComponent(filterStartDate)}`;
      if (filterEndDate) url += `&end_date=${encodeURIComponent(filterEndDate)}T23:59:59`;

      const res = await authFetch(url);
      if (res.ok) {
        const data = await res.json();
        setSearchLogs(data.results || []);
        setSearchTotal(data.total || 0);
      }
    } catch (err) {
      console.error('Failed to fetch search audit logs:', err);
    } finally {
      setLoadingSearches(false);
    }
  }, [authFetch, filterPlate, filterOutcome, filterOfficer, filterStartDate, filterEndDate]);

  // Fetch the full officer roster once, independent of the active filters,
  // so the dropdown doesn't shrink to whatever the last filter happened to
  // match.
  useEffect(() => {
    let alive = true;
    authFetch(`${API}/audit/journey-queries?limit=1000`)
      .then(res => (res.ok ? res.json() : null))
      .then(data => {
        if (!alive || !data) return;
        const seen = new Map();
        for (const row of data.results || []) {
          if (row.user_id != null && !seen.has(row.user_id)) {
            seen.set(row.user_id, row.username || `officer_${row.user_id}`);
          }
        }
        setOfficerOptions(Array.from(seen, ([user_id, username]) => ({ user_id, username })));
      })
      .catch(() => {});
    return () => { alive = false; };
  }, [authFetch]);

  // Fetch general system audit logs
  const fetchSystemLogs = useCallback(async () => {
    setLoadingSystem(true);
    try {
      const res = await authFetch(`${API}/audit-log?limit=100`);
      if (res.ok) setSystemRows(await res.json());
    } catch (err) {
      console.error('Failed to fetch system audit logs:', err);
    } finally {
      setLoadingSystem(false);
    }
  }, [authFetch]);

  useEffect(() => {
    if (activeTab === 'searches') {
      fetchSearchLogs();
      const interval = setInterval(fetchSearchLogs, 3000);
      return () => clearInterval(interval);
    } else {
      fetchSystemLogs();
      const interval = setInterval(fetchSystemLogs, 5000);
      return () => clearInterval(interval);
    }
  }, [activeTab, fetchSearchLogs, fetchSystemLogs]);

  const watchlistHits = searchLogs.filter(r => r.is_watchlist).length;
  const foundCount = searchLogs.filter(r => (r.outcome || '').toUpperCase() === 'FOUND').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header & Mode Tabs */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 800, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>🛡️</span>
            <span>Security & Search Audit Trail</span>
            <span style={{
              fontSize: 10, fontWeight: 800, background: 'rgba(16,185,129,0.2)', color: '#34d399',
              border: '1px solid rgba(16,185,129,0.4)', borderRadius: 12, padding: '2px 8px', letterSpacing: '0.5px'
            }}>
              ● LIVE SYNC (3s)
            </span>
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>
            Permanent non-repudiable surveillance query register & RBAC integrity log (ISO 27001 / DPDP Act compliant)
          </div>
        </div>

        {/* Tab Selector */}
        <div style={{ display: 'flex', background: 'rgba(255,255,255,0.06)', borderRadius: 8, padding: 3, gap: 4 }}>
          <button
            type="button"
            onClick={() => setActiveTab('searches')}
            style={{
              padding: '6px 14px', borderRadius: 6, border: 'none', fontSize: 12, fontWeight: 700, cursor: 'pointer',
              background: activeTab === 'searches' ? '#3b82f6' : 'transparent',
              color: activeTab === 'searches' ? '#fff' : '#94a3b8',
              transition: 'all 0.15s',
            }}
          >
            🔍 Plate Search Queries ({searchTotal})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('system')}
            style={{
              padding: '6px 14px', borderRadius: 6, border: 'none', fontSize: 12, fontWeight: 700, cursor: 'pointer',
              background: activeTab === 'system' ? '#3b82f6' : 'transparent',
              color: activeTab === 'system' ? '#fff' : '#94a3b8',
              transition: 'all 0.15s',
            }}
          >
            ⚙️ System Admin Events ({systemRows.length})
          </button>
        </div>
      </div>

      {activeTab === 'searches' ? (
        <>
          {/* Top Stat Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            <div style={{ background: 'rgba(30,41,59,0.5)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '12px 16px' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Total Queries Recorded</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: '#f8fafc', marginTop: 4 }}>{searchTotal}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Permanent journey_query_log</div>
            </div>
            <div style={{ background: 'rgba(30,41,59,0.5)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '12px 16px' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Successful Lookups</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: '#10b981', marginTop: 4 }}>{foundCount}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Sightings found in CCTV index</div>
            </div>
            <div style={{ background: 'rgba(30,41,59,0.5)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '12px 16px' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Watchlist Queries</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: '#ef4444', marginTop: 4 }}>{watchlistHits}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Wanted vehicle audit triggers</div>
            </div>
            <div style={{ background: 'rgba(30,41,59,0.5)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '12px 16px' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>RBAC Enforcement</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: '#38bdf8', marginTop: 4 }}>Normalized</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Department-scoped isolation</div>
            </div>
          </div>

          {/* Filter Bar */}
          <div style={{
            background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 10, padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1, minWidth: 200 }}>
              <span style={{ fontSize: 14 }}>🔍</span>
              <input
                type="text"
                placeholder="Filter by plate number (e.g. GJ03, MH12)..."
                value={filterPlate}
                onChange={e => setFilterPlate(e.target.value)}
                style={{
                  width: '100%', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 6, padding: '6px 10px', color: '#fff', fontSize: 13,
                }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>Outcome:</span>
              <select
                value={filterOutcome}
                onChange={e => setFilterOutcome(e.target.value)}
                style={{
                  background: '#1e293b', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 6, padding: '6px 10px', color: '#fff', fontSize: 12,
                }}
              >
                <option value="ALL">All Outcomes</option>
                <option value="FOUND">FOUND Only</option>
                <option value="NOT_FOUND">NOT_FOUND Only</option>
                <option value="FORBIDDEN">FORBIDDEN (403)</option>
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>Officer:</span>
              <select
                value={filterOfficer}
                onChange={e => setFilterOfficer(e.target.value)}
                style={{
                  background: '#1e293b', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 6, padding: '6px 10px', color: '#fff', fontSize: 12,
                }}
              >
                <option value="ALL">All Officers</option>
                {officerOptions.map(o => (
                  <option key={o.user_id} value={o.user_id}>{o.username}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>From:</span>
              <input
                type="date"
                value={filterStartDate}
                onChange={e => setFilterStartDate(e.target.value)}
                style={{
                  background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 6, padding: '5px 8px', color: '#fff', fontSize: 12,
                }}
              />
              <span style={{ fontSize: 12, color: '#94a3b8' }}>To:</span>
              <input
                type="date"
                value={filterEndDate}
                onChange={e => setFilterEndDate(e.target.value)}
                style={{
                  background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 6, padding: '5px 8px', color: '#fff', fontSize: 12,
                }}
              />
              {(filterStartDate || filterEndDate) && (
                <button
                  type="button"
                  onClick={() => { setFilterStartDate(''); setFilterEndDate(''); }}
                  style={{
                    background: 'transparent', border: 'none', color: '#94a3b8',
                    cursor: 'pointer', fontSize: 12,
                  }}
                  title="Clear date range"
                >
                  ✕
                </button>
              )}
            </div>

            <button
              type="button"
              onClick={fetchSearchLogs}
              style={{
                background: '#3b82f6', border: 'none', color: '#fff', borderRadius: 6,
                padding: '6px 14px', fontSize: 12, fontWeight: 700, cursor: 'pointer',
              }}
            >
              ↻ Refresh Logs
            </button>
          </div>

          {/* Table Container */}
          {loadingSearches ? (
            <div style={{ textAlign: 'center', padding: 40, color: '#94a3b8' }}>⏳ Loading search audit records…</div>
          ) : searchLogs.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 40, color: '#64748b', background: 'rgba(0,0,0,0.2)', borderRadius: 10 }}>
              No search audit records found for the selected filter.
            </div>
          ) : (
            <div style={{ overflowX: 'auto', borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, textAlign: 'left' }}>
                <thead>
                  <tr style={{ background: 'rgba(15,23,42,0.9)', color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <th style={{ padding: '10px 14px' }}>Query Time</th>
                    <th style={{ padding: '10px 14px' }}>Officer / User</th>
                    <th style={{ padding: '10px 14px' }}>Role</th>
                    <th style={{ padding: '10px 14px' }}>Target Plate</th>
                    <th style={{ padding: '10px 14px' }}>Result</th>
                    <th style={{ padding: '10px 14px' }}>Source IP</th>
                    <th style={{ padding: '10px 14px' }}>Audit Reason / Details</th>
                  </tr>
                </thead>
                <tbody>
                  {searchLogs.map(row => {
                    const isFound = (row.outcome || '').toUpperCase() === 'FOUND';
                    const isForbidden = (row.outcome || '').toUpperCase() === 'FORBIDDEN';
                    const displayPlate = (row.plate && row.plate !== '—')
                      ? row.plate
                      : ((row.reid_id && row.reid_id !== '—') ? row.reid_id : 'GENERAL_SEARCH');

                    return (
                      <tr
                        key={row.id}
                        style={{
                          borderBottom: '1px solid rgba(255,255,255,0.04)',
                          background: row.is_watchlist ? 'rgba(239,68,68,0.08)' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '10px 14px', color: '#94a3b8', whiteSpace: 'nowrap', fontSize: 12 }}>
                          {row.query_time ? new Date(row.query_time).toLocaleString() : '—'}
                        </td>
                        <td style={{ padding: '10px 14px', fontWeight: 600, color: '#e2e8f0' }}>
                          👤 {row.username || `officer_${row.user_id}`}
                        </td>
                        <td style={{ padding: '10px 14px', color: '#60a5fa', textTransform: 'uppercase', fontSize: 11, fontWeight: 700 }}>
                          {row.role}
                        </td>
                        <td style={{ padding: '10px 14px' }}>
                          <span style={{
                            fontFamily: 'monospace', fontWeight: 800, fontSize: 13, letterSpacing: '0.5px',
                            color: row.is_watchlist ? '#fca5a5' : '#38bdf8',
                            background: row.is_watchlist ? 'rgba(239,68,68,0.2)' : 'rgba(56,189,248,0.12)',
                            padding: '4px 10px', borderRadius: 6, border: row.is_watchlist ? '1px solid #ef4444' : '1px solid rgba(56,189,248,0.25)',
                            display: 'inline-block',
                          }}>
                            {displayPlate}
                          </span>
                          {row.is_watchlist && (
                            <span style={{
                              marginLeft: 6, fontSize: 10, fontWeight: 800, color: '#ef4444',
                              background: 'rgba(239,68,68,0.15)', padding: '2px 6px', borderRadius: 4,
                            }}>
                              🚨 WATCHLIST
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '10px 14px' }}>
                          <span style={{
                            fontSize: 11, fontWeight: 800, padding: '3px 8px', borderRadius: 4,
                            background: isForbidden ? 'rgba(239,68,68,0.15)' : isFound ? 'rgba(16,185,129,0.15)' : 'rgba(148,163,184,0.15)',
                            color: isForbidden ? '#f87171' : isFound ? '#34d399' : '#94a3b8',
                          }}>
                            {row.outcome}
                          </span>
                        </td>
                        <td style={{ padding: '10px 14px', color: '#64748b', fontSize: 12, fontFamily: 'monospace' }}>
                          {row.source_ip}
                        </td>
                        <td style={{ padding: '10px 14px', color: '#cbd5e1', fontSize: 12 }}>
                          {row.reason}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        /* System Events Tab */
        <>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={fetchSystemLogs}
              style={{
                background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)',
                color: '#fff', borderRadius: 6, padding: '5px 12px', fontSize: 12, cursor: 'pointer',
              }}
            >
              ↻ Refresh System Log
            </button>
          </div>
          {loadingSystem ? (
            <div style={{ textAlign: 'center', padding: 40, color: '#94a3b8' }}>⏳ Loading system events…</div>
          ) : systemRows.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>No system events recorded.</div>
          ) : (
            <div style={{ overflowX: 'auto', borderRadius: 10, border: '1px solid rgba(255,255,255,0.08)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, textAlign: 'left' }}>
                <thead>
                  <tr style={{ background: 'rgba(15,23,42,0.9)', color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <th style={{ padding: '10px 14px' }}>Time</th>
                    <th style={{ padding: '10px 14px' }}>User</th>
                    <th style={{ padding: '10px 14px' }}>Action</th>
                    <th style={{ padding: '10px 14px' }}>Resource</th>
                    <th style={{ padding: '10px 14px' }}>IP</th>
                    <th style={{ padding: '10px 14px' }}>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {systemRows.map(r => (
                    <tr key={r.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '10px 14px', color: '#94a3b8', whiteSpace: 'nowrap', fontSize: 12 }}>
                        {r.created_at ? new Date(r.created_at).toLocaleString() : '—'}
                      </td>
                      <td style={{ padding: '10px 14px', color: '#e2e8f0' }}>#{r.user_id || '—'}</td>
                      <td style={{ padding: '10px 14px' }}>
                        <span style={{ color: ACTION_COLORS[r.action] || '#38bdf8', fontWeight: 700, fontSize: 12 }}>
                          {r.action}
                        </span>
                      </td>
                      <td style={{ padding: '10px 14px', color: '#94a3b8' }}>
                        {r.resource_type || '—'}{r.resource_id ? ` #${r.resource_id}` : ''}
                      </td>
                      <td style={{ padding: '10px 14px', color: '#64748b', fontSize: 11, fontFamily: 'monospace' }}>
                        {r.ip_address || '—'}
                      </td>
                      <td style={{ padding: '10px 14px', color: '#cbd5e1', fontSize: 11, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {r.details ? JSON.stringify(r.details) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
