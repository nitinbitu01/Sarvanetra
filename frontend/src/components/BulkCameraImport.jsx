// frontend/src/components/BulkCameraImport.jsx
//
// Bulk/CSV camera onboarding — the Model 1 deliverable "bulk import ...
// camera onboarding". The backend has carried POST /cameras/bulk and
// POST /cameras/bulk/csv since earlier work this project, but neither had
// a frontend surface: a department handing over its estate as a spreadsheet
// had no UI path to use it, only curl/Postman.
import { useState, useRef } from 'react';
import { useAuth, API } from '../context/AuthContext';

const CSV_TEMPLATE =
  'name,url,department,zone,risk_level,gps_lat,gps_lon,ip_address,protocol,vendor,camera_type,installed_at\n' +
  'Example Junction Cam,,Traffic Police,Central,MEDIUM,23.03,72.58,192.168.1.50,RTSP,hikvision,PTZ,2022-03-01\n';

export default function BulkCameraImport({ onImported }) {
  const { authFetch } = useAuth();
  const [mode, setMode] = useState('csv'); // 'csv' | 'json'
  const [jsonText, setJsonText] = useState(
    '[\n  {"name": "Example Cam", "url": "", "department": "Traffic Police",\n   "zone": "Central", "camera_type": "Fixed"}\n]'
  );
  const [csvText, setCsvText] = useState('');
  const [fileName, setFileName] = useState('');
  const [negotiateUrls, setNegotiateUrls] = useState(false);
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = () => setCsvText(String(reader.result || ''));
    reader.readAsText(file);
  };

  const downloadTemplate = () => {
    const blob = new Blob([CSV_TEMPLATE], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'camera_bulk_import_template.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleImport = async () => {
    setError('');
    setResult(null);
    setImporting(true);
    try {
      let res;
      if (mode === 'csv') {
        if (!csvText.trim()) { setError('Choose a CSV file first.'); setImporting(false); return; }
        res = await authFetch(`${API}/cameras/bulk/csv`, {
          method: 'POST',
          headers: { 'Content-Type': 'text/csv' },
          body: csvText,
        });
      } else {
        let parsed;
        try {
          parsed = JSON.parse(jsonText);
        } catch {
          setError('That is not valid JSON.');
          setImporting(false);
          return;
        }
        if (!Array.isArray(parsed)) {
          setError('JSON must be an array of camera objects.');
          setImporting(false);
          return;
        }
        res = await authFetch(`${API}/cameras/bulk`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cameras: parsed, negotiate_urls: negotiateUrls }),
        });
      }
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || `Import failed (${res.status})`);
        setImporting(false);
        return;
      }
      setResult(data);
      if (data.succeeded > 0) onImported?.();
    } catch {
      setError('Network error reaching the import endpoint.');
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 700, marginTop: 20 }}>
      <div className="card-header">
        <div className="card-title">📦 Bulk / CSV Camera Import</div>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
        <button type="button" className={`btn ${mode === 'csv' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('csv')}>CSV Upload</button>
        <button type="button" className={`btn ${mode === 'json' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('json')}>Paste JSON</button>
      </div>

      {mode === 'csv' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <input ref={fileInputRef} type="file" accept=".csv,text/csv" onChange={handleFile} />
            <button type="button" className="btn btn-secondary" onClick={downloadTemplate}>
              ⬇ Download CSV template
            </button>
          </div>
          {fileName && (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Loaded: {fileName} ({csvText.split('\n').filter(Boolean).length - 1} data row(s))
            </div>
          )}
          <label className="form-label" style={{ marginTop: 4 }}>
            Or paste CSV text directly
          </label>
          <textarea className="form-input" rows={6} style={{ fontFamily: 'monospace', fontSize: 12 }}
            value={csvText} onChange={(e) => { setCsvText(e.target.value); setFileName(''); }}
            placeholder={CSV_TEMPLATE} />
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <textarea className="form-input" rows={8} style={{ fontFamily: 'monospace', fontSize: 12 }}
            value={jsonText} onChange={(e) => setJsonText(e.target.value)} />
          <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={negotiateUrls}
              onChange={(e) => setNegotiateUrls(e.target.checked)} />
            Negotiate stream URLs for rows with vendor + IP but no URL (slower — up to 25s per such row)
          </label>
        </div>
      )}

      {error && <div className="test-result error" style={{ marginTop: 12 }}>❌ {error}</div>}

      {result && (
        <div className={`test-result ${result.failed === 0 ? 'success' : 'error'}`} style={{ marginTop: 12 }}>
          {result.succeeded} of {result.total} camera(s) imported
          {result.failed > 0 ? `, ${result.failed} failed` : ''}.
          {(result.csv_parse_errors || []).length > 0 && (
            <div style={{ marginTop: 6, fontSize: 12 }}>
              Rows skipped before import: {result.csv_parse_errors.map(
                (e) => `row ${e.row} (${e.error})`).join(', ')}
            </div>
          )}
          {(result.results || []).some((r) => !r.success) && (
            <div style={{ marginTop: 6, fontSize: 12 }}>
              {result.results.filter((r) => !r.success).map((r, i) => (
                <div key={i}>Row {r.row} ({r.name}): {r.error}</div>
              ))}
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: 14 }}>
        <button type="button" className="btn btn-primary" onClick={handleImport} disabled={importing}>
          {importing ? '⏳ Importing…' : '📥 Import'}
        </button>
      </div>
    </div>
  );
}
