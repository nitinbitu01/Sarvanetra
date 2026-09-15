// frontend/src/components/CameraOnboardingForm.jsx
import { useState, useEffect } from 'react';
import { useAuth, API } from '../context/AuthContext';

const PROTOCOLS  = ['RTSP', 'ONVIF', 'HTTP', 'HLS'];
const DEPARTMENTS = ['Traffic Police', 'Municipality', 'Railway', 'Other'];
const RISK_LEVELS = ['LOW', 'MEDIUM', 'HIGH'];
const CAMERA_TYPES = ['Fixed', 'PTZ', 'Dome', 'Bullet'];

// A department onboarding an existing camera estate hands over vendor + IP +
// credentials, not a ready stream URL (see docs/MODEL_3_4_ARCHITECTURE.md
// §2) — the backend already negotiates a working URL from that via
// CameraAdapterFactory's fallback chain (backend/routers/v1/cameras.py
// ::_negotiate_and_create_camera, POST /cameras/probe). This form previously
// had no field to even supply a vendor, so that whole negotiation path — the
// part of the system that answers "interoperable" — had no way to be
// exercised from the UI. Vendor list is fetched from GET
// /cameras/adapters/supported rather than hardcoded, so this form can never
// list a vendor the backend does not actually have a fallback chain for.
export default function CameraOnboardingForm({ onCameraAdded }) {
  const { authFetch } = useAuth();
  const [form, setForm] = useState({
    name: '', ip_address: '', protocol: 'RTSP',
    zone: '', department: 'Traffic Police', risk_level: 'LOW',
    gps_lat: '', gps_lon: '',
    vendor: 'unknown', username: 'admin', password: '', channel: 101,
    camera_type: 'Fixed', installed_at: '',
  });
  const [testResult, setTestResult]   = useState(null);
  const [testing, setTesting]         = useState(false);
  const [saving, setSaving]           = useState(false);
  const [savedMsg, setSavedMsg]       = useState('');
  const [errors, setErrors]           = useState({});

  const [vendors, setVendors] = useState(['unknown']);
  const [chains, setChains] = useState({});
  const [showMultiVendor, setShowMultiVendor] = useState(false);
  const [probing, setProbing] = useState(false);
  const [probeResult, setProbeResult] = useState(null);
  const [discovering, setDiscovering] = useState(false);
  const [discovered, setDiscovered] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch(`${API}/cameras/adapters/supported`);
        if (cancelled || !res.ok) return;
        const data = await res.json();
        if (Array.isArray(data.supported_vendors)) setVendors(data.supported_vendors);
        if (data.fallback_chains) setChains(data.fallback_chains);
      } catch {
        // Vendor dropdown just keeps its 'unknown' default; the field is
        // optional server-side too (defaults to "unknown" in CameraCreateRequest).
      }
    })();
    return () => { cancelled = true; };
  }, [authFetch]);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleTest = async () => {
    if (!form.ip_address) { setErrors({ ip_address: 'Required' }); return; }
    setErrors({});
    setTesting(true);
    setTestResult(null);
    try {
      const res = await authFetch(`${API}/cameras/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip_address: form.ip_address, protocol: form.protocol }),
      });
      const data = await res.json();
      setTestResult(data);
    } catch {
      setTestResult({ success: false, message: 'Network error' });
    } finally {
      setTesting(false);
    }
  };

  // Unlike handleTest (a bare TCP-port check), this walks the vendor's
  // adapter fallback chain and reports which protocol actually delivered a
  // frame — POST /cameras/probe, the same negotiation /cameras (create) runs,
  // without creating a camera row.
  const handleProbe = async () => {
    if (!form.ip_address) { setErrors({ ip_address: 'Required to negotiate' }); return; }
    setErrors({});
    setProbing(true);
    setProbeResult(null);
    try {
      const res = await authFetch(`${API}/cameras/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vendor: form.vendor, ip_address: form.ip_address,
          username: form.username, password: form.password,
          channel: form.channel ? parseInt(form.channel, 10) : 101,
        }),
      });
      const data = await res.json();
      setProbeResult(data);
    } catch {
      setProbeResult({ success: false, message: 'Network error reaching the negotiation endpoint' });
    } finally {
      setProbing(false);
    }
  };

  const handleDiscoverOnvif = async () => {
    setDiscovering(true);
    setDiscovered(null);
    try {
      const res = await authFetch(`${API}/cameras/discover/onvif`, { method: 'POST' });
      const data = await res.json();
      setDiscovered({ devices: data.devices || [], liveProbeCount: data.live_probe_count || 0 });
    } catch {
      setDiscovered({ devices: [], liveProbeCount: 0 });
    } finally {
      setDiscovering(false);
    }
  };

  // A device with source "configured_fallback" is a row echoed from the
  // existing camera registry (no physical ONVIF device answered the network
  // probe — see backend/services/camera_adapters/onvif_adapter.py's own
  // comment) rather than something the multicast probe actually found.
  // Applying it still needs a vendor picked deliberately, not defaulted to
  // "onvif" as if the protocol had been confirmed live.
  const applyDiscovered = (dev) => {
    set('ip_address', dev.ip_address || dev.ip || form.ip_address);
    set('vendor', dev.source === 'live_probe' ? 'onvif' : 'unknown');
    setShowMultiVendor(true);
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!form.name) { setErrors({ name: 'Required' }); return; }
    setErrors({});
    setSaving(true);
    setSavedMsg('');
    try {
      const payload = {
        ...form,
        channel: form.channel ? parseInt(form.channel, 10) : 101,
        gps_lat: form.gps_lat ? parseFloat(form.gps_lat) : null,
        gps_lon: form.gps_lon ? parseFloat(form.gps_lon) : null,
      };
      const res = await authFetch(`${API}/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Save failed');
      const cam = await res.json();
      setSavedMsg(`✅ Camera "${cam.name}" added (ID: ${cam.id})`);
      onCameraAdded?.(cam);
      setForm({
        name: '', ip_address: '', protocol: 'RTSP', zone: '', department: 'Traffic Police',
        risk_level: 'LOW', gps_lat: '', gps_lon: '',
        vendor: 'unknown', username: 'admin', password: '', channel: 101,
        camera_type: 'Fixed', installed_at: '',
      });
      setTestResult(null);
      setProbeResult(null);
    } catch (err) {
      setSavedMsg(`❌ ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 700 }}>
      <div className="card-header">
        <div className="card-title">📡 Add New Camera</div>
      </div>

      <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Camera Name *</label>
            <input id="cam-name" className="form-input" placeholder="e.g. Sardar Bridge Cam-1"
              value={form.name} onChange={e => set('name', e.target.value)} required />
            {errors.name && <span style={{ color: 'var(--accent-red)', fontSize: 11 }}>{errors.name}</span>}
          </div>
          <div className="form-group">
            <label className="form-label">IP Address</label>
            <input id="cam-ip" className="form-input" placeholder="192.168.1.100"
              value={form.ip_address} onChange={e => set('ip_address', e.target.value)} />
            {errors.ip_address && <span style={{ color: 'var(--accent-red)', fontSize: 11 }}>{errors.ip_address}</span>}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">🔴 Live RTSP / HTTP Stream URL (Optional for Direct Ingestion)</label>
          <input id="cam-stream-url" className="form-input" placeholder="rtsp://admin:pass@192.168.1.100:554/live or http://..."
            value={form.stream_url || ''} onChange={e => set('stream_url', e.target.value)} style={{ fontFamily: 'monospace' }} />
        </div>

        {/* Model 3: most departments hand over vendor + IP + credentials,
          * not a stream URL — leave the field above blank and the vendor
          * fields below drive the same negotiation the backend already runs
          * on save (CameraAdapterFactory's fallback chain). */}
        <div style={{ border: '1px solid var(--border, #334155)', borderRadius: 8, padding: showMultiVendor ? 14 : '8px 14px' }}>
          <button type="button" onClick={() => setShowMultiVendor(v => !v)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center', gap: 6, width: '100%', fontSize: 13, fontWeight: 600 }}>
            {showMultiVendor ? '▾' : '▸'} 🔀 Multi-Vendor Onboarding — no stream URL yet? Negotiate one
          </button>

          {showMultiVendor && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
              <div className="form-grid-3">
                <div className="form-group">
                  <label className="form-label">Vendor</label>
                  <select id="cam-vendor" className="form-select" value={form.vendor} onChange={e => set('vendor', e.target.value)}>
                    {vendors.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  {chains[form.vendor] && (
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      Tries: {chains[form.vendor].join(' → ')}
                    </span>
                  )}
                </div>
                <div className="form-group">
                  <label className="form-label">Username</label>
                  <input id="cam-vendor-user" className="form-input"
                    value={form.username} onChange={e => set('username', e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">Password</label>
                  <input id="cam-vendor-pass" className="form-input" type="password"
                    value={form.password} onChange={e => set('password', e.target.value)} />
                </div>
              </div>

              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <button id="btn-probe-camera" type="button" className="btn btn-secondary"
                  onClick={handleProbe} disabled={probing}>
                  {probing ? '⏳ Negotiating…' : '🛰️ Negotiate Stream (Probe)'}
                </button>
                <button id="btn-discover-onvif" type="button" className="btn btn-secondary"
                  onClick={handleDiscoverOnvif} disabled={discovering}>
                  {discovering ? '⏳ Scanning…' : '📶 Discover ONVIF Cameras'}
                </button>
              </div>

              {probeResult && (
                <div className={`test-result ${probeResult.success ? 'success' : 'error'}`}>
                  {probeResult.success ? '✅' : '❌'} {probeResult.message}
                  {probeResult.resolved_url && (
                    <div style={{ fontFamily: 'monospace', fontSize: 11, marginTop: 4, wordBreak: 'break-all' }}>
                      {probeResult.resolved_url}
                    </div>
                  )}
                </div>
              )}

              {discovered && (
                discovered.devices.length === 0 ? (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No ONVIF devices found on this network.</div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {discovered.liveProbeCount === 0 && (
                      <div style={{ fontSize: 11, color: 'var(--accent-amber, #f59e0b)', marginBottom: 2 }}>
                        ⚠ No device answered the live network probe. Showing the existing camera
                        registry instead — these are NOT confirmed ONVIF-reachable.
                      </div>
                    )}
                    {discovered.devices.map((dev, i) => (
                      <button key={dev.ip_address || dev.ip || i} type="button"
                        onClick={() => applyDiscovered(dev)}
                        className="btn btn-secondary" style={{ textAlign: 'left', fontSize: 12 }}>
                        {dev.source === 'live_probe' ? '📷' : '📋'} {dev.ip_address || dev.ip} {dev.name ? `— ${dev.name}` : ''}
                        {dev.source !== 'live_probe' && (
                          <span style={{ color: 'var(--text-muted)', fontSize: 10 }}> (registry, not live-probed)</span>
                        )}
                      </button>
                    ))}
                  </div>
                )
              )}
            </div>
          )}
        </div>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Protocol</label>
            <select id="cam-protocol" className="form-select" value={form.protocol} onChange={e => set('protocol', e.target.value)}>
              {PROTOCOLS.map(p => <option key={p}>{p}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Department</label>
            <select id="cam-dept" className="form-select" value={form.department} onChange={e => set('department', e.target.value)}>
              {DEPARTMENTS.map(d => <option key={d}>{d}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Risk Level</label>
            <select id="cam-risk" className="form-select" value={form.risk_level} onChange={e => set('risk_level', e.target.value)}>
              {RISK_LEVELS.map(r => <option key={r}>{r}</option>)}
            </select>
          </div>
        </div>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Camera Type</label>
            <select id="cam-type" className="form-select" value={form.camera_type} onChange={e => set('camera_type', e.target.value)}>
              {CAMERA_TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Installed On (optional)</label>
            <input id="cam-installed" className="form-input" type="date"
              value={form.installed_at} onChange={e => set('installed_at', e.target.value)} />
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              Powers the ageing-infrastructure gap report — left blank, this camera is
              reported as "install date unknown" rather than assumed new.
            </span>
          </div>
        </div>

        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Zone</label>
            <input id="cam-zone" className="form-input" placeholder="e.g. Ahmedabad East"
              value={form.zone} onChange={e => set('zone', e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">GPS Latitude</label>
            <input id="cam-lat" className="form-input" placeholder="23.0225" type="number" step="any"
              value={form.gps_lat} onChange={e => set('gps_lat', e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">GPS Longitude</label>
            <input id="cam-lon" className="form-input" placeholder="72.5714" type="number" step="any"
              value={form.gps_lon} onChange={e => set('gps_lon', e.target.value)} />
          </div>
        </div>

        {/* Test result */}
        {testResult && (
          <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
            {testResult.success ? '✅' : '❌'} {testResult.message}
            {testResult.latency_ms != null && ` (${testResult.latency_ms}ms)`}
          </div>
        )}

        {savedMsg && (
          <div className={`test-result ${savedMsg.startsWith('✅') ? 'success' : 'error'}`}>
            {savedMsg}
          </div>
        )}

        <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
          <button id="btn-test-camera" type="button" className="btn btn-secondary"
            onClick={handleTest} disabled={testing}>
            {testing ? '⏳ Testing…' : '🔌 Test Connection'}
          </button>
          <button id="btn-save-camera" type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? '⏳ Saving…' : '💾 Save Camera'}
          </button>
        </div>
      </form>
    </div>
  );
}
