// frontend/src/components/OfficerApp.jsx
// Mobile PWA view for on-duty Gujarat Police Patrol Officers
import React, { useState, useEffect } from 'react';
import { useAuth, API } from '../context/AuthContext';
import { useWebSocketEvent } from '../context/WebSocketContext';

export default function OfficerApp() {
  const { authFetch, user } = useAuth();
  const [officer, setOfficer] = useState({
    name: user?.name || "Inspector Patel",
    badge: user?.badge || "AHM001",
    status: "AVAILABLE",
    zone: "Central (Ahmedabad)",
    lat: 23.0395,
    lon: 72.5797,
  });

  const [activeAlerts, setActiveAlerts] = useState([
    {
      id: "ALERT-9821",
      crime_type: "STOLEN_VEHICLE",
      severity: "critical",
      danger_score: 9.1,
      camera_name: "Ahmedabad - Ellis Bridge",
      distance_km: 0.8,
      eta_min: 2,
      description: "🚗 Stolen Vehicle GJ05AB1234 detected moving towards Naroda",
      plate_text: "GJ05AB1234",
      timestamp: "Just now",
    }
  ]);

  useWebSocketEvent('alert', (alert) => {
    setActiveAlerts((prev) => [
      {
        id: alert.incident_id || `ALT-${Date.now().toString().slice(-4)}`,
        crime_type: alert.crime_type || alert.alert_type,
        severity: alert.severity || "critical",
        danger_score: alert.danger_score || 8.5,
        camera_name: alert.camera_name || alert.camera_id,
        distance_km: alert.officer?.distance_km || 1.2,
        eta_min: alert.officer?.eta_min || 4,
        description: alert.description,
        plate_text: alert.plate_text,
        timestamp: "Just now",
      },
      ...prev.slice(0, 9),
    ]);
  });

  return (
    <div style={{
      maxWidth: 480, margin: '0 auto', minHeight: '100%',
      background: '#090d16', color: '#f8fafc', padding: 16,
      display: 'flex', flexDirection: 'column', gap: 16,
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Officer Mobile Status Header */}
      <div style={{
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        border: '1px solid #334155', borderRadius: 12, padding: 16,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
      }}>
        <div>
          <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, letterSpacing: 1 }}>
            GUJARAT POLICE PATROL UNIT
          </div>
          <div style={{ fontSize: 18, fontWeight: 700, marginTop: 2 }}>{officer.name}</div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Badge: {officer.badge} • {officer.zone}</div>
        </div>
        <div style={{
          background: '#064e3b', border: '1px solid #10b981', color: '#6ee7b7',
          padding: '6px 12px', borderRadius: 20, fontSize: 11, fontWeight: 700,
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }}></span>
          ON DUTY
        </div>
      </div>

      {/* Quick GPS Status */}
      <div style={{
        background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8,
        padding: 12, display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8'
      }}>
        <span>📍 GPS: 23.0395° N, 72.5797° E</span>
        <span style={{ color: '#10b981' }}>✓ Radio Link Active</span>
      </div>

      {/* Dispatched Incidents */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
          <span>🚨 DISPATCHED INCIDENTS</span>
          <span style={{ color: '#ef4444' }}>{activeAlerts.length} Active</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {activeAlerts.map((alert) => (
            <div
              key={alert.id}
              style={{
                background: 'linear-gradient(180deg, #1e1b2e 0%, #13111c 100%)',
                border: '1px solid #7f1d1d', borderRadius: 10, padding: 14,
                boxShadow: '0 4px 16px rgba(239, 68, 68, 0.15)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <span style={{
                    background: '#ef4444', color: '#fff', fontSize: 10,
                    fontWeight: 800, padding: '2px 6px', borderRadius: 4, textTransform: 'uppercase'
                  }}>
                    {alert.crime_type}
                  </span>
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#94a3b8' }}>{alert.timestamp}</span>
                </div>
                <div style={{
                  background: '#450a0a', border: '1px solid #dc2626', color: '#fca5a5',
                  fontSize: 11, fontWeight: 800, padding: '2px 6px', borderRadius: 4
                }}>
                  DANGER {alert.danger_score}/10
                </div>
              </div>

              <div style={{ fontSize: 14, fontWeight: 600, color: '#f8fafc', marginTop: 10, lineHeight: 1.4 }}>
                {alert.description}
              </div>

              <div style={{
                marginTop: 10, padding: '8px 10px', background: '#0b0f19',
                borderRadius: 6, fontSize: 12, color: '#cbd5e1', display: 'flex', justifyContent: 'space-between'
              }}>
                <span>📍 {alert.camera_name}</span>
                <span style={{ color: '#38bdf8', fontWeight: 600 }}>{alert.distance_km} km away ({alert.eta_min} min ETA)</span>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                <a
                  href={`https://www.google.com/maps/dir/?api=1&destination=23.0395,72.5797`}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    flex: 1, background: '#2563eb', color: '#fff', textAlign: 'center',
                    padding: '8px 12px', borderRadius: 6, fontSize: 12, fontWeight: 700,
                    textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  }}
                >
                  🗺️ Navigate (Google Maps)
                </a>
                {/* Deliberately NOT wired to POST /alerts/{id}/ack or the Day 17
                    offline queue (utils/ackQueue.js). activeAlerts here is
                    seeded from hardcoded mock data (see top of this file),
                    not a real alert with a real id — enqueueAck() against a
                    fake id would queue forever and never resolve. This whole
                    screen is a PWA shell/mockup; AlertDetailCard.jsx is the
                    real officer-facing ACK surface and is the one wired to
                    the offline queue. See BACKLOG.md. */}
                <button
                  onClick={() => window.alert(`Radio acknowledged for ${alert.id}`)}
                  style={{
                    background: '#16a34a', color: '#fff', border: 'none',
                    padding: '8px 12px', borderRadius: 6, fontSize: 12, fontWeight: 700,
                    cursor: 'pointer',
                  }}
                >
                  ✓ ACK Incident
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
