// frontend/src/components/ControlRoom.jsx
//
// Sarvanetra Command-and-Control (C2) Operational Dashboard
// 3-Zone Architecture with react-resizable-panels, C2 Haired Layout, and Palantir Gotham Aesthetics.
import { useState, useCallback, useEffect, useRef } from 'react';
import { Group, Panel, Separator } from 'react-resizable-panels';
import '../styles/dashboard.css';

import { PanelErrorBoundary } from './PanelErrorBoundary';
import C2PanelHeader from './C2PanelHeader';
import SeverityPanel from './SeverityPanel';
import ROIPanel from './ROIPanel';
import LiveMap from './LiveMap';
import ReviewQueueBadge from './ReviewQueueBadge';
import CameraGrid from './CameraGrid';
import CameraModal from './CameraModal';
import LiveDetectionPanel from './LiveDetectionPanel';
import LiveANPRPanel from './LiveANPRPanel';
import UnroutedBanner from './UnroutedBanner';
import Icon from './Icon';

const LAYOUT_STORAGE_KEY = 'sarvanetra.c2.layout';

export default function ControlRoom({
  liveAlert, statusEvent, mergedEvent, zoneIncidentEvent, routingEvent, onNavigate,
}) {
  const [expandedCamera, setExpandedCamera] = useState(null);
  const [fullscreenPanel, setFullscreenPanel] = useState(null);
  const [activePanelKey, setActivePanelKey] = useState('map');

  const handleExpand = useCallback((cam) => setExpandedCamera(cam), []);
  const handleClose = useCallback(() => setExpandedCamera(null), []);

  const toggleFullscreen = useCallback((panelId) => {
    setFullscreenPanel(prev => prev === panelId ? null : panelId);
  }, []);

  // Keyboard Shortcuts:
  // - 'F': Toggle fullscreen on active panel
  // - 'Cmd/Ctrl + \': Reset default layout
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't intercept when typing in inputs
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target?.tagName)) return;

      if (e.key.toLowerCase() === 'f') {
        e.preventDefault();
        toggleFullscreen(activePanelKey);
      } else if ((e.metaKey || e.ctrlKey) && e.key === '\\') {
        e.preventDefault();
        try {
          localStorage.removeItem(LAYOUT_STORAGE_KEY);
          window.location.reload();
        } catch {}
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activePanelKey, toggleFullscreen]);

  // Reset corrupt legacy layout keys from localStorage on mount
  useEffect(() => {
    try {
      Object.keys(localStorage).forEach((k) => {
        if (k.startsWith('react-resizable-panels:c2-')) {
          localStorage.removeItem(k);
        }
      });
    } catch {}
  }, []);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        minHeight: 0,
        minWidth: 0,
        position: 'relative',
        background: 'var(--bg-base, #0A0E1A)',
        padding: '6px 8px 8px',
        overflow: 'hidden',
        boxSizing: 'border-box',
      }}
    >
      {/* ── ZONE 1: 48px Top Alert Strip ─────────────────────────────────── */}
      <UnroutedBanner refreshSignal={routingEvent} />

      {/* ── Fullscreen Overlay for any maximized panel ──────────────────── */}
      {fullscreenPanel && (
        <div className="c2-fullscreen-overlay">
          <div className="dashboard-panel" style={{ height: '100%', width: '100%' }}>
            <C2PanelHeader
              title={`${fullscreenPanel.toUpperCase()} · FULLSCREEN`}
              onToggleMaximize={() => setFullscreenPanel(null)}
              isMaximized={true}
              badge="EXPANDED (PRESS F TO RESTORE)"
            />
            <div style={{ flex: 1, minHeight: 0, overflow: 'hidden', width: '100%' }}>
              {fullscreenPanel === 'severity' && <SeverityPanel />}
              {fullscreenPanel === 'roi' && <ROIPanel />}
              {fullscreenPanel === 'map' && <LiveMap />}
              {fullscreenPanel === 'queue' && <ReviewQueueBadge onOpen={() => onNavigate?.('reid-queue')} />}
              {fullscreenPanel === 'cameras' && (
                <CameraGrid onExpand={handleExpand}
                  expandedCameraId={expandedCamera ? String(expandedCamera.id).toUpperCase() : null} />
              )}
              {fullscreenPanel === 'anpr' && <LiveANPRPanel limit={30} height="100%" />}
              {fullscreenPanel === 'tracking' && (
                <LiveDetectionPanel
                  height="100%"
                  onToggleExpand={() => setFullscreenPanel(null)}
                  isExpanded={true}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── MAIN WORKSPACE (Zone 2 & Zone 3 via react-resizable-panels) ──── */}
      <div style={{
        flex: 1,
        width: '100%',
        minHeight: 0,
        minWidth: 0,
        position: 'relative',
        visibility: fullscreenPanel ? 'hidden' : 'visible',
        pointerEvents: fullscreenPanel ? 'none' : 'auto',
      }}>
        <Group
          orientation="vertical"
          id="c2-v5-workspace-group"
          style={{ height: '100%', width: '100%' }}
        >
          {/* ── ZONE 2: 3-Column Resizable Grid ───────────────────────────── */}
          <Panel id="c2-v5-zone2-main" defaultSize="68%" minSize="40%" style={{ width: '100%' }}>
            <Group
              orientation="horizontal"
              id="c2-v5-columns-group"
              style={{ height: '100%', width: '100%' }}
            >
              {/* Left Column (default ~22%) */}
              <Panel
                id="c2-v5-col-left"
                defaultSize="22%"
                minSize="18%"
                maxSize="30%"
                onMouseEnter={() => setActivePanelKey('severity')}
              >
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 6 }}>
                  {/* Panel 1: ALERT SUMMARY · TODAY */}
                  <div className="dashboard-panel" style={{ flex: '1 1 auto', minHeight: 125 }}>
                    <C2PanelHeader
                      title="ALERT SUMMARY · TODAY"
                      badge="REAL-TIME"
                      onToggleMaximize={() => toggleFullscreen('severity')}
                    />
                    <PanelErrorBoundary panelName="Alert Summary">
                      <SeverityPanel />
                    </PanelErrorBoundary>
                  </div>

                  {/* Panel 2: SYSTEM IMPACT */}
                  <div className="dashboard-panel" style={{ flex: '1 1 auto', minHeight: 125 }}>
                    <C2PanelHeader
                      title="SYSTEM IMPACT"
                      badge="ESTIMATES"
                      onToggleMaximize={() => toggleFullscreen('roi')}
                    />
                    <PanelErrorBoundary panelName="System Impact">
                      <ROIPanel />
                    </PanelErrorBoundary>
                  </div>

                  {/* Panel 3: REVIEW QUEUE (Compact ~110px) */}
                  <div className="dashboard-panel" style={{ flex: '0 0 100px', minHeight: 90 }}>
                    <C2PanelHeader
                      title="REVIEW QUEUE"
                      badge="HITL"
                      onToggleMaximize={() => toggleFullscreen('queue')}
                    />
                    <PanelErrorBoundary panelName="Review Queue">
                      <ReviewQueueBadge onOpen={() => onNavigate?.('reid-queue')} />
                    </PanelErrorBoundary>
                  </div>

                  {/* Quick Shortcut Card to Full-Screen GIS Camera Map */}
                  <div
                    className="dashboard-panel"
                    style={{
                      padding: '8px 12px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: 8,
                      background: 'var(--bg-surface, #0F172A)',
                      border: '1px solid var(--border-default, #374151)',
                      borderRadius: 'var(--radius-md, 4px)',
                      flexShrink: 0,
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary, #F9FAFB)' }}>
                        Gujarat GIS Map
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted, #6B7280)', marginTop: 1 }}>
                        30 nodes · Sector navigation
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => onNavigate?.('map')}
                      style={{
                        background: 'var(--accent-primary, #2563EB)',
                        color: '#FFFFFF',
                        border: 'none',
                        borderRadius: 'var(--radius-sm, 2px)',
                        padding: '4px 10px',
                        fontSize: 10.5,
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 5,
                      }}
                      title="Open dedicated full-screen GIS map workstation"
                    >
                      <Icon name="map" size={12} />
                      Open Map ↗
                    </button>
                  </div>
                </div>
              </Panel>

              {/* Vertical Resize Handle: Left | Center */}
              <Separator
                className="c2-col-resizer"
                title="Drag to resize columns (Cmd+[ / Cmd+])"
              />

              {/* Center Column: CAMERAS Video Wall (Prominent ~52%) */}
              <Panel
                id="c2-v5-col-center"
                defaultSize="52%"
                minSize="38%"
                onMouseEnter={() => setActivePanelKey('cameras')}
              >
                <div className="dashboard-panel" style={{ height: '100%', minHeight: 0 }}>
                  <C2PanelHeader
                    title="CAMERAS · SURVEILLANCE FLEET"
                    badge="30 NODES"
                    onToggleMaximize={() => toggleFullscreen('cameras')}
                  />
                  <PanelErrorBoundary panelName="Camera Grid">
                    <CameraGrid onExpand={handleExpand}
                      expandedCameraId={expandedCamera ? String(expandedCamera.id).toUpperCase() : null} />
                  </PanelErrorBoundary>
                </div>
              </Panel>

              {/* Vertical Resize Handle: Center | Right */}
              <Separator
                className="c2-col-resizer"
                title="Drag to resize columns (Cmd+[ / Cmd+])"
              />

              {/* Right Column: LIVE ANPR Plate Stream (default ~26%) */}
              <Panel
                id="c2-v5-col-right"
                defaultSize="26%"
                minSize="20%"
                maxSize="38%"
                onMouseEnter={() => setActivePanelKey('anpr')}
              >
                <div className="dashboard-panel" style={{ height: '100%', minHeight: 0 }}>
                  <C2PanelHeader
                    title="LIVE ANPR · REAL-TIME READS"
                    badge="LAST 10 READS"
                    onToggleMaximize={() => toggleFullscreen('anpr')}
                  />
                  <PanelErrorBoundary panelName="Live ANPR">
                    <LiveANPRPanel height="100%" limit={20} />
                  </PanelErrorBoundary>
                </div>
              </Panel>
            </Group>
          </Panel>

          {/* Horizontal Resize Handle: Zone 2 | Zone 3 */}
          <Separator
            className="c2-row-resizer"
            title="Drag to resize bottom tracking panel"
          />

          {/* ── ZONE 3: Resizable Bottom Panel (Live AI Tracking, default ~280px / 32%) ── */}
          <Panel
            id="c2-v5-zone3-bottom"
            defaultSize="32%"
            minSize="20%"
            maxSize="50%"
            style={{ width: '100%' }}
            onMouseEnter={() => setActivePanelKey('tracking')}
          >
            <div className="dashboard-panel" style={{ height: '100%', minHeight: 0, width: '100%' }}>
              <PanelErrorBoundary panelName="Live AI Tracking">
                <LiveDetectionPanel
                  height={220}
                  onToggleExpand={() => toggleFullscreen('tracking')}
                  isExpanded={fullscreenPanel === 'tracking'}
                />
              </PanelErrorBoundary>
            </div>
          </Panel>
        </Group>
      </div>

      {/* Expanded Camera Modal (Single-Camera Deep-Dive) */}
      {expandedCamera && (
        <CameraModal camera={expandedCamera} onClose={handleClose} />
      )}
    </div>
  );
}
