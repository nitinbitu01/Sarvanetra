import { useState } from 'react';
import Icon from './Icon';

export default function C2PanelHeader({
  title,
  onRefresh,
  onToggleMaximize,
  isMaximized = false,
  onToggleCollapse,
  isCollapsed = false,
  extraActions = null,
  badge = null
}) {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div
      className="c2-panel-header"
      onDoubleClick={onToggleMaximize}
      title="Double click to toggle fullscreen"
    >
      <div className="c2-panel-header-left">
        <span className="c2-panel-title">{title}</span>
        {badge && <span className="c2-panel-badge">{badge}</span>}
      </div>

      <div className="c2-panel-header-actions">
        {extraActions}

        {onRefresh && (
          <button
            type="button"
            className="c2-icon-btn"
            onClick={(e) => { e.stopPropagation(); onRefresh(); }}
            title="Refresh panel data"
            aria-label="Refresh"
          >
            <Icon name="refresh" size={14} />
          </button>
        )}

        {onToggleMaximize && (
          <button
            type="button"
            className="c2-icon-btn"
            onClick={(e) => { e.stopPropagation(); onToggleMaximize(); }}
            title={isMaximized ? "Restore panel" : "Maximize panel"}
            aria-label={isMaximized ? "Restore" : "Maximize"}
          >
            <Icon name={isMaximized ? "minimize" : "maximize"} size={14} />
          </button>
        )}

        {onToggleCollapse && (
          <button
            type="button"
            className="c2-icon-btn"
            onClick={(e) => { e.stopPropagation(); onToggleCollapse(); }}
            title={isCollapsed ? "Expand panel" : "Collapse panel"}
            aria-label={isCollapsed ? "Expand" : "Collapse"}
          >
            <Icon name={isCollapsed ? "chevronDown" : "chevronUp"} size={14} />
          </button>
        )}

        <div style={{ position: 'relative' }}>
          <button
            type="button"
            className="c2-icon-btn"
            onClick={(e) => { e.stopPropagation(); setShowMenu(prev => !prev); }}
            title="More options"
            aria-label="More options"
          >
            <Icon name="moreVertical" size={14} />
          </button>

          {showMenu && (
            <div
              className="c2-menu-dropdown"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="c2-menu-item" onClick={() => { setShowMenu(false); onRefresh?.(); }}>
                <Icon name="refresh" size={13} /> Refresh Data
              </div>
              <div className="c2-menu-item" onClick={() => { setShowMenu(false); onToggleMaximize?.(); }}>
                <Icon name={isMaximized ? "minimize" : "maximize"} size={13} /> {isMaximized ? 'Restore View' : 'Fullscreen'}
              </div>
              <div className="c2-menu-item" onClick={() => { setShowMenu(false); alert(`Panel Telemetry: ${title} OK`); }}>
                <Icon name="settings" size={13} /> Panel Diagnostics
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
