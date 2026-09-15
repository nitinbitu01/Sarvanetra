// frontend/src/components/CameraMap.jsx
//
// The Gujarat Camera Network GIS map view.
// Uses the high-performance, watermark-free LiveMap engine featuring:
// - Multi-basemap support (Street, Tactical Dark, Satellite)
// - Sector jumps (All Gujarat, Ahmedabad, Gandhinagar, Rajkot, Surat, Junagadh)
// - Layer controls (Departments, Zones, Statuses, Risk Levels, Types)
// - Real-time WebSocket officer locations and active alert pins
import LiveMap from './LiveMap';

export default function CameraMap(props) {
  return <LiveMap {...props} />;
}
