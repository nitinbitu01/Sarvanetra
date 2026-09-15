"""
scripts/sync_tracks_to_journeys.py
Sync all vehicle tracks with detected plates into journey_events.
Ensures 'Trace Designated Vehicle' and 'Plate Search' index every real-time sighting.
"""
import uuid
import sqlite3
from datetime import datetime

def sync():
    conn = sqlite3.connect("output/sentinel.db")
    cur = conn.cursor()

    # Get camera metadata
    cur.execute("SELECT camera_id, name, lat, lon, zone, district FROM cameras")
    cams = {r[0]: {"name": r[1], "lat": r[2], "lon": r[3], "zone": r[4], "district": r[5]} for r in cur.fetchall()}

    # Check existing (reid_id, camera_id, timestamp) in journey_events to avoid duplicates
    cur.execute("SELECT reid_id, camera_id, timestamp FROM journey_events")
    existing = set()
    for r in cur.fetchall():
        ts_str = str(r[2])[:19] if r[2] else ""
        existing.add((r[0], r[1], ts_str))

    # Fetch vehicle tracks with plates
    cur.execute("""
        SELECT camera_id, vehicle_class, plate_text, plate_confidence, last_seen, first_seen
        FROM vehicle_track
        WHERE plate_text IS NOT NULL AND TRIM(plate_text) != ''
    """)
    tracks = cur.fetchall()
    print(f"[*] Found {len(tracks)} vehicle tracks with plate_text")

    inserted = 0
    for cam_id, vclass, plate, conf, last_seen, first_seen in tracks:
        clean_plate = plate.upper().replace(" ", "").strip()
        if not clean_plate:
            continue
        
        ts = last_seen or first_seen or datetime.utcnow().isoformat()
        if isinstance(ts, (int, float)):
            ts_dt = datetime.utcfromtimestamp(ts)
            ts_str = ts_dt.isoformat()
        else:
            ts_str = str(ts)
        
        ts_key = ts_str[:19]
        if (clean_plate, cam_id, ts_key) in existing:
            continue

        c_meta = cams.get(cam_id, {})
        lat = c_meta.get("lat")
        lon = c_meta.get("lon")
        zone = c_meta.get("zone", "Saurashtra")
        district = c_meta.get("district", "Junagadh")

        event_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO journey_events (
                id, reid_id, camera_id, lat, lon, zone, district,
                is_cross_zone, timestamp, object_class, color, subtype,
                plate_text, visual_score, final_score, match_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, clean_plate, cam_id, lat, lon, zone, district,
            0, ts_str, "vehicle", None, vclass or "car",
            clean_plate, 1.0, float(conf or 0.95), "plate_confirmed"
        ))
        existing.add((clean_plate, cam_id, ts_key))
        inserted += 1

    conn.commit()
    conn.close()
    print(f"[✓] Successfully inserted {inserted} new journey events into journey_events table!")

if __name__ == "__main__":
    sync()
