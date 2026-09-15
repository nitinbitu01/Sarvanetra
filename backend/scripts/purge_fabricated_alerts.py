"""Remove alerts that were written by a seeding script, and their evidence.

backend/scripts/seed_realistic_crime_alerts.py wrote one alert per impressive
category — a wanted-fugitive intercept, a stolen-vehicle hotlist hit, a
loitering subject, an abandoned bag — with every detail a literal in the
source file:

    "Wanted Fugitive: Vikas Dubey (FIR #412/2026, IPC 302/120B)"
    "94.2% cosine similarity"
    "GJ01AB1234 (White Mahindra Bolero)", "FIR #982/2026"
    "Inspector Vikram Solanki", call sign "GARUDA-4", "eta_minutes": 2.5
    "bbox": [430, 220, 590, 580], "track_id": 12

No detector produced any of it. The bounding box is why the "AI suspect
targeting reticle" in those clips marks a fixed rectangle rather than a
subject, and the FIR numbers name records that do not exist.

Their evidence is removed with them. A clip is only proof of an event that
happened; footage attached to an invented event is footage of nothing, and
leaving it in place would leave 60 alerts sharing 26 files, five congestion
alerts on CAM_02 hours apart all pointing at the same seconds of video.

Alerts the pipeline actually raised are kept. Their evidence is deleted only
where it was produced by the old capture path, which had no record of where in
the footage the event occurred and cut the middle of the camera's first file
instead; those alerts can have real evidence re-cut once the pipeline runs
again with provenance recording enabled.

Run:  python -m backend.scripts.purge_fabricated_alerts [--apply]
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "output" / "sentinel.db"

# Categories that exist only because the seeder wrote them. Each had exactly
# one row, on a different camera, twelve minutes apart.
SEEDED_TYPES = (
    "WATCHLIST_FACE_MATCH",
    "STOLEN_VEHICLE_WATCHLIST_HIT",
    "SUSPICIOUS_LOITERING",
    "ABANDONED_OBJECT",
    "TRIPLE_RIDING_NO_HELMET",
    "NIGHT_PERIMETER_INTRUSION",
    "WRONG_WAY_HAZARD",
    "CROWD_SURGE_ANOMALY",
)


def main() -> int:
    apply = "--apply" in sys.argv
    con = sqlite3.connect(str(DB))
    print(f"mode: {'APPLY' if apply else 'DRY RUN (pass --apply to write)'}\n")

    ph = ",".join("?" * len(SEEDED_TYPES))
    seeded = con.execute(
        f"SELECT id, camera_id, alert_type, evidence_path FROM alerts "
        f"WHERE alert_type IN ({ph})", SEEDED_TYPES).fetchall()

    print(f"seeded alerts to delete: {len(seeded)}")
    for _, cam, atype, _ in seeded:
        print(f"   {cam:<8} {atype}")

    # Evidence produced before provenance existed cannot be verified against
    # its source, so it is cleared rather than left looking authoritative.
    stale = con.execute(
        "SELECT a.id, a.camera_id, a.alert_type, a.evidence_path "
        "FROM alerts a WHERE a.evidence_path IS NOT NULL "
        f"AND a.alert_type NOT IN ({ph})", SEEDED_TYPES).fetchall()
    print(f"\nreal alerts whose evidence predates provenance: {len(stale)}")

    dirs = set()
    for aid, _, _, ep in seeded + stale:
        if ep:
            dirs.add((ROOT / ep).parent)

    print(f"evidence directories to remove: {len(dirs)}")
    total_bytes = sum(
        f.stat().st_size for d in dirs if d.is_dir()
        for f in d.rglob("*") if f.is_file())
    print(f"  {total_bytes / 1e6:.1f} MB")

    if not apply:
        print("\ndry run — nothing written.")
        con.close()
        return 0

    ids = [r[0] for r in seeded]
    if ids:
        con.executemany("DELETE FROM alerts WHERE id = ?", [(i,) for i in ids])
        con.executemany("DELETE FROM evidence WHERE alert_id = ?",
                        [(str(i),) for i in ids])
    stale_ids = [r[0] for r in stale]
    if stale_ids:
        con.executemany(
            "UPDATE alerts SET evidence_path = NULL, evidence_hash = NULL "
            "WHERE id = ?", [(i,) for i in stale_ids])
        con.executemany("DELETE FROM evidence WHERE alert_id = ?",
                        [(str(i),) for i in stale_ids])
    con.commit()

    removed = 0
    for d in dirs:
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1

    print(f"\ndeleted {len(ids)} seeded alerts, cleared evidence on "
          f"{len(stale_ids)} real ones, removed {removed} directories")
    print("\nremaining:")
    for atype, n in con.execute(
            "SELECT alert_type, COUNT(*) FROM alerts GROUP BY 1 "
            "ORDER BY 2 DESC LIMIT 8"):
        print(f"   {atype:<40} {n:>6}")
    n_ev = con.execute(
        "SELECT COUNT(*) FROM alerts WHERE evidence_path IS NOT NULL").fetchone()[0]
    print(f"\nalerts still carrying evidence: {n_ev}")
    print("Real evidence is cut when the pipeline next raises an alert, from "
          "the frame it was looking at.")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
