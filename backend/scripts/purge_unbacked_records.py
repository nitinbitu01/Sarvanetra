"""Remove database rows that no measurement stands behind.

Three groups, each fabricated in a different way:

  vault entries with no crop on disk
      The harvest wrote a row and formatted crop_minio_path from the row's
      uuid, but never saved the JPEG and never computed the embedding. The
      crops cannot be reviewed and cannot be retrained on, which is what a
      vault is for. The harvest now writes pixels before the row, so new rows
      are real; these are not.

  camera_calibrations fitted to invented control points
      backend/scripts/calibrate_pilot_cameras.py types four image points and
      four world points per camera by hand. The image points form a
      symmetrical trapezoid — (320,240), (1600,240), (1880,1040), (40,1040) —
      and the world points a symmetrical rectangle, which is why the resulting
      matrices contain exact zeros that a surveyed homography never would.
      Four points determine an 8-DOF homography exactly, so the reported
      0.017 m reprojection error is zero by construction and measures nothing;
      held_out_error_m is a copy of it, so the "held-out" validation is the
      training error under another name.

  speeds on cameras with no calibration
      _load_calibrations synthesised a homography for every CAM_01..CAM_30
      that lacked one, from a single fixed trapezoid reused for a bridge deck,
      a market junction and an indoor bus station. Those cameras reported
      km/h that nothing supports.

Nothing here is deleted to make numbers look better; every group is removed
because the number it produced was not measured. The counts that remain are
smaller and true.

Run:  python -m backend.scripts.purge_unbacked_records [--apply]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "output" / "sentinel.db"
VAULT_ROOT = ROOT / "output"


def main() -> int:
    apply = "--apply" in sys.argv
    if not DB.is_file():
        print(f"no database at {DB}")
        return 1

    con = sqlite3.connect(str(DB))
    con.execute("PRAGMA foreign_keys = ON")

    print(f"database {DB}")
    print(f"mode     {'APPLY' if apply else 'DRY RUN (pass --apply to write)'}")

    # ── vault entries whose crop was never written ──────────────────────────
    rows = con.execute(
        "select id, crop_minio_path from vault_entries "
        "where crop_minio_path is not null"
    ).fetchall()
    missing = [r[0] for r in rows if not (VAULT_ROOT / r[1]).is_file()]
    total = con.execute("select count(*) from vault_entries").fetchone()[0]
    print(f"\nvault_entries                     {total:>9,}")
    print(f"  crop file present on disk       {len(rows)-len(missing):>9,}")
    print(f"  crop missing -> delete          {len(missing):>9,}")

    with_emb = con.execute(
        "select count(*) from vault_entries where embedding_vector is not null"
    ).fetchone()[0]
    print(f"  carrying a real embedding       {with_emb:>9,}")

    # ── calibrations built from invented control points ─────────────────────
    calib = con.execute(
        "select id, camera_id, calibrated_by, reprojection_error_m, "
        "held_out_error_m from camera_calibrations"
    ).fetchall()
    invented = [c[0] for c in calib
                if c[2] == "surveyor_lead_gujarat"
                or (c[3] is not None and c[3] == c[4])]
    print(f"\ncamera_calibrations               {len(calib):>9,}")
    for cid, cam, by, rms, held in calib:
        flag = "  <- invented points" if cid in invented else ""
        same = " (held-out == training)" if rms == held else ""
        print(f"  {cam:<9} by={by:<24} rms={rms}{same}{flag}")

    # ── speeds on cameras that have no homography ───────────────────────────
    calibrated = {c[1] for c in calib if c[0] not in invented}
    if calibrated:
        placeholders = ",".join("?" * len(calibrated))
        bad = con.execute(
            f"select count(*) from vehicle_track where speed_kmh is not null "
            f"and camera_id not in ({placeholders})", sorted(calibrated)
        ).fetchone()[0]
    else:
        bad = con.execute(
            "select count(*) from vehicle_track where speed_kmh is not null"
        ).fetchone()[0]
    tracks = con.execute("select count(*) from vehicle_track").fetchone()[0]
    print(f"\nvehicle_track                     {tracks:>9,}")
    print(f"  speeds on uncalibrated cameras  {bad:>9,}  -> set to NULL")
    print("     (the tracks are kept: the detections and geometry are real,")
    print("      only the metric speed was unsupported)")

    if not apply:
        print("\ndry run — nothing written. Re-run with --apply to commit.")
        con.close()
        return 0

    cur = con.cursor()
    if missing:
        cur.executemany("delete from vault_entries where id = ?",
                        [(i,) for i in missing])
    if invented:
        cur.executemany("delete from camera_calibrations where id = ?",
                        [(i,) for i in invented])
    if calibrated:
        placeholders = ",".join("?" * len(calibrated))
        cur.execute(
            f"update vehicle_track set speed_kmh = NULL, speed_ci_kmh = NULL "
            f"where camera_id not in ({placeholders})", sorted(calibrated))
    else:
        cur.execute("update vehicle_track set speed_kmh = NULL, "
                    "speed_ci_kmh = NULL")
    con.commit()

    print(f"\ndeleted {len(missing):,} vault rows, {len(invented)} "
          f"calibrations; cleared {bad:,} unsupported speeds")
    print("\nremaining, all measurement-backed:")
    for t in ("vault_entries", "camera_calibrations", "vehicle_track"):
        n = con.execute(f"select count(*) from {t}").fetchone()[0]
        print(f"  {t:<22} {n:>9,}")
    n_speed = con.execute(
        "select count(*) from vehicle_track where speed_kmh is not null"
    ).fetchone()[0]
    print(f"  {'tracks with a speed':<22} {n_speed:>9,}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
