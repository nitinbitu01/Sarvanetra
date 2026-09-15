"""Live ANPR performance, split the way the failure modes split.

One blended "accuracy" number cannot be acted on, because three different
failures produce the same low figure and each needs a different fix:

    no plate found      the detector never located a plate region. OCR
                        quality is irrelevant; this is a detection, camera
                        angle or resolution problem.
    found, not read     a region was found but produced no plate string —
                        the crop failed the quality gate, or the grammar
                        rejected the decode. This is a preprocessing or
                        threshold problem.
    read                a plate string was emitted, with a confidence.

The industry terms are capture rate (fraction of vehicles whose plate was
located) and read rate (fraction of located plates decoded). They multiply:
a system that finds 60% of plates and reads 80% of those delivers 48%
end to end, and knowing which of the two is 60% tells you where to work.

This reads what the pipeline now records per track, so it describes live
footage rather than a curated set. It reports no exact-match accuracy: that
needs ground truth, and there is none for live tracks. What it does show is
where reads are being lost, per camera and per hour — which is what a
deployment can act on without any labelling at all.

Run:  python -m backend.scripts.anpr_live_report [--hours 24]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "output" / "sentinel.db"


def bar(pct: float, width: int = 18) -> str:
    filled = int(round(pct / 100 * width))
    return "#" * filled + "." * (width - filled)


def main() -> int:
    hours = 24
    if "--hours" in sys.argv:
        hours = int(sys.argv[sys.argv.index("--hours") + 1])

    con = sqlite3.connect(str(DB))
    where = ("WHERE first_seen >= datetime('now', ?)", (f"-{hours} hours",))

    total, detected, read_ok = con.execute(
        f"SELECT COUNT(*), "
        f"       SUM(CASE WHEN plate_detected = 1 THEN 1 ELSE 0 END), "
        f"       SUM(CASE WHEN plate_text IS NOT NULL AND plate_text != '' "
        f"                THEN 1 ELSE 0 END) "
        f"FROM vehicle_track {where[0]}", where[1]).fetchone()
    total = int(total or 0)
    detected = int(detected or 0)
    read_ok = int(read_ok or 0)

    print(f"\nLive ANPR, last {hours}h")
    print("=" * 64)
    if total == 0:
        print("No tracks in this window. Run the pipeline:")
        print("    python -m backend.scripts.run_pipeline")
        con.close()
        return 0

    print(f"  vehicles tracked      {total:>8,}")
    print(f"  plate region found    {detected:>8,}   capture rate "
          f"{100*detected/total:>5.1f}%")
    print(f"  plate string emitted  {read_ok:>8,}   decode rate  "
          f"{100*read_ok/max(detected,1):>5.1f}%  (of those found)")
    print(f"  end to end            {read_ok:>8,}                "
          f"{100*read_ok/total:>5.1f}%  (capture x decode)")
    print("\n  Decode rate is not read accuracy. It counts plates that")
    print("  produced a string, not plates read correctly — live tracks have")
    print("  no ground truth to check against. Character accuracy is measured")
    print("  separately on the labelled set. A decode rate near 100% with a")
    print("  low capture rate means the work belongs in detection, not OCR.")

    # Where reads are lost. Naming the stage is the whole point.
    lost_detect = total - detected
    lost_read = detected - read_ok
    print(f"\n  lost at detection     {lost_detect:>8,}  "
          f"{100*lost_detect/total:>5.1f}%  no plate region located")
    print(f"  lost at recognition   {lost_read:>8,}  "
          f"{100*lost_read/total:>5.1f}%  region found, no string decoded")

    print(f"\n{'camera':<10}{'tracks':>8}{'capture':>9}{'read':>8}"
          f"{'end-end':>9}  ")
    print("-" * 64)
    rows = con.execute(
        f"SELECT camera_id, COUNT(*), "
        f"       SUM(CASE WHEN plate_detected = 1 THEN 1 ELSE 0 END), "
        f"       SUM(CASE WHEN plate_text IS NOT NULL AND plate_text != '' "
        f"                THEN 1 ELSE 0 END) "
        f"FROM vehicle_track {where[0]} "
        f"GROUP BY camera_id HAVING COUNT(*) >= 5 "
        f"ORDER BY 4 * 1.0 / COUNT(*) DESC", where[1]).fetchall()
    for cam, n, det, rd in rows:
        n, det, rd = int(n), int(det or 0), int(rd or 0)
        cap_pct = 100 * det / n
        read_pct = 100 * rd / max(det, 1)
        e2e = 100 * rd / n
        print(f"{cam:<10}{n:>8,}{cap_pct:>8.1f}%{read_pct:>7.1f}%"
              f"{e2e:>8.1f}%  {bar(e2e)}")

    # Confidence distribution says whether a threshold is throwing away good
    # reads or letting bad ones through.
    print(f"\n{'confidence':<14}{'reads':>8}   distribution")
    print("-" * 50)
    band_rows = con.execute(
        f"SELECT CAST(plate_confidence * 10 AS INT), COUNT(*) "
        f"FROM vehicle_track {where[0]} AND plate_confidence IS NOT NULL "
        f"GROUP BY 1 ORDER BY 1", where[1]).fetchall()
    peak = max((int(c) for _, c in band_rows), default=1)
    for band, cnt in band_rows:
        lo = int(band) / 10
        print(f"{f'{lo:.1f}-{lo+0.1:.1f}':<14}{int(cnt):>8,}   "
              f"{'#' * max(1, int(28 * int(cnt) / peak))}")

    locked = con.execute(
        f"SELECT SUM(CASE WHEN plate_locked = 1 THEN 1 ELSE 0 END), "
        f"       AVG(plate_votes) "
        f"FROM vehicle_track {where[0]} AND plate_text IS NOT NULL",
        where[1]).fetchone()
    if locked and locked[0] is not None:
        print(f"\n  reads confirmed by temporal vote: {int(locked[0]):,} "
              f"of {read_ok:,}")
        if locked[1]:
            print(f"  mean frames agreeing per read   : {float(locked[1]):.1f}")

    print("""
Capture rate and read rate multiply. Whichever is lower is where the next
piece of work belongs: a low capture rate is a detection, angle or
resolution problem and no OCR change will move it; a low read rate on
well-captured plates is a preprocessing, threshold or model problem.""")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
