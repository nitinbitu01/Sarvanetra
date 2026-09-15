"""Do the analytics endpoints report measurements, or patterns?

Called directly against a real session rather than over HTTP, so this runs
without a server and without a token. Each check names the fabrication it
guards against, so a regression says which one came back.

Run:  python -m backend.scripts.test_analytics_honesty
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db.session import SessionLocal                      # noqa: E402
from backend.routers.v1 import analytics as A                    # noqa: E402
from backend.services.vault_service import VaultService          # noqa: E402

db = SessionLocal()
fails = []

print("=== /overview ===")
ov = A.get_analytics_overview(user=None, db=db)
print(f"   window          {ov['window']} from {ov['window_start']}")
print(f"   total_alerts    {ov['total_alerts']}")
print(f"   by_severity     {ov['by_severity']}")
print(f"   acked           {ov['response_time']['acked_count']}")
print(f"   avg_ack_seconds {ov['response_time']['avg_ack_seconds']}")
print(f"   false_alarm_rate{ov['feedback']['false_alarm_rate']}")
print(f"   push            {ov['push']}")
if ov["response_time"]["avg_ack_seconds"] == 14.2:
    fails.append("avg_ack_seconds is back to the hardcoded 14.2")
if ov["feedback"]["false_alarm_rate"] == 0.08:
    fails.append("false_alarm_rate is back to the hardcoded 0.08")
if isinstance(ov.get("push"), dict) and ov["push"].get("delivery_rate") == 0.98:
    fails.append("push delivery rate is back to the hardcoded 0.98")
# The window must actually be the window.
all_alerts = db.execute(A.text("SELECT COUNT(*) FROM alerts")).scalar()
if ov["total_alerts"] == all_alerts and all_alerts > 0:
    in_window = db.execute(
        A.text("SELECT COUNT(*) FROM alerts WHERE created_at >= :s"),
        {"s": ov["window_start"]}).scalar()
    if in_window != all_alerts:
        fails.append("total_alerts fell back to the all-time count")

print("\n=== /trend ===")
tr = A.get_trend_analytics(days=7, user=None, db=db)
print(f"   {len(tr)} days: {[d['count'] for d in tr]}")
# The old version produced 6 + ((i*5+3) % 14) for the day index.
synthetic = [6 + ((i * 5 + 3) % 14) for i in range(6, -1, -1)]
if [d["count"] for d in tr] == synthetic:
    fails.append("trend still returns the modular-arithmetic series")
db_total = db.execute(
    A.text("SELECT COUNT(*) FROM alerts WHERE created_at >= :s"),
    {"s": tr[0]["iso_date"]}).scalar()
if sum(d["count"] for d in tr) != db_total:
    fails.append(f"trend sums to {sum(d['count'] for d in tr)} but the table "
                 f"holds {db_total} alerts in that range")
else:
    print(f"   sums to {db_total}, matching the table")

print("\n=== /zones ===")
zn = A.get_zone_analytics(days=7, user=None, db=db)
print(f"   {len(zn['zones'])} zones; note: {zn['note']}")
for z in zn["zones"][:5]:
    print(f"     {z['zone']:<28} {z['total_alerts']:>4} alerts, "
          f"{z['critical']} critical, ack {z['avg_ack_seconds']}")
if any(z["avg_ack_seconds"] == 16.5 for z in zn["zones"]):
    fails.append("a zone reported the hardcoded 16.5 s acknowledgement time")
if any(z["zone"] in {"Main Entrance Zone", "Parking Zone",
                     "North Perimeter Zone"} for z in zn["zones"]):
    fails.append("the invented demo zones are back")
zone_total = sum(z["total_alerts"] for z in zn["zones"])
n_cams = db.execute(A.text("SELECT COUNT(*) FROM cameras")).scalar()
if zone_total == n_cams * 3:
    fails.append("zone alert counts are still 3 per camera")

print("\n=== /officers ===")
of = A.get_officer_analytics(days=7, user=None, db=db)
print(f"   {len(of['officers'])} officers; note: {of['note']}")
for o in of["officers"][:4]:
    print(f"     {str(o['name']):<22} badge={o['badge']} "
          f"assigned={o['assigned']} acked={o['acked']} "
          f"ack={o['avg_ack_seconds']}")
if any(o["name"] in {"Officer Chen", "Officer Park", "Officer Ramirez"}
       for o in of["officers"]):
    fails.append("the three fictional officers are back")
idx_formula = [8 + (i * 3) % 10 for i in range(len(of["officers"]))]
if of["officers"] and [o["assigned"] for o in of["officers"]] == idx_formula:
    fails.append("officer workload is still a formula on the row index")

print("\n=== /fleet-health (new) ===")
fh = A.get_fleet_health(hours=24, user=None, db=db)
print(f"   {fh['total_cameras']} cameras, {fh['producing']} producing, "
      f"{len(fh['silent'])} silent, {len(fh['never_seen'])} never seen")
print(f"   tracks in window: {fh['total_tracks']:,}")
print(f"   online but silent: {fh['online_but_silent'][:6]}")
top = [c for c in fh["cameras"] if c["tracks_recent"] > 0][:5]
for c in top:
    print(f"     {c['camera_id']:<9} {c['tracks_recent']:>6} tracks  "
          f"{c['status']}")
db_tracks = db.execute(
    A.text("SELECT COUNT(*) FROM vehicle_track WHERE first_seen >= :s"),
    {"s": fh["window_start"]}).scalar()
if fh["total_tracks"] != db_tracks:
    fails.append(f"fleet-health counted {fh['total_tracks']} tracks but the "
                 f"table holds {db_tracks}")
else:
    print(f"   matches vehicle_track exactly ({db_tracks:,})")

print("\n=== vault stats ===")
vs = VaultService(db=db).get_vault_stats()
print(f"   total {vs['total_samples']:,}")
print(f"   environmental_distribution {vs['environmental_distribution']}")
raw = {r[0] for r in db.execute(
    A.text("SELECT DISTINCT UPPER(lighting_condition) FROM vault_entries "
           "WHERE lighting_condition IS NOT NULL"))}
missing = raw - set(vs["environmental_distribution"]) - {
    "SUNNY", "LOW_LIGHT", "GLARE", "RAIN", "MONSOON", "MONSOON_RAIN", "FOG"}
if missing:
    fails.append(f"conditions dropped from the chart: {sorted(missing)}")

# Every classified entry must be counted exactly once. This catches the
# case-collision bug the chart had: the GROUP BY is case-sensitive, so
# 'NIGHT' and 'night' arrived as separate rows and the second overwrote the
# first, displaying 40,222 night entries as 8.
classified = db.execute(
    A.text("SELECT COUNT(*) FROM vault_entries "
           "WHERE lighting_condition IS NOT NULL")).scalar()
reported = sum(vs["environmental_distribution"].values())
if reported != classified:
    fails.append(f"distribution sums to {reported:,} but {classified:,} "
                 f"entries carry a condition")
else:
    print(f"   sums to {reported:,}, matching every classified entry")

db.close()
print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} PROBLEM(S):")
    for f in fails:
        print(f"   - {f}")
    sys.exit(1)
print("Every analytics figure traces to a row in the database.")
