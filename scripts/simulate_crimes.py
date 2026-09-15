"""
Sentinel Gujarat — Multi-Scenario Crime Simulator
Injects demo crime events into the REAL pipeline.
Real cameras continue running. Simulation adds test alerts on top.
"""

import asyncio
import json
import os
import sys
import uuid
import click
from datetime import datetime, timezone

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCENARIOS = {
    "A": {
        "name": "🚗 Stolen Vehicle Chase (Ahmedabad → Naroda)",
        "events": [
            {
                "cam": "CAM_02", "type": "STOLEN_VEHICLE", "plate": "GJ05AB1234",
                "score": 8.4, "sev": "critical", "delay": 0,
                "desc": "🚗 STOLEN VEHICLE GJ05AB1234 at Ahmedabad Janpath"
            },
            {
                "cam": "CAM_01", "type": "STOLEN_VEHICLE", "plate": "GJ05AB1234",
                "score": 9.1, "sev": "critical", "delay": 2,
                "desc": "🚗 SAME VEHICLE at Ellis Bridge — moving north"
            },
            {
                "cam": "CAM_04", "type": "STOLEN_VEHICLE", "plate": "GJ05AB1234",
                "score": 9.8, "sev": "critical", "delay": 2,
                "desc": "🚗 VEHICLE at Naroda Industrial — INTERCEPT NOW"
            },
        ]
    },
    "B": {
        "name": "👤 Wanted Suspect Multi-City Journey",
        "events": [
            {
                "cam": "CAM_03", "type": "WANTED_SUSPECT",
                "reid_id": "REID_20260821_98A4",
                "score": 9.6, "sev": "critical", "delay": 0,
                "desc": "⚠️ WANTED SUSPECT at Kalupur Station — Murder warrant"
            },
            {
                "cam": "CAM_17", "type": "WANTED_SUSPECT",
                "reid_id": "REID_20260821_98A4",
                "score": 9.8, "sev": "critical", "delay": 2,
                "desc": "⚠️ SAME SUSPECT at Rajkot Bus Port — Cross-city confirmed"
            },
        ]
    },
    "C": {
        "name": "🌙 Night Intrusion — Sabarmati Ashram",
        "events": [
            {
                "cam": "CAM_05", "type": "NIGHT_INTRUSION",
                "score": 8.7, "sev": "critical", "delay": 0,
                "desc": "🌙 3 persons at Sabarmati Ashram Gate at 23:45 — RESTRICTED"
            },
        ]
    },
    "D": {
        "name": "📦 Abandoned Bag — Rajkot Bus Port",
        "events": [
            {
                "cam": "CAM_17", "type": "ABANDONED_OBJECT",
                "score": 7.9, "sev": "high", "delay": 0,
                "desc": "📦 Unattended bag at Rajkot Bus Port for 7 minutes"
            },
        ]
    },
    "E": {
        "name": "👥 Crowd Surge — Junagadh Festival",
        "events": [
            {
                "cam": "CAM_12", "type": "CROWD_SURGE",
                "score": 7.2, "sev": "high", "delay": 0,
                "desc": "👥 14 persons clustered at Junagadh Majevadi Gate — stampede risk"
            },
        ]
    },
    "F": {
        "name": "⚡ Impossible Speed — Identity Spoofing",
        "events": [
            {
                "cam": "CAM_18", "type": "IMPOSSIBLE_SPEED",
                "reid_id": "REID_20260821_A1B2",
                "score": 6.8, "sev": "high", "delay": 0,
                "desc": "⚡ Person traveled 250km in 4 minutes — identity spoofing suspected"
            },
        ]
    },
}

CAM_META = {
    "CAM_01": (23.0395, 72.5797, "Central", "Ahmedabad", "Traffic Police", "Ahmedabad - Ellis Bridge"),
    "CAM_02": (23.0225, 72.5714, "Central", "Ahmedabad", "Municipal Corp", "Ahmedabad - Janpath"),
    "CAM_03": (23.0269, 72.6019, "Central", "Ahmedabad", "Railway Police", "Kalupur Railway Station"),
    "CAM_04": (23.0876, 72.6461, "East", "Ahmedabad", "State Police", "Naroda Industrial"),
    "CAM_05": (23.0600, 72.5806, "North", "Ahmedabad", "Heritage", "Sabarmati Ashram Gate"),
    "CAM_12": (21.5220, 70.4579, "Saurashtra", "Junagadh", "State Police", "Junagadh Majevadi Gate"),
    "CAM_17": (22.3100, 70.8100, "Saurashtra", "Rajkot", "State Police", "Rajkot Bus Port"),
    "CAM_18": (23.0853, 70.1337, "Kutch", "Kutch", "Port Authority", "Kandla Port"),
}


async def inject(r, event: dict, scenario_name: str):
    cam = event["cam"]
    lat, lon, zone, district, dept, cam_name = CAM_META.get(
        cam, (23.0, 72.5, "Central", "Ahmedabad", "State Police", cam)
    )
    alert = {
        "type": "alert",
        "incident_id": f"SIM_{str(uuid.uuid4())[:8].upper()}",
        "crime_type": event["type"],
        "camera_id": cam,
        "camera_name": cam_name,
        "lat": lat, "lon": lon,
        "zone": zone, "district": district,
        "department": dept,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "danger_score": event["score"],
        "severity": event["sev"],
        "score_breakdown": {
            "base_score": round(event["score"] / 1.4, 2),
            "time_multiplier": 1.4,
            "zone_multiplier": 1.6,
            "final_score": event["score"],
        },
        "description": event["desc"],
        "reid_id": event.get("reid_id"),
        "plate_text": event.get("plate"),
        "officer": {
            "id": "OFF_AHM_01", "name": "Inspector Patel",
            "badge": "AHM001", "distance_km": 1.2, "eta_min": 4
        },
        "evidence": {
            "sha256": "a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
        },
        "is_simulated": True,
        "scenario": scenario_name,
    }

    # Save to SQLite database so REST endpoints show it
    try:
        from backend.db.session import SessionLocal
        from backend.db.models import Alert as DBAlert, JourneyEvent as DBJourney
        db = SessionLocal()
        db_alert = DBAlert(
            camera_id=cam,
            alert_type=event["type"],
            severity=event["sev"],
            danger_score=event["score"],
            description=event["desc"],
            reid_id=event.get("reid_id"),
            plate_text=event.get("plate"),
            lat=lat,
            lon=lon,
            zone=zone,
            district=district,
            department=dept,
            evidence_hash=alert["evidence"]["sha256"],
            is_simulated=True,
            timestamp=datetime.utcnow(),
        )
        db.add(db_alert)

        if event.get("reid_id"):
            db.add(DBJourney(
                reid_id=event.get("reid_id"),
                camera_id=cam,
                lat=lat,
                lon=lon,
                zone=zone,
                district=district,
                is_cross_zone=(zone != "Central"),
                timestamp=datetime.utcnow(),
            ))
        db.commit()
        db.close()
    except Exception:
        pass

    if r:
        try:
            raw = json.dumps(alert, default=str)
            await r.publish("sentinel:alerts", raw)
            await r.lpush("sentinel:alert_feed", raw)
            await r.ltrim("sentinel:alert_feed", 0, 199)
        except Exception:
            pass

    return alert


@click.command()
@click.option("--scenario", "-s", default=None, help="A/B/C/D/E/F")
@click.option("--loop", "-l", is_flag=True)
@click.option("--redis-url", default="redis://localhost:6379/0")
def main(scenario, loop, redis_url):
    """Sentinel Gujarat Crime Simulator"""

    async def run():
        r = None
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(redis_url, decode_responses=True, socket_timeout=1.0)
            await r.ping()
            print("✅ Connected to Sentinel Redis")
        except Exception:
            print("ℹ️ Redis not reachable — running direct in-process database injection")
            r = None

        while True:
            keys = [scenario.upper()] if scenario else list(SCENARIOS.keys())
            for key in keys:
                sc = SCENARIOS.get(key)
                if not sc:
                    continue
                print(f"\n🎬 {sc['name']}")
                for ev in sc["events"]:
                    if ev["delay"] > 0:
                        await asyncio.sleep(ev["delay"])
                    alert = await inject(r, ev, sc["name"])
                    print(
                        f"  🚨 {ev['type']} | "
                        f"Camera: {ev['cam']} | "
                        f"Score: {ev['score']} | "
                        f"{ev['sev'].upper()}"
                    )
                await asyncio.sleep(2)

            if not loop:
                break
            print("\n🔄 Repeating in 10s...")
            await asyncio.sleep(10)

        if r:
            await r.aclose()

    asyncio.run(run())


if __name__ == "__main__":
    main()
