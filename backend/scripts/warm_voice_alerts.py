"""Pre-synthesise the spoken broadcast for recent alerts, in all three languages.

The PA broadcast uses gTTS, which needs the network. Generating the audio ahead
of time means a demonstration does not depend on the connection holding: the
service reuses an existing mp3 before it tries to synthesise, so once these
files exist the button plays real speech offline.

Without this, a dropped connection sends synthesis through to the fallback
chime — a beep in place of a Hindi broadcast, while someone is listening.

Run:  python -m backend.scripts.warm_voice_alerts [--limit 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LANGS = ("hi", "gu", "en")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20,
                    help="most recent alerts to warm")
    ap.add_argument("--force", action="store_true",
                    help="re-synthesise even if audio already exists")
    args = ap.parse_args()

    from backend.db.models import Alert, Camera
    from backend.db.session import SessionLocal
    from backend.services.voice_alert_service import get_voice_service

    svc = get_voice_service()
    db = SessionLocal()
    try:
        rows = (db.query(Alert)
                .filter(Alert.is_deleted == False)               # noqa: E712
                .order_by(Alert.created_at.desc())
                .limit(args.limit).all())
        cams = {c.id: c for c in db.query(Camera).all()}
        for c in list(cams.values()):
            if c.camera_id:
                cams.setdefault(c.camera_id, c)

        print("warming %d alert(s) x %d languages\n" % (len(rows), len(LANGS)))
        made = reused = failed = 0
        for a in rows:
            cam = cams.get(a.camera_id)
            cam_name = cam.name if cam else (a.camera_id or "Station Camera")
            for lang in LANGS:
                prefix = "".join(ch for ch in a.id if ch.isalnum() or ch in "-_")[:32]
                target = svc.output_dir / ("%s_%s.mp3" % (prefix, lang))
                if target.exists() and target.stat().st_size > 1024 and not args.force:
                    reused += 1
                    continue
                if args.force and target.exists():
                    target.unlink()
                try:
                    prompt = svc.build_prompt(
                        alert_type=a.alert_type or "SECURITY_ALERT",
                        camera_name=cam_name,
                        subject=a.subject_label or "Suspect",
                        lang=lang)
                    p = svc.synthesize_speech(prompt, lang, a.id)
                    if p.suffix == ".mp3" and p.exists() and p.stat().st_size > 1024:
                        made += 1
                        print("   %s %-3s %6d bytes  %s"
                              % (a.alert_type[:28].ljust(28), lang,
                                 p.stat().st_size, p.name))
                    else:
                        failed += 1
                        print("   %s %-3s FELL BACK to %s — no network?"
                              % (a.alert_type[:28].ljust(28), lang, p.suffix))
                except Exception as exc:                          # noqa: BLE001
                    failed += 1
                    print("   %s %-3s FAILED: %s" % (a.alert_type[:28], lang, exc))

        print("\n%d synthesised, %d already present, %d fell back" % (made, reused, failed))
        if failed:
            print("\nA fallback means gTTS could not reach the network. Those")
            print("alerts will play the chime, not speech — re-run with a")
            print("connection before demonstrating.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
