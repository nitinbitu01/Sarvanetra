"""
backend/scripts/regenerate_65b_certificates.py — Regenerates official Section 65B Evidence Act PDFs
for all sealed evidence rows in the database.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.db.session import SessionLocal
from backend.db.models import Evidence, Alert
from backend.services.evidence_capture import _write_custody_pdf, evidence_dir_for


def main():
    db = SessionLocal()
    try:
        evidence_rows = db.query(Evidence).filter(Evidence.status == "COMPLETE").all()
        print(f"Found {len(evidence_rows)} completed evidence rows.")
        count = 0
        for row in evidence_rows:
            target_dir = evidence_dir_for(row.alert_id)
            target_pdf = target_dir / "custody.pdf"
            if _write_custody_pdf(db, row, target_pdf):
                row.pdf_path = f"data/evidence/{row.alert_id}/custody.pdf"
                count += 1
        db.commit()
        print(f"Successfully generated {count} Section 65B Indian Evidence Act certificates!")
    finally:
        db.close()


if __name__ == "__main__":
    main()
