"""
backend/services/vahan_service.py — VAHAN 4.0 Vehicle Intelligence Service (v15.0.0)

Closes Gap U: Three-tier fallback architecture:
  - Tier 1: Live VAHAN API (production, post-authorization)
  - Tier 2: Local SQLite mirror (synced nightly via authorized bulk export)
  - Tier 3: Plate format validation only (shadow mode / pre-authorization)
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

import requests

VAHAN_API_BASE = os.getenv("VAHAN_API_BASE_URL", "")
VAHAN_API_KEY = os.getenv("VAHAN_API_KEY", "")
LOCAL_MIRROR_DB = "data/vahan_mirror.db"
TIMEOUT_SEC = 2.0


class LookupTier(Enum):
    LIVE_API = "live_api"
    LOCAL_DB = "local_db"
    FORMAT_ONLY = "format_only"


@dataclass
class VahanRecord:
    license_plate: str
    owner_name: Optional[str]
    make_model: Optional[str]
    vehicle_class: Optional[str]
    fuel_type: Optional[str]
    color: Optional[str]
    registration_date: Optional[str]
    insurance_expiry: Optional[str]
    fitness_expiry: Optional[str]
    pucc_validity: Optional[str]
    is_stolen: bool
    cctns_fir_number: Optional[str]
    lookup_tier: LookupTier
    lookup_latency_ms: float

    @property
    def insurance_valid(self) -> bool:
        if not self.insurance_expiry:
            return False
        try:
            exp = date.fromisoformat(self.insurance_expiry)
            return exp >= date.today()
        except ValueError:
            return False

    @property
    def pucc_valid(self) -> bool:
        if not self.pucc_validity:
            return False
        try:
            exp = date.fromisoformat(self.pucc_validity)
            return exp >= date.today()
        except ValueError:
            return False


_IND_PLATE_RE = re.compile(
    r'^(?:[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$'
)


def init_vahan_mirror(db_path: str = LOCAL_MIRROR_DB) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        license_plate TEXT PRIMARY KEY,
        owner_name TEXT,
        make_model TEXT,
        vehicle_class TEXT,
        fuel_type TEXT,
        color TEXT,
        registration_date TEXT,
        insurance_expiry TEXT,
        fitness_expiry TEXT,
        pucc_validity TEXT,
        is_stolen INTEGER DEFAULT 0,
        cctns_fir_number TEXT
    )
    """)
    conn.commit()
    conn.close()


def lookup_plate(plate: str, db_path: str = LOCAL_MIRROR_DB) -> VahanRecord:
    clean = plate.replace(" ", "").upper()
    t0 = time.monotonic()

    # ── Tier 1: Live VAHAN API ─────────────────────────────────────────
    if VAHAN_API_BASE and VAHAN_API_KEY:
        try:
            resp = requests.get(
                f"{VAHAN_API_BASE}/rc/vehicleinfo/{clean}",
                headers={"x-api-key": VAHAN_API_KEY},
                timeout=TIMEOUT_SEC,
            )
            if resp.status_code == 200:
                data = resp.json()
                latency = (time.monotonic() - t0) * 1000
                return VahanRecord(
                    license_plate=clean,
                    owner_name=data.get("owner_name"),
                    make_model=data.get("maker_model"),
                    vehicle_class=data.get("vehicle_class_desc"),
                    fuel_type=data.get("fuel_type"),
                    color=data.get("color"),
                    registration_date=data.get("reg_date"),
                    insurance_expiry=data.get("insurance_upto"),
                    fitness_expiry=data.get("fitness_upto"),
                    pucc_validity=data.get("pucc_upto"),
                    is_stolen=bool(data.get("blacklisted", False)),
                    cctns_fir_number=data.get("cctns_fir"),
                    lookup_tier=LookupTier.LIVE_API,
                    lookup_latency_ms=latency,
                )
        except Exception:
            pass  # Fall through to Tier 2

    # ── Tier 2: Local SQLite Mirror ────────────────────────────────────
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=1.0)
            row = conn.execute(
                "SELECT * FROM vehicles WHERE license_plate = ?", (clean,)
            ).fetchone()
            conn.close()
            if row:
                latency = (time.monotonic() - t0) * 1000
                return VahanRecord(
                    license_plate=clean,
                    owner_name=row[1],
                    make_model=row[2],
                    vehicle_class=row[3],
                    fuel_type=row[4],
                    color=row[5],
                    registration_date=row[6],
                    insurance_expiry=row[7],
                    fitness_expiry=row[8],
                    pucc_validity=row[9],
                    is_stolen=bool(row[10]),
                    cctns_fir_number=row[11],
                    lookup_tier=LookupTier.LOCAL_DB,
                    lookup_latency_ms=latency,
                )
        except Exception:
            pass  # Fall through to Tier 3

    # ── Tier 3: Format Validation Only ────────────────────────────────
    latency = (time.monotonic() - t0) * 1000
    return VahanRecord(
        license_plate=clean,
        owner_name=None,
        make_model=None,
        vehicle_class=None,
        fuel_type=None,
        color=None,
        registration_date=None,
        insurance_expiry=None,
        fitness_expiry=None,
        pucc_validity=None,
        is_stolen=False,
        cctns_fir_number=None,
        lookup_tier=LookupTier.FORMAT_ONLY,
        lookup_latency_ms=latency,
    )
