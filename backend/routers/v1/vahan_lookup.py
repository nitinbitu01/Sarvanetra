"""
backend/routers/v1/vahan_lookup.py — VAHAN 4.0 Vehicle Intelligence REST API.
"""
from fastapi import APIRouter, HTTPException, Depends
from backend.services.vahan_service import lookup_plate, VahanRecord

router = APIRouter(prefix="/v1/vahan", tags=["vahan"])


@router.get("/{license_plate}")
async def get_vahan_details(license_plate: str):
    rec = lookup_plate(license_plate)
    return {
        "license_plate": rec.license_plate,
        "owner_name": rec.owner_name,
        "make_model": rec.make_model,
        "vehicle_class": rec.vehicle_class,
        "fuel_type": rec.fuel_type,
        "color": rec.color,
        "registration_date": rec.registration_date,
        "insurance_expiry": rec.insurance_expiry,
        "insurance_valid": rec.insurance_valid,
        "fitness_expiry": rec.fitness_expiry,
        "pucc_validity": rec.pucc_validity,
        "pucc_valid": rec.pucc_valid,
        "is_stolen": rec.is_stolen,
        "cctns_fir_number": rec.cctns_fir_number,
        "lookup_tier": rec.lookup_tier.value,
        "lookup_latency_ms": round(rec.lookup_latency_ms, 2),
    }
