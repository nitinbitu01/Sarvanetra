from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session
from backend.auth.dependencies import get_current_user
from backend.db.models import User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit
from backend.services.cad_dispatch import get_dispatcher, UnitStatus

router = APIRouter(prefix="/v1/cad", tags=["cad_patrol"])


class GPSUpdateRequest(BaseModel):
    unit_id: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class DispatchRequest(BaseModel):
    incident_id: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    severity: str = "HIGH"
    description: str = ""


class StatusUpdateRequest(BaseModel):
    status: str
    incident_id: Optional[str] = None


@router.get("/units")
async def get_all_units(current_user: User = Depends(get_current_user)):
    disp = get_dispatcher()
    return {"units": disp.get_fleet_telemetry()}


@router.post("/update_gps")
async def update_unit_gps(
    req: GPSUpdateRequest,
    current_user: User = Depends(get_current_user),
):
    disp = get_dispatcher()
    ok = disp.update_unit_gps(req.unit_id, req.lat, req.lon)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unit {req.unit_id} not found in fleet")
    return {"status": "ok", "unit_id": req.unit_id, "lat": req.lat, "lon": req.lon}


@router.post("/dispatch")
async def dispatch_unit(
    req: DispatchRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    disp = get_dispatcher()
    res = disp.dispatch_nearest(
        incident_lat=req.lat,
        incident_lon=req.lon,
        severity=req.severity,
        description=req.description,
        incident_id=req.incident_id,
    )
    if not res:
        raise HTTPException(status_code=503, detail="No available patrol units with fresh GPS telemetry")

    log_audit(
        db,
        current_user,
        "CAD_PATROL_DISPATCH",
        "pcr_unit",
        res.unit_id,
        {"incident_id": req.incident_id, "distance_km": res.distance_km, "eta_minutes": res.eta_minutes},
        request.client.host if request.client else None,
    )

    return {
        "status": "dispatched",
        "unit_id": res.unit_id,
        "call_sign": res.call_sign,
        "officer": res.officer_name,
        "contact": res.contact,
        "distance_km": res.distance_km,
        "eta_minutes": res.eta_minutes,
    }


@router.post("/units/{unit_id}/status")
async def update_unit_status(
    unit_id: str,
    req: StatusUpdateRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    disp = get_dispatcher()
    if unit_id not in disp.fleet:
        raise HTTPException(status_code=404, detail=f"Unit {unit_id} not found in fleet")
    ok = disp.update_status_by_name(unit_id, req.status, req.incident_id)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid status value '{req.status}'")

    log_audit(
        db,
        current_user,
        "CAD_STATUS_UPDATE",
        "pcr_unit",
        unit_id,
        {"status": req.status.upper(), "incident_id": req.incident_id},
        request.client.host if request.client else None,
    )

    return {"status": "ok", "unit_id": unit_id, "new_status": req.status.upper()}

