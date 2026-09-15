"""backend/services/calibration_validation.py — the gate every calibration
must pass before anything reaches `camera_calibrations`.

WHY A SHARED GATE
  `camera_calibrations.homography_matrix` is read by speed_estimator, which
  turns it into km/h on the dashboard. A matrix written there without an
  independent check produces confident, wrong numbers — the same class of
  failure already corrected twice in this project (a four-point homography
  whose 0.017 m "error" was zero by construction, and a rollup table whose
  summed vehicle_count was neither a vehicle count nor anything else). One gate,
  used by every tier, is how that stays fixed.

WHAT CAN ACTUALLY BE CHECKED HERE, AND WHAT CANNOT
  The obvious check — reprojection error against surveyed ground control
  points — is unavailable: this fleet has no GCPs, which is the reason the
  automatic tiers exist at all. Claiming a reprojection error without them
  would be inventing the very number the gate is supposed to protect.

  Two real checks remain, and they are the ones implemented:

    physical plausibility   a pole between 3 and 12 m, a CCTV field of view
                            between 15 and 120 degrees, a ground plane in
                            front of the camera rather than behind it. These
                            reject the failure modes seen in practice — a
                            0.41 m "pole", a negative depth — without needing
                            any external data.

    independent speed       vehicles seen at two cameras give a speed from
                            GPS distance and timestamps alone, touching no
                            homography. A calibration whose own speeds
                            disagree with that by more than a stated factor is
                            not measuring the road.

  The second check is weak on this deployment and is reported as such: 36
  usable legs exist and only one camera pair (CAM_11 -> CAM_08, n=15) carries
  enough of them to compare against. A camera with no legs gets
  `speed_cross_check: "unavailable"`, never a silent pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# Physical bounds. A pole-mounted traffic camera sits between a shopfront
# bracket and a gantry; anything outside is an estimation failure, not an
# unusual installation.
MIN_HEIGHT_M, MAX_HEIGHT_M = 3.0, 12.0
MIN_FOV_DEG, MAX_FOV_DEG = 15.0, 120.0
# A height estimate whose interquartile spread approaches the estimate itself
# is not an estimate.
MAX_HEIGHT_IQR_RATIO = 0.45
# How far a camera's own median speed may sit from the independent leg speed
# before the calibration is considered to disagree with it. Straight-line GPS
# distance understates road distance and legs include stops, so the leg figure
# is a LOWER bound — the band is deliberately generous and asymmetric.
SPEED_RATIO_MIN, SPEED_RATIO_MAX = 0.5, 3.0


# ── Confidence grades ────────────────────────────────────────────────────────
# Passing the gate is not one state. A calibration that is merely physically
# plausible and one that agrees with an independent measurement are different
# claims, and collapsing them into a single boolean is how "probably fine"
# becomes "verified" three weeks later.
#
# The grades are ordered, and a camera moves UP on its own as more vehicles are
# seen at two cameras — the cross-check needs no new fieldwork, only traffic.
# So `unconfirmed` is a waiting room, not a verdict.
CONFIDENCE_ORDER = ["rejected", "unconfirmed", "cross_validated", "surveyed"]

# Legs behind a cross-check before it counts as evidence rather than an
# anecdote. Measured on this deployment: 36 usable legs exist across 15 camera
# pairs, and only CAM_11 -> CAM_08 (n=15) currently clears this.
MIN_LEGS_FOR_CONFIRMATION = 8


@dataclass
class ValidationResult:
    camera_id: str
    method: str
    passed: bool
    confidence: str = "rejected"
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    error_m: Optional[float] = None
    speed_cross_check: str = "unavailable"
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"camera_id": self.camera_id, "method": self.method,
                "passed": self.passed, "confidence": self.confidence,
                "reasons": self.reasons, "warnings": self.warnings,
                "error_m": self.error_m,
                "speed_cross_check": self.speed_cross_check,
                "detail": self.detail}

    def emits_absolute_kmh(self) -> bool:
        """Whether speed_estimator may publish a bare km/h for this camera.

        Only a cross-validated or surveyed calibration earns an unqualified
        number. An `unconfirmed` one is still usable — it just has to carry its
        uncertainty with it rather than being read as measured fact.
        """
        return self.confidence in ("cross_validated", "surveyed")


def _fov_deg(frame_w: float, focal_px: float) -> float:
    return 2.0 * math.degrees(math.atan(frame_w / (2.0 * max(focal_px, 1e-6))))


def validate_calibration(
    camera_id: str,
    method: str,
    params: dict,
    leg_speeds_kmh: Optional[list] = None,
    camera_speeds_kmh: Optional[list] = None,
) -> ValidationResult:
    """Decide whether one calibration may be written to the live table.

    `params` carries whatever the producing tier estimated: `height_m`,
    `height_iqr_m`, `focal_px`, `frame_size`, `vp_y`, `inliers`.
    `leg_speeds_kmh` are independent camera-to-camera speeds for this camera;
    `camera_speeds_kmh` are the speeds this calibration itself produces.
    """
    r = ValidationResult(camera_id=camera_id, method=method, passed=False)
    r.detail = dict(params)

    h = params.get("height_m")
    if h is None:
        r.reasons.append("no height estimate — scale is undetermined")
    elif not (MIN_HEIGHT_M <= float(h) <= MAX_HEIGHT_M):
        r.reasons.append(f"implausible mounting height {float(h):.2f} m "
                         f"(expected {MIN_HEIGHT_M}-{MAX_HEIGHT_M} m)")

    iqr = params.get("height_iqr_m")
    if h and iqr is not None and float(h) > 0:
        ratio = float(iqr) / float(h)
        r.detail["height_iqr_ratio"] = round(ratio, 3)
        if ratio > MAX_HEIGHT_IQR_RATIO:
            r.reasons.append(f"height observations disagree "
                             f"(IQR {float(iqr):.2f} m is {ratio:.0%} of the "
                             f"estimate)")

    f = params.get("focal_px")
    fs = params.get("frame_size")
    if f is None:
        r.reasons.append("no focal length — depth along the road is not metric")
    elif fs:
        fov = _fov_deg(float(fs[0]), float(f))
        r.detail["fov_deg"] = round(fov, 1)
        if not (MIN_FOV_DEG <= fov <= MAX_FOV_DEG):
            r.reasons.append(f"implausible field of view {fov:.1f} deg")

    # A ground plane behind the camera is a sign-flip, not a calibration.
    vp_y, frame_h = params.get("vp_y"), (fs[1] if fs else None)
    if vp_y is not None and frame_h is not None and float(vp_y) >= float(frame_h):
        r.reasons.append("horizon below the bottom of the frame — the fitted "
                         "plane is behind the camera")

    inl = params.get("inliers")
    if inl is not None and int(inl) < 80:
        r.warnings.append(f"only {int(inl)} inliers behind the second "
                          f"vanishing point")

    # ── Independent cross-check ────────────────────────────────────────────
    n_legs = len(leg_speeds_kmh or [])
    if leg_speeds_kmh and camera_speeds_kmh:
        leg = sorted(leg_speeds_kmh)[n_legs // 2]
        own = sorted(camera_speeds_kmh)[len(camera_speeds_kmh) // 2]
        ratio = own / max(leg, 1e-6)
        r.detail.update({"leg_median_kmh": round(leg, 1),
                         "calibrated_median_kmh": round(own, 1),
                         "speed_ratio": round(ratio, 2),
                         "legs_n": n_legs})
        if not (SPEED_RATIO_MIN <= ratio <= SPEED_RATIO_MAX):
            r.speed_cross_check = "disagrees"
            r.reasons.append(
                f"calibrated median {own:.0f} km/h against an independent "
                f"{leg:.0f} km/h from {n_legs} camera-to-camera legs "
                f"(ratio {ratio:.2f})")
        elif n_legs >= MIN_LEGS_FOR_CONFIRMATION:
            r.speed_cross_check = "agrees"
        else:
            # Agreement on three legs is agreement with three vehicles.
            r.speed_cross_check = "agrees_weak"
            r.warnings.append(
                f"agrees with the independent speed, but on only {n_legs} "
                f"legs (>= {MIN_LEGS_FOR_CONFIRMATION} needed to confirm)")
    else:
        r.speed_cross_check = "unavailable"
        r.warnings.append("no camera-to-camera legs for this camera, so the "
                          "scale rests on physical plausibility alone")

    r.passed = not r.reasons
    # Grade, rather than a bare pass. A camera sitting at `unconfirmed` needs
    # no rework — it needs more vehicles to be seen at two cameras, which
    # happens by itself, and it upgrades on the next run.
    if not r.passed:
        r.confidence = "rejected"
    elif r.speed_cross_check == "agrees":
        r.confidence = "cross_validated"
    else:
        r.confidence = "unconfirmed"
    r.detail["legs_available"] = n_legs
    r.detail["legs_needed_to_confirm"] = MIN_LEGS_FOR_CONFIRMATION
    return r


def summarise(results: list) -> str:
    """One line per camera, for the dated report file."""
    out = ["%-9s %-20s %-16s %-13s %5s  %s"
           % ("camera", "method", "confidence", "cross-check", "legs",
              "reason / note"),
           "-" * 112]
    for v in sorted(results, key=lambda x: (CONFIDENCE_ORDER.index(x.confidence),
                                            x.camera_id)):
        note = "; ".join(v.reasons) or "; ".join(v.warnings) or "ok"
        out.append("%-9s %-20s %-16s %-13s %5s  %s"
                   % (v.camera_id, v.method, v.confidence,
                      v.speed_cross_check,
                      v.detail.get("legs_available", 0), note[:52]))
    n = {g: sum(1 for v in results if v.confidence == g) for g in CONFIDENCE_ORDER}
    out.append("")
    out.append("cross_validated %d   unconfirmed %d   rejected %d"
               % (n["cross_validated"], n["unconfirmed"], n["rejected"]))
    out.append("Only cross_validated cameras may publish a bare km/h; "
               "unconfirmed ones carry their uncertainty.")
    return "\n".join(out)
