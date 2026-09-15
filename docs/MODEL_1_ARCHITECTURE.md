# Model 1 (Centralised CCTV Registry & GIS Mapping) — Coverage and API

Companion to `docs/MODEL_2_ARCHITECTURE.md` and `docs/MODEL_3_4_ARCHITECTURE.md`, same
discipline: every claim below is a file:line citation to code that runs, or a number
measured against the running system, never an assertion.

This is the mandatory foundation model — every other model builds on this registry.

---

## Summary

| Official requirement | Status | Evidence |
|---|---|---|
| Bulk import, manual entry, API-based onboarding | ✅ All three, including UI | §1 |
| Interactive GIS map — department, camera type, status, coverage layers | ✅ All four layers | §2 |
| Camera health **and maintenance-status** monitoring | ✅ Both, distinct states | §3 |
| Gap-analysis: uncovered zones **and ageing infrastructure** | ✅ Both | §4 |
| Role-based search, filtering, **export**, audit trails | ✅ Including export | §5 |
| Working registry portal with GIS map view | ✅ | §6 |
| Registry API documentation | ✅ This document + `/docs` | §7 |
| Sample gap-analysis report | ✅ `docs/sample_gap_analysis_report.json` | §8 |

Every row marked in **bold** above was a real gap as of the last checklist review
against the official deliverable list — found by evidence-based audit, not assumed.
Closed in this revision.

---

## 1. Onboarding — bulk import, manual entry, API, all three with a UI

- **Manual entry**: `CameraOnboardingForm.jsx` → `POST /cameras`
  (`backend/routers/v1/cameras.py:388-419`).
- **API-based / vendor negotiation**: the same form's "Multi-Vendor Onboarding" section
  → `POST /cameras/probe` and the `vendor`/`username`/`password` fields on `POST /cameras`
  — see `docs/MODEL_3_4_ARCHITECTURE.md` §2 for the adapter layer this drives.
- **Bulk import**: `POST /cameras/bulk` (JSON) and `POST /cameras/bulk/csv` (raw CSV text)
  — `cameras.py:501-588`, shared per-row validation via `_bulk_create_cameras_core`
  (`cameras.py:406-499`) so one bad row cannot sink a batch (Pydantic validates a
  `list[CameraCreateRequest]` for the whole request before any route code runs, which is
  exactly the failure mode this design avoids).

**What changed**: bulk/CSV import had a working, tested backend but no frontend — reachable
only via curl/Postman. `frontend/src/components/BulkCameraImport.jsx` is new: a CSV file
upload (with a downloadable template) and a JSON paste mode, both wired to the endpoints
above, rendered on the Camera Registry page (`App.jsx`, `page === 'cameras'`).

## 2. GIS map — department, camera type, status, and coverage layers

`frontend/src/components/CameraMap.jsx` is the registry's own dedicated GIS view (the
"Camera Map" nav item — distinct from `LiveMap.jsx`, the Control Room's operational map,
which layers in officers and live alerts on top of the same registry data for a different
purpose). All four official layers:

- **Department** filter — dropdown over the fetched camera list.
- **Camera type** filter — new: `camera_type` column on `Camera`
  (`backend/db/models.py`, migration `d16_camera_registry_enrichment`), collected at
  onboarding (manual form, bulk JSON, and CSV all accept it), free-text with a curated
  default set (Fixed/PTZ/Dome/Bullet) rather than a fixed enum — a real department's own
  vocabulary should not be rejected at onboarding time.
- **Status** filter — ONLINE / OFFLINE / UNREACHABLE / MAINTENANCE.
- **Coverage layer** — a toggleable `Circle` per camera at an explicitly disclosed,
  illustrative radius (120m — not a measured field of view; no per-camera lens/FOV data
  exists to compute a real one from, and the UI's own tooltip says so).

**What changed**: `camera_type` had no backing column at all before this revision — the
map could not have filtered by it regardless of UI work. `CameraMap.jsx` previously had
zero filter controls (only a static colour legend) and plotted GPS-less cameras at a
**random** point inside Gujarat's bounding box — a dot with no relationship to the
camera's real location, and a direct contradiction of this registry's own gap-analysis
report (§4), which exists specifically to flag missing GPS as a gap. Cameras without GPS
are now excluded from the map (matching `LiveMap.jsx`'s pre-existing, correct pattern),
with the count disclosed in the toolbar instead of silently misplaced.

## 3. Camera health **and** maintenance-status monitoring — two distinct states

- **Automatic health**: `backend/services/camera_heartbeat.py` — real probes (not the
  earlier `DEMO_MODE` no-op), ONLINE/OFFLINE, surfaced in `CameraMap.jsx`, `LiveMap.jsx`,
  `CameraGrid.jsx`, `CameraModal.jsx`, and the sidebar chip in `App.jsx`.
- **Manual maintenance**: new `PATCH /cameras/{id}/maintenance`
  (`cameras.py`, `MaintenanceRequest`/`set_camera_maintenance`) — admin-only, sets a
  distinct `MAINTENANCE` status plus `maintenance_note`/`maintenance_since`. Critically,
  `camera_heartbeat.py::_apply_result` now checks `camera.maintenance_mode` first and
  returns immediately if set — an automatic health check can no longer silently flip a
  deliberately-parked camera back to ONLINE (briefly reachable mid-repair) or OFFLINE
  (fighting the manual flag). Verified with a direct call into `_apply_result` in
  `tests/test_camera_registry_enrichment.py::test_heartbeat_never_overwrites_a_maintenance_flagged_camera`,
  not just inferred from reading the code.
- Toggled from the registry page (`App.jsx`'s camera list) and reflected as a purple
  badge/marker everywhere status is shown, including `GapAnalysisPanel.jsx`'s dedicated
  "Under maintenance" section.

**What changed**: only binary ONLINE/OFFLINE/UNREACHABLE existed before — no concept of a
camera deliberately pulled for repair, which is a different fact than "broken."

## 4. Gap-analysis — uncovered zones **and** ageing infrastructure

`GET /cameras/gap-analysis` (`cameras.py:718-903`) reports, per department-scoped caller:

| Gap | What it means |
|---|---|
| `missing_gps` | not placeable on the map |
| `missing_department` | invisible to every non-admin account by design |
| `offline` | not currently ONLINE (excludes maintenance — see below) |
| `never_health_checked` | onboarded, never actually probed |
| `stale_health` | last probed over 15 minutes ago |
| `ageing_infrastructure` | installed over 3 years ago, by recorded `installed_at` |
| `ageing_infrastructure.unknown_install_date` | no install date on record — reported separately from confirmed-ageing units so the two are never conflated |
| (top-level) `under_maintenance` | deliberately flagged — shown for visibility, not counted as a gap |

**What changed**: `installed_at` had no backing column, so "ageing infrastructure" was
structurally impossible to report — there was nothing to age against. New nullable
`DateTime` column (migration `d16_camera_registry_enrichment`), collected at onboarding,
defaulting to "unknown" rather than "new" when absent (an unknown install date is a real,
reportable gap, not assumed-recent equipment). `offline` now excludes maintenance-flagged
cameras — a department should not be penalised in this report for correctly taking a
camera down for service (verified in
`test_gap_analysis_excludes_maintenance_cameras_from_offline`).

Also newly true: this report is now surfaced in the frontend at all —
`GapAnalysisPanel.jsx`, rendered on the Camera Registry page, was the entire missing
piece; the endpoint itself was already correct and tested.

## 5. Role-based search, filtering, **export**, and audit trails

- Department-scoped RBAC: `assert_camera_in_scope`, `caller_department`, enforced on
  every camera route, not just the list.
- `GET /audit-log` filters: action, resource_type, resource_id, user_id, date range
  (`backend/routers/v1/audit.py`).
- **Export**: new `GET /cameras/export` (`cameras.py`) — CSV, same department scoping as
  the list endpoint (an export cannot contain more than a caller could already page
  through), columns include every registry field plus the new `camera_type`,
  `installed_at`, `maintenance_mode`. Logged to the audit trail
  (`CAMERA_REGISTRY_EXPORT`) with the row count and scope, so an export is itself an
  auditable action, not a silent bulk read. Wired to a download button in
  `GapAnalysisPanel.jsx`.

**What changed**: no export existed anywhere in the registry or audit log before this —
confirmed by grep, not assumed absent.

## 6. Working registry portal with GIS map view

The Camera Registry page (`App.jsx`, `page === 'cameras'`) now composes, in one place:
`CameraOnboardingForm` (manual + vendor negotiation), `BulkCameraImport` (CSV/JSON),
`GapAnalysisPanel` (report + export), and the registered-cameras list (now showing
`camera_type` and a maintenance toggle per row) — one nav click away from `CameraMap`
(`page === 'map'`), which shares the exact same `cameras` state already fetched at the
app level, no separate round-trip.

## 7. Registry API documentation

- Live, always-current: FastAPI's auto-generated OpenAPI docs at `/docs` (Swagger UI) and
  `/redoc` — no `docs_url`/`redoc_url` override in `backend/main.py`, so both are enabled
  by default. This documents every endpoint below with live request/response schemas and
  a "Try it out" console.
- Written reference (this document): endpoint list —

  | Method | Path | Purpose |
  |---|---|---|
  | GET | `/cameras` | List, department-scoped |
  | POST | `/cameras` | Manual/vendor-negotiated onboarding |
  | POST | `/cameras/bulk` | Bulk JSON onboarding |
  | POST | `/cameras/bulk/csv` | Bulk CSV onboarding |
  | POST | `/cameras/probe` | Test vendor negotiation without creating a camera |
  | POST | `/cameras/test` | Bare TCP-port reachability check |
  | PATCH | `/cameras/{id}/maintenance` | Set/clear manual maintenance flag |
  | DELETE | `/cameras/{id}` | Soft delete |
  | GET | `/cameras/gap-analysis` | Registry completeness report |
  | GET | `/cameras/export` | CSV export, department-scoped |
  | GET | `/cameras/scope` | What this caller can/cannot see, and why |
  | GET | `/cameras/adapters/supported` | Vendor fallback chains (Model 3) |
  | POST | `/cameras/discover/onvif` | ONVIF network discovery (Model 3) |
  | GET | `/cameras/{id}/snapshot` | Live JPEG (Model 2 unified viewer) |

## 8. Sample gap-analysis report

`docs/sample_gap_analysis_report.json` — not fabricated, not hand-written: the actual,
live response of `GET /cameras/gap-analysis` against this deployment's real 32-camera
fleet, captured and pretty-printed. Reproducible at any time by calling the endpoint
directly, or viewed formatted in `GapAnalysisPanel.jsx`.
