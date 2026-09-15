# start_fleet_demo.ps1 — Run all 30 cameras from recorded clips
#
# WHAT THIS SHOWS
#   Every camera runs the complete pipeline at once using its own recorded clips:
#   detection, tracking, appearance, speed where calibrated, ANPR on CAM_09.
#   Fleet Operations (SYSTEM → Fleet Operations) reports each camera's rate,
#   each worker's ms/frame, and the GPU — so the claim is checkable on screen.
#
# WHY CLIPS AND NOT THE PORTAL
#   Measured 2026-09-12: the corp8 account is metered on wall-clock watching,
#   account-wide. 30 cameras were all refused within 10 s of each other at 15
#   minutes, and HTTP 429 throttling held the whole account to ~7 camera-seconds
#   per wall second. That is a supply limit at the source; no compute helps.
#   The footage here is each camera's own recording; the frame banner says
#   "RECORDED VIDEO" on every tile.
#
#   For a live picture of CAM_09 specifically, run start_cam09_live_demo.ps1.
#
# MEASURED SETTINGS  (Run Book §15, 2026-09-12)
#   9 workers, embed every frame, imgsz 640, reader_fps 10:
#     ~9.5 fps per camera, ~15% drop rate, GPU ~91%

$ErrorActionPreference = 'Stop'
$root = "C:\Users\24bcscs031\Downloads\sentinel_gujarat_day8\sentinel gujarat"
$py   = "C:\Users\24bcscs031\AppData\Local\Programs\Python\Python311\python.exe"

# Kill any orphaned workers from a previous run.
Write-Output "Clearing previous workers..."
& $py -m backend.scripts.kill_workers --wait 12
Start-Sleep -Seconds 4

# ── Environment ───────────────────────────────────────────────────────────
$env:SENTINEL_FORCE_CLIPS           = "1"     # every camera plays its own recording
$env:SENTINEL_STRICT_LIVE           = "0"
$env:SENTINEL_FLEET_WORKERS         = "9"     # 9 × avg 3 cameras = 30 cameras
$env:SENTINEL_FLEET_READER_FPS      = "10"    # measured best: 10 fps per camera
$env:SENTINEL_FOCUS_CAMERAS         = "CAM_09"
$env:SENTINEL_ANPR_CAMERAS          = "CAM_09" # only camera with readable plates
$env:SENTINEL_YOLO_IMGSZ            = "640"
$env:SENTINEL_PUBLISH_ASYNC         = "1"     # JPEG encode off the inference thread
$env:SENTINEL_PUBLISH_MAX_WIDTH     = "960"   # 3.9 ms vs 15 ms at 1920
$env:SENTINEL_PUBLISH_QUALITY       = "75"    # indistinguishable on fleet tiles
$env:SENTINEL_WORKER_GPU_FRACTION   = "0.1"   # 9 × 0.1 leaves the card room

# ── Launch supervisor ─────────────────────────────────────────────────────
$log = Join-Path $root "output\fleet_demo.log"
$p = Start-Process -FilePath $py -ArgumentList "-m","backend.scripts.fleet_supervisor" `
     -WorkingDirectory $root -RedirectStandardOutput $log `
     -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden

Write-Output ""
Write-Output ("Fleet supervisor started, PID {0}." -f $p.Id)
Write-Output "All 9 workers load in ~90 s. Watch SYSTEM → Fleet Operations."
Write-Output "Proof endpoint: http://localhost:8000/api/v1/fleet/proof"
Write-Output ("Log: {0}" -f $log)
Write-Output ("Log (stderr): {0}.err" -f $log)
