# start_top10_live.ps1  — Top-10 Camera Fleet: All features on 10 cameras simultaneously
#
# WHAT THIS DOES
#   Runs the complete Sentinel Gujarat AI pipeline on the top 10 selected live
#   CCTV cameras simultaneously. Every camera receives:
#     - Vehicle detection   (YOLOv8s@640, batched GPU inference)
#     - Vehicle tracking    (per-camera BoT-SORT, no cross-camera leakage)
#     - ANPR/OCR            (async worker — does NOT block 10fps inference)
#     - Trajectory          (Theil-Sen speed, calibrated homography)
#     - Event detection     (congestion, anomaly, stolen-vehicle alerts)
#     - Cross-camera ReID   (OSNet-IBN + FAISS)
#     - DB storage          (full persistence — tracks, plates, alerts, journeys)
#     - Live dashboard      (Top-10 Command Centre at http://localhost:3000)
#
# ARCHITECTURE  (2 workers × 5 cameras = equal priority, 10fps each)
#   Worker A:  CAM_09, CAM_08, CAM_07, CAM_10, CAM_18
#   Worker B:  CAM_21, CAM_27, CAM_06, CAM_04, CAM_22
#   ANPR:      Async worker thread per process (does not block detection)
#
# PROVEN SETTINGS (RTX 4070, 12 GB VRAM)
#   2 workers × 5 cameras × 10fps
#   GPU budget:  ~4.5 GB VRAM  (well under 12 GB)
#   Per-worker:  detect 28ms + track 25ms + ANPR queue <1ms → ~10fps
#
# STREAM SOURCE
#   Primary:  RTSP rtsp://103.250.160.189:8554/stream/camNN
#   Fallback:  Local clips in data/clips/{cam_id}/*.mp4 (auto-activates on drop)
#   Corp8:    https://live.corp8.cloud (credentials in config/corp8_credentials.json)
#
# HOW TO START
#   .\start_top10_live.ps1
#   Then open http://localhost:3000  → MONITOR → Top-10 Command Centre
#
# HOW TO VERIFY
#   After ~90s (CUDA model load): all 10 tiles show AI State = ACTIVE
#   Live proof: http://localhost:8000/api/v1/top10/status

$ErrorActionPreference = 'Stop'
$root = "C:\Users\24bcscs031\Downloads\sentinel_gujarat_day8\sentinel gujarat"
$py   = "C:\Users\24bcscs031\AppData\Local\Programs\Python\Python311\python.exe"

Set-Location $root

# ── Step 1: Kill any orphaned workers from a previous run ──────────────────
Write-Output ""
Write-Output "================================================================"
Write-Output " Sentinel Gujarat — Top-10 Camera Fleet"
Write-Output "================================================================"
Write-Output ""
Write-Output "[1/4] Clearing previous workers..."
try {
    & $py -m backend.scripts.kill_workers --wait 12
} catch {
    Write-Output "      (kill_workers not critical — continuing)"
}
Start-Sleep -Seconds 3

# ── Step 2: Environment ────────────────────────────────────────────────────
Write-Output "[2/4] Setting environment..."

$env:PYTHONUTF8          = "1"
$env:PYTHONIOENCODING    = "utf-8"
$env:PYTHONUNBUFFERED    = "1"

# ── Corp8 credentials ──────────────────────────────────────────────────────
$credsPath = "$root\config\corp8_credentials.json"
if (Test-Path $credsPath) {
    $creds = Get-Content $credsPath | ConvertFrom-Json
    $env:CORP8_EMAIL    = $creds.email
    $env:CORP8_PASSWORD = $creds.password
    $env:CORP8_BASE     = $creds.base
    Write-Output "      Corp8 credentials loaded."
} else {
    Write-Output "      WARNING: config\corp8_credentials.json not found — live streams may fail."
}

# ── RTSP primary stream ────────────────────────────────────────────────────
$env:CORP8_RTSP_HOST = "103.250.160.189"
$env:CORP8_RTSP_PORT = "8554"

# TCP transport — required for cam21 and cam25-30 (UDP NAT issues via corp8)
$env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000|fflags;nobuffer|flags;low_delay|max_delay;500000"

# ── Stream source policy ───────────────────────────────────────────────────
$env:SENTINEL_FORCE_CLIPS  = "0"   # prefer RTSP live; fallback to clips on disconnect
$env:SENTINEL_STRICT_LIVE  = "0"   # allow clip failover (never blank screen)

# ── TOP-10 CAMERA SELECTION ────────────────────────────────────────────────
# 10 cameras selected for: traffic density, ANPR capability, geographic coverage
# Priority: Junagadh corridor (9,8,7,10) + Rajkot (18) + North/South Gujarat (21,27,06,04,22)
$TOP10 = "CAM_09,CAM_08,CAM_07,CAM_10,CAM_18,CAM_21,CAM_27,CAM_06,CAM_04,CAM_22"
$env:SENTINEL_ONLY_CAMERAS = $TOP10

# ── Fleet workers — 2 workers × 5 cameras each ────────────────────────────
# Equal priority: no focus/non-focus asymmetry. All cameras get same resources.
$env:SENTINEL_FLEET_WORKERS      = "2"   # 2 workers × 5 cameras = 10fps each
$env:SENTINEL_EQUAL_PRIORITY     = "1"   # disables focus-worker asymmetry
$env:SENTINEL_FLEET_READER_FPS   = "10"  # all cameras equal at 10fps
$env:SENTINEL_FOCUS_CAMERAS      = ""    # no special focus camera

# ── ANPR — async worker, does not block 10fps detection ───────────────────
# ANPR-capable cameras (plates confirmed readable >50px per plate_capability.json):
$env:SENTINEL_ANPR_CAMERAS = "CAM_09,CAM_08,CAM_07,CAM_10,CAM_18,CAM_21,CAM_27,CAM_06"
$env:SENTINEL_ANPR_ASYNC   = "1"   # KEY FIX: ANPR runs on separate worker thread
                                    # Without this, 8 ANPR cameras = 1.2fps not 10fps

# ── GPU / inference tuning ────────────────────────────────────────────────
$env:SENTINEL_YOLO_IMGSZ          = "640"   # 640 proven for 10 cameras
$env:SENTINEL_INFER_BATCH         = "8"     # drain up to 8 frames per cycle
$env:SENTINEL_WORKER_GPU_FRACTION = "0.45"  # 2 workers × 0.45 = 0.90 of 12GB

# ── Publishing ────────────────────────────────────────────────────────────
$env:SENTINEL_PUBLISH_ASYNC       = "1"
$env:SENTINEL_PUBLISH_MAX_WIDTH   = "960"   # 960px tiles (3.9ms vs 15ms at 1920)
$env:SENTINEL_PUBLISH_QUALITY     = "75"    # indistinguishable on tiles

# ── Rollups ───────────────────────────────────────────────────────────────
# Supervisor already sets SENTINEL_ROLLUPS=0 for non-rollup workers

# ── Cross-camera ReID ─────────────────────────────────────────────────────
$env:SENTINEL_GLOBALID_CAMERAS = "ALL"   # cross-camera identity on all 10

# ── Step 3: Launch supervisor ──────────────────────────────────────────────
Write-Output "[3/4] Launching fleet supervisor (2 workers × 5 cameras)..."
$logDir = Join-Path $root "output"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log    = Join-Path $root "output\fleet_top10.log"
$logErr = Join-Path $root "output\fleet_top10.log.err"

$supervisor = Start-Process -FilePath $py `
    -ArgumentList "-m","backend.scripts.fleet_supervisor" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $log `
    -RedirectStandardError $logErr `
    -PassThru -WindowStyle Hidden

Write-Output "      Supervisor PID: $($supervisor.Id)"
Write-Output "      Log: $log"

# ── Step 4: Print dashboard URL ───────────────────────────────────────────
Write-Output "[4/4] Ready."
Write-Output ""
Write-Output "================================================================"
Write-Output " Sarvanetra — Top-10 Camera Fleet STARTED"
Write-Output "================================================================"
Write-Output ""
Write-Output "  Workers:       2  (equal priority, 5 cameras each)"
Write-Output "  Cameras:       $TOP10"
Write-Output "  ANPR cameras:  CAM_09, 08, 07, 10, 18, 21, 27, 06  (async)"
Write-Output "  Stream source: RTSP rtsp://103.250.160.189:8554/stream/camNN"
Write-Output "  ANPR mode:     ASYNC (does not block 10fps detection)"
Write-Output "  GPU budget:    ~4.5 GB / 12 GB VRAM"
Write-Output ""
Write-Output "  Dashboard:     http://localhost:3000"
Write-Output "    → MONITOR → Top-10 Command Centre"
Write-Output ""
Write-Output "  Live proof:    http://localhost:8000/api/v1/top10/status"
Write-Output "  Fleet ops:     http://localhost:3000 → SYSTEM → Fleet Operations"
Write-Output ""
Write-Output "  CUDA model load takes ~90s. Watch Top-10 Command Centre"
Write-Output "  for AI State badges to turn GREEN on all 10 cameras."
Write-Output ""
Write-Output "  Logs:          $log"
Write-Output "  Logs (stderr): $logErr"
Write-Output "================================================================"
