# Run CAM_09 from its recorded clips — the full pipeline, no portal involved.
#
# Use this when the corp8 portal's watch-time limit (~15 min on, ~15 min off,
# measured 2026-09-11) would interrupt a demo. The picture is CAM_09's own
# footage (data/clips/CAM_09/CAM_09_0730.mp4 and _0830.mp4, 1920x1080 at
# 25 fps, the two busiest daytime clips) and every stage runs on it: detection,
# tracking, ANPR, speed (CAM_09's calibration), persistence.
#
# It is labelled as what it is. The frame banner reads "CAM_09 · RECORDED
# VIDEO" and the camera view's header reads RECORDED — the burnt-in camera
# clock shows the footage's own date, so claiming LIVE would not survive a
# judge reading the screen.
#
# Only the CAM_09 pipeline and the portal relay are stopped; the CAM_M1-M4
# pipeline is left running. To go back to the portal: start_cam09_live_demo.ps1

$ErrorActionPreference = 'Stop'
$root = "C:\Users\24bcscs031\Downloads\sentinel_gujarat_day8\sentinel gujarat"
$py   = "C:\Users\24bcscs031\AppData\Local\Programs\Python\Python311\python.exe"

function Get-PyProcs($pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $pattern }
}

# The CAM_M pipeline has the same command line; its launcher records its PID.
$mMarker = Join-Path $root "output\cam_m_pipeline.pid"
$mPid = if (Test-Path $mMarker) { [int](Get-Content $mMarker | Select-Object -First 1) } else { -1 }

# 1. Portal relay: not needed for clips, and stopping it saves the watch-time quota.
Get-PyProcs "cam09_live_relay" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
# 2. The CAM_09 pipeline (every run_pipeline that is not the CAM_M one) and
#    the ffmpeg that was reading the relay.
Get-PyProcs "run_pipeline" | Where-Object { $_.ProcessId -ne $mPid } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "127\.0\.0\.1:8091" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 3

$env:SENTINEL_WORKER_ID              = "cam09-clip"   # names it in Fleet Operations
$env:SENTINEL_PIPELINE_CAMERAS       = "CAM_09"
$env:SENTINEL_ONLY_CAMERAS           = "CAM_09"
$env:SENTINEL_FORCE_CLIPS            = "1"          # ignore the portal URL
$env:SENTINEL_STRICT_LIVE            = "0"
$env:SENTINEL_CLIP_MATCH             = "0730,0830"  # busiest daytime footage
$env:SENTINEL_READER_FPS             = "13"         # 25 fps source, every 2nd frame: real speed
$env:SENTINEL_SMOOTH_PUBLISH_CAMERAS = "CAM_09"     # screen gets every frame
$env:SENTINEL_INFER_EVERY            = "2"          # inference 6.5 fps, headroom under load
$env:SENTINEL_ALIGN_OVERLAY_CAMERAS  = "CAM_09"     # boxes drawn on the frame they describe
$env:SENTINEL_INFER_BATCH            = "8"          # keeps GPU memory low beside CAM_M
# Stronger detector for this overhead, sun-glared view. Measured on 60 raw
# CAM_09 frames: yolov8s 60 vehicles in 35 frames (missed a plain white SUV
# even at conf 0.10), yolov8m 69 in 40, at 29 vs 14 ms/frame — affordable at
# 6.5 fps inference. Same COCO classes, so nothing downstream changes.
$env:SENTINEL_YOLO_WEIGHTS           = "yolov8m.pt"
$env:SENTINEL_HLS_LOCAL_BUFFER       = ""
$env:SENTINEL_GLOBALID_CAMERAS       = ""

$log = Join-Path $root "output\cam09_clip_demo.log"
$p = Start-Process -FilePath $py -ArgumentList "-m","backend.scripts.run_pipeline" `
     -WorkingDirectory $root -RedirectStandardOutput $log `
     -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
Set-Content -Path (Join-Path $root "output\cam09_pipeline.pid") -Value $p.Id -Encoding ascii
Write-Output ("CAM_09 clip pipeline started, PID {0}. Picture in ~30 s." -f $p.Id)
Write-Output ("Log: {0}.err" -f $log)
