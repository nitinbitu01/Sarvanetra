# Start CAM_M1-M4 — the four recorded phone clips replayed as cameras — at 15 fps.
#
# Runs as its OWN pipeline process, next to the CAM_09 one, so starting or
# stopping it never touches the CAM_09 live stream. It reads only local files
# (data/clips/CAM_M*/), never the corp8 portal.
#
# The clips are 478x850 at ~60 fps. Reader 15 fps = every 4th source frame, so
# the video plays at its real speed. The screen gets all 15 fps (smooth
# publish); inference gets every 4th frame (3.75 fps per camera, 15 in total),
# leaving headroom under the ~20 fps one process sustains for the four — a
# steady cadence keeps the tracks, and with them the cross-camera id, intact.

$ErrorActionPreference = 'Stop'
$root = "C:\Users\24bcscs031\Downloads\sentinel_gujarat_day8\sentinel gujarat"
$py   = "C:\Users\24bcscs031\AppData\Local\Programs\Python\Python311\python.exe"
$cams = "CAM_M1,CAM_M2,CAM_M3,CAM_M4"

# Replace a previous CAM_M pipeline only. Both pipelines have the same command
# line, so the PID recorded at launch identifies this one — and it is stopped
# only if that PID is still a run_pipeline process that started before the
# record was written, so a reused PID can never take down the CAM_09 pipeline.
$marker = Join-Path $root "output\cam_m_pipeline.pid"
if (Test-Path $marker) {
    $old = (Get-Content $marker -ErrorAction SilentlyContinue | Select-Object -First 1)
    $proc = if ($old) { Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$old)" -ErrorAction SilentlyContinue }
    if ($proc -and $proc.CommandLine -match "run_pipeline" -and
        $proc.CreationDate -le (Get-Item $marker).LastWriteTime.AddSeconds(5)) {
        Stop-Process -Id $proc.ProcessId -Force
        Start-Sleep -Seconds 2
    }
}

$env:SENTINEL_WORKER_ID              = "cam-m"        # names it in Fleet Operations
$env:SENTINEL_PIPELINE_CAMERAS       = $cams
$env:SENTINEL_ONLY_CAMERAS           = $cams
$env:SENTINEL_SMOOTH_PUBLISH_CAMERAS = $cams
$env:SENTINEL_GLOBALID_CAMERAS       = $cams   # same vehicle -> same GV_ id
# Hold each frame back until inference has covered it, so boxes sit on the
# vehicles instead of trailing a panning handheld camera (~0.3-1 s delay,
# invisible on a replayed recording).
$env:SENTINEL_ALIGN_OVERLAY_CAMERAS  = $cams
$env:SENTINEL_READER_FPS             = "15"
# Every 4th frame to inference: 4 cameras x 3.75 fps = 15 fps against a
# measured capacity of ~20. At every 3rd frame (20 fps offered) the queue sat
# at 30-60 frames (2026-09-11 rollup logs), i.e. ~3 s behind, so the overlay
# aged past its limit and all four cameras lost their boxes together.
$env:SENTINEL_INFER_EVERY            = "4"
# Share the GPU with the CAM_09 pipeline: small detection batches keep this
# process's GPU memory low. With the default 30 it held 3.2 GB, the card filled,
# and both pipelines froze ~1.5 s at a time (CAM_09 alone ran clean).
$env:SENTINEL_INFER_BATCH            = "4"
# The overlay (boxes + GV_ id) is only drawn when it is fresh enough
# (default 0.6 s, tuned for CAM_09's single-camera inference cadence). With
# four cameras sharing one small batch, inference for any one of them lags
# past 0.6 s routinely, so EVERY box was silently dropped — the picture played
# but showed nothing, confirmed 2026-09-11 (zero boxes on a clearly-visible
# rider). Raised only for this process; CAM_09 keeps its tighter default.
$env:SENTINEL_OVERLAY_MAX_AGE        = "3.0"
# The clips are 478x850 portrait. Detect on a portrait canvas of about that
# shape (almost no padding) at 896 px, instead of the fleet's 1920x1080
# landscape canvas at 1280 px, which was ~70% grey padding: about half the
# pixels per frame, and more of them on the vehicles. Measured before this:
# 0 of 4 cameras kept up (rtf 0.866, queue growing), boxes on 52-98% of frames.
$env:SENTINEL_DETECT_INPUT_SIZE      = "480x864"
$env:SENTINEL_YOLO_IMGSZ             = "896"
$env:SENTINEL_STRICT_LIVE            = "0"
$env:SENTINEL_FORCE_CLIPS            = "1"
$env:SENTINEL_CLIP_MATCH             = ""
Remove-Item Env:SENTINEL_HLS_LOCAL_BUFFER -ErrorAction SilentlyContinue

$log = Join-Path $root "output\cam_m_demo.log"
$p = Start-Process -FilePath $py -ArgumentList "-m","backend.scripts.run_pipeline" `
     -WorkingDirectory $root -RedirectStandardOutput $log `
     -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
Set-Content -Path $marker -Value $p.Id -Encoding ascii
Write-Output ("CAM_M1-M4 pipeline started, PID {0}. Tiles fill in within ~30 s." -f $p.Id)
Write-Output ("Log: {0}.err" -f $log)
