# Start CAM_09 LIVE from the corp8 portal at 12.5 fps — run this just before a demo.
#
# The chain, measured 2026-09-11:
#   portal (AES-128 HLS)  ->  cam09_live_relay  (downloads ahead, decrypts, serves
#   http://127.0.0.1:8091/CAM_09.ts in real time)  ->  pipeline (ffmpeg -re at
#   12.5 fps, YOLO/ANPR, frames published from the reader thread)  ->  dashboard
#   tile (MJPEG).  Measured: 375 frames in 30 s = 12.50 fps, median gap 79 ms,
#   worst 118 ms, 0 gaps over 250 ms. Every byte shown comes from the portal;
#   nothing is read from data/clips.
#
# THE PORTAL METERS WATCH TIME. On 2026-09-11 it streamed 13 min 21 s, then
# refused the account with HTTP 403 "watch time limit reached — please wait for
# your cooldown". A login does not lift it; the relay waits it out on its own
# (one check every 150 s) and resumes. So start this right before the judges
# look, not an hour early — otherwise the quota is spent with nobody watching.
#
# The portal's CAM_09 feed is a recording dated 14/06/2026. The relay plays it
# aligned to the time of day, so the burnt-in clock matches now; the date does not.
#
# The API must run with SENTINEL_CORP8_HEALTH=0 and SENTINEL_SNAPSHOT_PORTAL_GRAB=0
# so the relay is the only thing spending the portal session.

$ErrorActionPreference = 'Stop'
$root = "C:\Users\24bcscs031\Downloads\sentinel_gujarat_day8\sentinel gujarat"
$py   = "C:\Users\24bcscs031\AppData\Local\Programs\Python\Python311\python.exe"

function Get-PyProcs($pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $pattern }
}

# 1. Relay: start it only if it is not already running. Restarting a relay that
#    is sitting out a cooldown gains nothing and costs a portal request.
if (Get-PyProcs "cam09_live_relay") {
    Write-Output "Relay already running."
} else {
    $rlog = Join-Path $root "output\cam09_relay.log"
    Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden `
        -ArgumentList "-m","backend.scripts.cam09_live_relay","--camera","CAM_09","--workers","6","--ahead","24" `
        -RedirectStandardOutput $rlog -RedirectStandardError "$rlog.err" | Out-Null
    Write-Output "Relay started (log: output\cam09_relay.log)."
}

# 2. Pipeline: replace the previous CAM_09 one and its ffmpeg. The CAM_M1-M4
#    pipeline has the same command line; its launcher records its PID, and
#    it is left running (this used to stop every run_pipeline process).
$mMarker = Join-Path $root "output\cam_m_pipeline.pid"
$mPid = if (Test-Path $mMarker) { [int](Get-Content $mMarker | Select-Object -First 1) } else { -1 }
Get-PyProcs "run_pipeline" | Where-Object { $_.ProcessId -ne $mPid } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 3
Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "127\.0\.0\.1:8091" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 2

$env:SENTINEL_PIPELINE_CAMERAS       = "CAM_09"
$env:SENTINEL_ONLY_CAMERAS           = "CAM_09"
$env:SENTINEL_STRICT_LIVE            = "1"      # never fall back to local clips
$env:SENTINEL_FORCE_CLIPS            = "0"
$env:SENTINEL_CLIP_MATCH             = ""
$env:SENTINEL_HLS_LOCAL_BUFFER       = "http://127.0.0.1:8091/CAM_09.ts"
$env:SENTINEL_HLS_HIGH_CAMERAS       = "cam09"
$env:SENTINEL_HLS_FPS_HIGH           = "12.5"   # half the source's 25 fps
$env:SENTINEL_READER_FPS             = "13"
$env:SENTINEL_SMOOTH_PUBLISH_CAMERAS = "CAM_09" # publish every decoded frame
$env:SENTINEL_YOLO_IMGSZ             = "640"    # detection input only; crops stay full-res
# Detection batch cap. The CUDA allocator keeps the peak of its largest batch;
# at the default 30 this process held 7.6-8.3 GB, and with the CAM_M1-M4
# pipeline beside it the 12 GB card filled and both froze ~1.5 s at a time.
$env:SENTINEL_INFER_BATCH            = "8"

$log = Join-Path $root "output\cam09_live_demo.log"
$p = Start-Process -FilePath $py -ArgumentList "-m","backend.scripts.run_pipeline" `
     -WorkingDirectory $root -RedirectStandardOutput $log `
     -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
Set-Content -Path (Join-Path $root "output\cam09_pipeline.pid") -Value $p.Id -Encoding ascii
Write-Output ("Pipeline started, PID {0}. Give it ~60 s, then open Control Room -> CH-09." -f $p.Id)

# 3. Say plainly if the portal is in a watch-time cooldown right now.
$status = Join-Path $root "output\hls_buffer\CAM_09_live\relay_status.json"
Start-Sleep -Seconds 8
if (Test-Path $status) {
    $s = Get-Content $status -Raw | ConvertFrom-Json
    if ($s.cooldown_since) {
        Write-Output ("PORTAL COOLDOWN since {0} - live resumes by itself when the portal allows it." -f $s.cooldown_since)
    } else {
        Write-Output ("Relay cushion {0} s, now playing burnt-in clock {1}." -f $s.cushion_seconds, $s.now_playing_clock)
    }
}
