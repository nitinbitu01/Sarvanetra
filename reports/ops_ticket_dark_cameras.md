# Ops ticket — three cameras have never delivered a frame, and the health signal did not notice

**Raised:** 2026-09-07
**Affects:** CAM_15, CAM_17, CAM_22
**Also affects (and is arguably the bigger item):** fleet-wide camera health reporting

---

## Summary

Three of the thirty deployed cameras have produced **zero video and zero
detections since the platform was set up on 2026-08-21** — not degraded output,
none at all.

All three currently report `status = ONLINE` in the cameras table. That status
is not evidence they are up: every camera in the fleet carries the *same*
`last_heartbeat_at` value, so the field is a bulk write rather than a liveness
check. **The three cameras are the visible problem; the health signal that
should have flagged them on day one is the underlying one.**

This is not fixable from our side. A site visit does not help either — there is
nothing to measure until a stream exists.

---

## Evidence

| Camera | Location | District | Stream URL on record | Tracks ever | Clip directory |
|---|---|---|---|---:|---|
| **CAM_15** | Ahmedabad - Suvidha Park | Ahmedabad / West | `https://cctv.corp8.cloud/cam15/index.m3u8` | 0 | exists, **empty** |
| **CAM_17** | Rajkot - Bus Port Main Gate | Rajkot / Saurashtra | `https://cctv.corp8.cloud/cam17/index.m3u8` | 0 | **never created** |
| **CAM_22** | Banaskantha - BK Mervada Tran Rasta | Banaskantha / North Gujarat | `https://cctv.corp8.cloud/cam22/index.m3u8` | 0 | exists, **empty** |

For contrast, a working camera on the same fleet — CAM_11 — has **9,955**
vehicle tracks and clips on disk. So the pipeline itself is fine; these three
inputs are not arriving.

**The two directory states are different failures and probably need different
answers:**

- **CAM_15 and CAM_22** have a clip directory that was created and stayed
  empty. Ingest reached them and got nothing back — consistent with the stream
  URL resolving but serving no media, an authentication rejection, or a camera
  that is powered down behind a reachable endpoint.
- **CAM_17** has no directory at all. Ingest never got far enough to create
  one — consistent with the host or path not resolving.

All three record `consecutive_failures = 1`. The counter moved, so *something*
registered a failure once, and then nothing escalated.

---

## What we need from IT / the camera operator

Per camera, in rough order of how quickly it settles the question:

1. **Is the endpoint reachable from the ingest host?**
   `curl -I https://cctv.corp8.cloud/cam15/index.m3u8` (and `cam17`, `cam22`).
   A 200 with an m3u8 body, a 401/403, a 404, and a DNS failure each point
   somewhere different. Please send the actual response, not a summary of it.
2. **Is the stream URL on record still the correct one?** These were entered on
   2026-08-21. If a camera was re-addressed, re-channelled, or moved to another
   NVR since, the URL above is simply stale and the fix is a corrected URL.
3. **Is the physical camera powered and connected?** For CAM_15 and CAM_22 the
   endpoint appears to answer while carrying no media, which is what a live
   proxy in front of a dead camera looks like.
4. **CAM_17 specifically:** confirm the hostname and path. Its failure mode
   differs from the other two — nothing was ever created for it locally.

If a camera has been **decommissioned**, that is a perfectly good answer. Tell
us and we will mark it deleted, which removes it from the fleet denominator
rather than leaving it as a permanent gap in coverage reporting.

---

## The second item: camera health is not actually being monitored

Worth fixing regardless of what happens to these three.

- **All 30 deployed cameras share one identical `last_heartbeat_at`**
  (`2026-09-04 17:59:42.746201`). A real per-camera heartbeat would produce 30
  different timestamps. This one was written to every row at once.
- Because of that, `status = ONLINE` and `is_online = 1` mean nothing. Three
  cameras that have never delivered a single frame in seventeen days sit in the
  dashboard indistinguishable from CAM_11 and its 9,955 tracks.
- `consecutive_failures` sits at 1 and never grew, so nothing escalated either.

**Suggested fix, in order of value:**

1. Derive liveness from data actually arriving — most recent clip or track per
   camera — rather than from a written status field. It cannot be wrong in this
   particular way, because it is the thing we care about.
2. If a genuine heartbeat is wanted alongside that, have it write per camera and
   alert on staleness rather than on a boolean.
3. Alert on the "never once" case explicitly. A camera that has produced nothing
   *ever* is a different, louder problem than one that stopped this morning, and
   it is the case that went unnoticed here for seventeen days.

---

## Impact if unresolved

Coverage is **27 of 30 cameras**, and multi-camera trajectory tracking has
three permanent holes: a vehicle passing CAM_15, CAM_17 or CAM_22 leaves no
record of having done so.

The three are not equally costly, which is the argument for fixing CAM_22
first:

- **CAM_22 is the only camera in Banaskantha.** That district is not thinly
  covered, it is entirely uncovered — and it is the one on the Rajasthan
  border, so an inter-state movement through it is invisible.
- **CAM_17 is one of only two in Rajkot.** Losing it halves the district and
  leaves no pair, so no cross-camera journey can be reconstructed within
  Rajkot at all.
- **CAM_15 is one of nine in Ahmedabad**, the best-covered district. Its loss
  is a gap in a dense area rather than the removal of a capability.

Until this is resolved, these three are excluded from calibration and coverage
reporting and stated as such — they are not counted toward, or against, any
accuracy number.
