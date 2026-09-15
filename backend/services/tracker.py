"""Per-camera BoT-SORT tracking.

What this file used to be: a class named BoTSORTTracker that ran greedy IoU
matching over a class named KalmanTrack that had no Kalman filter in it -
`predict()` incremented two counters and left the box where it was. Neither
name described what ran. Greedy IoU has no motion model, so it loses a vehicle
the moment another box overlaps it more, and it has no way to tell a PTZ
camera panning from the whole scene moving.

What it is now: Ultralytics' BoT-SORT, which brings the three things the name
promised.

  Kalman motion         a real constant-velocity filter per track, so an
                        occluded vehicle is predicted forward instead of
                        dropped
  GMC (sparseOptFlow)   global motion compensation. These are PTZ cameras;
                        when one pans, every box moves at once, and without
                        GMC that reads as every vehicle simultaneously
                        changing direction
  ReID association      appearance matching on OSNet-IBN embeddings, which
                        recovers an identity after an occlusion that motion
                        alone cannot bridge

The embeddings are supplied by the caller rather than computed here. The
pipeline already needs them for the Active Learning Vault, and passing them in
means one OSNet forward pass per frame serves both.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Ultralytics' shipped botsort.yaml defaults, with two departures noted below.
_TRACKER_ARGS = dict(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    # 30 frames at the pipeline's 5 fps read rate is 6 seconds of occlusion
    # tolerance, which is long enough for a car to pass behind a bus.
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
    gmc_method="ecc",
    proximity_thresh=0.5,
    appearance_thresh=0.8,
    with_reid=True,
    # "auto" makes the encoder a pass-through for features the caller supplies,
    # which is exactly how this pipeline feeds it OSNet embeddings.
    model="auto",
)


class _DetectionView:
    """The minimal surface BYTETracker.update() reads off a results object.

    Ultralytics normally hands its tracker a `Boxes`. The pipeline runs
    predict() and assembles detections itself, so this adapts a plain array
    without pulling a torch tensor back through the Boxes constructor.
    """

    __slots__ = ("conf", "cls", "xywh", "xyxy")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)
        if len(xyxy):
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            self.xywh = np.stack(
                [self.xyxy[:, 0] + w / 2, self.xyxy[:, 1] + h / 2, w, h], axis=1
            )
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, mask) -> "_DetectionView":
        """Boolean-mask into another view.

        BYTETracker._split_detections indexes the results object to separate
        high- from low-confidence detections, so the adapter has to survive
        being sliced, not just read.
        """
        return _DetectionView(self.xyxy[mask], self.conf[mask], self.cls[mask])


# Camera motion moves the whole frame; traffic moves a small part of it. On a
# 64x36 thumbnail the two produce different amounts of change, and comparing
# thumbnails costs microseconds against GMC's ~10 ms.
#
# Measured over 297 successive-frame pairs from all 27 cameras at the 5 fps
# the readers sample, against known translations of a real frame:
#
#     fixed cameras, real traffic   mean 3.22, p95 11.73, max 26.65
#     pan  4 px                      3.80
#     pan 10 px                      8.82
#     pan 20 px                     16.48
#     pan 40 px                     28.38
#
# The distributions overlap: a 4 px pan is genuinely indistinguishable from a
# busy junction at this resolution. So the threshold is not a clean separator
# and is chosen for its consequences. At 8.0 the check skips GMC on 83% of
# still-camera frames and stops detecting pans below about 10 px — an error
# BoT-SORT's Kalman step absorbs, since vehicles move considerably further
# than that between frames 200 ms apart. Larger pans, the ones that actually
# swap identities, still trigger it.
_MOTION_THUMB = (64, 36)
_MOTION_THRESHOLD = 8.0
# Frames of camera displacement kept for the sway statistics. At the
# pipeline's 1 fps sampling this is a two-minute window — long enough that a
# gust shows up, short enough that the number still describes now.
_MOTION_WINDOW = 120


class BoTSORTTracker:
    """One tracker per camera. Never share an instance across cameras."""

    def __init__(self, frame_rate: int = 5, with_reid: bool = True):
        from ultralytics.trackers.bot_sort import BOTSORT

        args = dict(_TRACKER_ARGS)
        args["with_reid"] = with_reid
        self._impl = BOTSORT(SimpleNamespace(**args))

        # BOTSORT builds its GMC with the library default downscale of 2, which
        # runs optical flow over a quarter of a 1080p frame. Camera motion is a
        # global 3x3 warp and does not need that resolution.
        #
        # Choosing the method needs the real workload, not a synthetic one.
        # Translating a single frame by a known offset picked ECC at
        # downscale 8 — 9.7 ms, sub-pixel — and that choice cost 188 ms per
        # frame in the live loop. Warping one image gives ECC a scene where
        # every pixel moves together and it converges immediately; real frames
        # 200 ms apart contain vehicles moving independently of the camera, no
        # single warp explains them, and ECC runs to its iteration limit.
        #
        # Measured on successive real frames at the 5 fps the readers sample:
        #
        #                        busy junction   empty road   known-shift err
        #     sparseOptFlow ds=2     39.7 ms       46.1 ms         0.01 px
        #     sparseOptFlow ds=8     12.6 ms        9.3 ms         0.56 px
        #     orb           ds=8      4.8 ms        3.4 ms         0.57 px
        #     ecc           ds=8     10.2 ms      685.5 ms         0.38 px
        #     none                    0.0 ms        0.0 ms        20.62 px
        #
        # ECC is worst exactly where it looks safest: an empty road has little
        # texture gradient, so it never converges. ORB is fastest but unstable
        # across pan sizes — the same test at 4/12/30/60 px gave it 4-8 px
        # residual, more than a distant vehicle's own width, which is when
        # identities swap. sparseOptFlow at 1/8 scale is sub-pixel at every
        # pan size tested and costs about 10 ms.
        from ultralytics.trackers.utils.gmc import GMC
        self._impl.gmc = GMC(method="sparseOptFlow", downscale=8)
        # The library's own no-op, swapped in when the camera has not moved.
        # It returns the identity warp — which is the correct estimate for a
        # still camera — and measured 0.0 ms against sparseOptFlow's 10.
        self._gmc_none = GMC(method="none")

        # Camera motion is worth estimating only when the camera moved. Most
        # of this fleet is fixed — the installation names say so — and even a
        # PTZ is stationary between operator moves. `_camera_moved` is a
        # thumbnail comparison costing microseconds; skipping GMC on a still
        # camera removes the largest per-frame cost in the loop.
        self._gmc_ref: Optional[np.ndarray] = None
        self._gmc_runs = 0
        self._gmc_skips = 0
        # Recent frame-to-frame camera displacements, in pixels, for the
        # sway telemetry the calibration studio reports.
        self._motion_window: List[float] = []

        # Ultralytics reads frame_rate off the args in some versions and the
        # constructor in others; set it directly so the track buffer is sized
        # in frames either way.
        self._impl.frame_rate = frame_rate
        self.with_reid = with_reid
        self._frames = 0

    def update(
        self,
        detections: List[dict],
        frame: Optional[np.ndarray] = None,
        feats: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Advance the tracker by one frame.

        Args:
            detections: [{"bbox": [x1,y1,x2,y2], "cls": int, "conf": float}]
            frame: the BGR frame. Required for GMC - without it the tracker
                cannot separate camera pan from object motion.
            feats: (N, D) appearance embeddings aligned with `detections`.
                Omit to fall back to motion-only association.

        Returns:
            [{"track_id", "bbox", "cls", "conf", "det_index"}] for confirmed
            tracks. `det_index` indexes back into `detections`, so a caller
            holding per-detection crops or embeddings can line them up.
        """
        self._frames += 1

        n = len(detections)
        if n:
            xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
            conf = np.array([d["conf"] for d in detections], dtype=np.float32)
            cls = np.array([d.get("cls", 0) for d in detections], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)

        view = _DetectionView(xyxy, conf, cls)

        if feats is not None and len(feats) != n:
            logger.warning(
                "feats length %d != detections %d — dropping appearance for "
                "this frame rather than misaligning identities",
                len(feats), n,
            )
            feats = None

        gmc_frame = frame
        if frame is not None and not self._camera_moved(frame):
            # Hand GMC nothing when the camera is still. Ultralytics then
            # applies an identity warp, which is the correct answer for a
            # fixed camera and costs nothing to compute.
            gmc_frame = None
            self._gmc_skips += 1
        elif frame is not None:
            self._gmc_runs += 1

        try:
            # BOTSORT.init_track only builds ReID features when it is given a
            # frame, so keep passing the real frame there and gate GMC alone.
            prev_gmc = self._impl.gmc
            if gmc_frame is None:
                self._impl.gmc = self._gmc_none
            try:
                out = self._impl.update(view, frame, feats)
            finally:
                self._impl.gmc = prev_gmc
                self._record_camera_motion(gmc_frame is not None)
        except Exception as exc:                                   # noqa: BLE001
            # One bad frame must not take down a 27-camera loop, but it also
            # must not pass silently — a tracker that throws every frame would
            # otherwise look like a camera with no vehicles on it.
            logger.warning("BoT-SORT update failed on frame %d: %s",
                           self._frames, exc)
            return []

        out = np.asarray(out)
        if out.size == 0:
            return []

        # Ultralytics returns [x1, y1, x2, y2, track_id, conf, cls, det_index].
        results: List[Dict[str, Any]] = []
        for row in out:
            results.append({
                "track_id": int(row[4]),
                "bbox": [float(row[0]), float(row[1]),
                         float(row[2]), float(row[3])],
                "conf": float(row[5]),
                "cls": int(row[6]),
                "det_index": int(row[7]) if len(row) > 7 else -1,
            })
        return results

    def _camera_moved(self, frame: np.ndarray) -> bool:
        """Cheap test for whether the camera itself moved since the last frame.

        A pan displaces every pixel, so a thumbnail of the scene changes
        globally. Vehicles crossing a fixed view change only the part of the
        thumbnail they occupy, which at 64x36 is a few pixels. The threshold
        sits between those two regimes; when in doubt the answer is "moved",
        because running GMC unnecessarily costs time while skipping it when
        the camera did move costs identities.
        """
        import cv2
        thumb = cv2.cvtColor(
            cv2.resize(frame, _MOTION_THUMB, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.int16)

        prev, self._gmc_ref = self._gmc_ref, thumb
        if prev is None or prev.shape != thumb.shape:
            return True
        return float(np.mean(np.abs(thumb - prev))) > _MOTION_THRESHOLD

    @property
    def active_track_count(self) -> int:
        return len(self._impl.tracked_stracks)

    def _record_camera_motion(self, gmc_ran: bool) -> None:
        """Keep the frame-to-frame camera displacement GMC just estimated.

        This is a measurement that already exists and was being thrown away.
        BoT-SORT's global motion compensation solves for the 2x3 affine warp
        between consecutive frames precisely so it can separate camera motion
        from vehicle motion; the translation column of that warp is how far
        the view actually shifted, in pixels.

        It is what a mast-mounted camera's wind sway looks like in the only
        units that matter for tracking. The endpoint that reports "shake" was
        returning 1.2 + 0.8*sin(t) + 0.4*cos(t) — a function of wall-clock
        time, identical for every camera, non-zero for a camera bolted to a
        wall and unchanged by an actual gale.
        """
        if not gmc_ran:
            # A skipped frame is a still camera, which is a zero displacement
            # and belongs in the window — dropping it would bias the RMS
            # upward by only ever sampling the frames that moved.
            self._motion_window.append(0.0)
        else:
            warp = getattr(self._impl.gmc, "prevAffine", None)
            if warp is None:
                warp = getattr(self._impl.gmc, "H", None)
            try:
                dx, dy = float(warp[0, 2]), float(warp[1, 2])
                self._motion_window.append(float(np.hypot(dx, dy)))
            except Exception:                                     # noqa: BLE001
                return
        if len(self._motion_window) > _MOTION_WINDOW:
            self._motion_window.pop(0)

    @property
    def gmc_stats(self) -> dict:
        """How often motion compensation was actually needed."""
        total = self._gmc_runs + self._gmc_skips
        return {
            "gmc_runs": self._gmc_runs,
            "gmc_skips": self._gmc_skips,
            "gmc_run_pct": round(100.0 * self._gmc_runs / max(1, total), 1),
        }

    @property
    def camera_motion(self) -> dict:
        """Measured frame-to-frame camera displacement over a recent window.

        Returns None for the statistics until enough frames have passed to
        compute them, rather than a placeholder — a caller that cannot tell
        "not measured yet" from "measured zero" will report the second.
        """
        w = self._motion_window
        if len(w) < 8:
            return {"samples": len(w), "jitter_rms_px": None,
                    "max_shift_px": None, "moving_frame_pct": None,
                    "status": "measuring"}
        a = np.asarray(w, dtype=np.float64)
        rms = float(np.sqrt(np.mean(a ** 2)))
        moving = float(np.mean(a > 1.0)) * 100.0
        # Thresholds in pixels of frame-to-frame shift. Below 1 px the view is
        # steady enough that tracking is unaffected; beyond 8 px a vehicle box
        # can move further from camera motion than from its own travel, which
        # is when identities start swapping.
        if rms < 1.0:
            status = "STABLE"
        elif rms < 8.0:
            status = "SWAY"
        else:
            status = "UNSTABLE"
        return {
            "samples": len(w),
            "jitter_rms_px": round(rms, 2),
            "max_shift_px": round(float(a.max()), 2),
            "moving_frame_pct": round(moving, 1),
            "status": status,
        }
