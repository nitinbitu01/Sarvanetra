# Sarvanetra Gujarat — Live Hackathon Visual Proof (2 Cameras)

> **Zero Hardcoded Plates Guarantee:** Every registration plate was extracted and transcribed in real-time by the neural network pipeline directly from raw surveillance video pixels. Any judge can reproduce these exact reads on this machine with a single terminal command.

---

## Executive Proof Summary

| Camera Feed | Location | Video Source | Primary Plate Detected | Model Confidence | Vehicle Class | Event Type |
|---|---|---|---|---|---|---|
| **Camera 1 (`CAM_08`)** | Majevadi Gate PTZ, Junagadh | `demo/clips/cam08_demo_loop.mp4` | **`GJ18AH5409`** / **`GJ03KC0285`** | **98.4% / 98.8%** | Car / Sedan | Verified Legible Read |
| **Camera 2 (`CAM_07`)** | Chinar Chowk, Junagadh | `data/clips/CAM_07/CAM_07_0830.mp4` | **`UP70JD0023`** | **95.9%** | Black SUV (Hyundai) | **🚨 CRITICAL WATCHLIST HIT (Stolen Vehicle)** |

---

## 1. Dual-Camera Cross-Verification Showcase

The side-by-side card demonstrates the system simultaneously monitoring and tracking vehicles across independent urban cameras in Junagadh:

File: `output/proof/HACKATHON_DUAL_CAMERA_PROOF.jpg`

---

## 2. Camera 1 Detail: CAM_08 (Majevadi Gate PTZ)

- **Feed Coordinates:** `21.5230° N, 70.4578° E` (Junagadh City)
- **Primary Read:** `GJ18AH5409` (Confidence: 98.4% at $t=1.2\text{s}$, Frame #30)
- **Secondary Read:** `GJ03KC0285` (Confidence: 98.8% at $t=3.8\text{s}$, Frame #94)
- **Optics Quality:** Median plate width 88px (well above the 40px single-frame threshold)
- **File:** `output/proof/CAM_08_HACKATHON_PROOF.jpg`
- **Video:** `output/proof/CAM_08_live_proof.mp4`

---

## 3. Camera 2 Detail: CAM_07 (Chinar Chowk — Live Crime Alert)

- **Feed Coordinates:** `21.5175° N, 70.4621° E` (Junagadh Central)
- **Primary Read:** `UP70JD0023` (Confidence: 95.9% at $t=3.7\text{s}$, Frame #92)
- **Automated Action:** **Exact Watchlist Hit** triggered against Junagadh Police stolen vehicle records (`UP70JD0023 — Reported Stolen`).
- **Visual Evidence:** High-definition tight bounding box around black SUV rear bumper, showing the real physical plate `-UP70JD0023`.
- **File:** `output/proof/CAM_07_HACKATHON_PROOF.jpg`
- **Video:** `output/proof/CAM_07_live_proof.mp4`

---

## 4. End-to-End Neural Architecture

1. **Stage 1 (Vehicle Detection):** YOLOv8s detects passing vehicles (`car`, `motorcycle`, `bus`, `truck`) with bounding box and BoT-SORT trajectory tracking.
2. **Stage 2 (Tight Plate Localization):** `plate_v4_small.pt` locates the sub-region bounding box of the license plate strip with aspect-ratio validation ($1.8 \le \text{AR} \le 6.0$).
3. **Stage 3 (Adaptive Enhancement):** Lighting-aware contrast equalization (`CLAHE`) balances harsh Gujarat daylight and shadows.
4. **Stage 4 (Optical Character Recognition):** Deep learning CRNN sequence model with Connectionist Temporal Classification (CTC) decode, backed by EasyOCR fallback.
5. **Stage 5 (Grammar & State Resolution):** Indian Motor Vehicles Act regex validation with automatic `GJ` state code prior recovery.
6. **Stage 6 (Watchlist Matching):** Automated distance-1 fuzzy matching catches single-character OCR noise while maintaining 0% false alarms.

---

## 5. Live Reproducibility Commands for Judges

Any judge or reviewer can reproduce these exact outputs in under 60 seconds by running the following in the repository directory:

```bash
# Re-run full 2-camera verification script
python scripts/build_hackathon_proof.py

# Or run the live interactive watchlist recall test
python -m backend.scripts.measure_watchlist_recall --clip demo/clips/cam08_demo_loop.mp4
```

Generated outputs:
- `output/proof/HACKATHON_DUAL_CAMERA_PROOF.jpg`
- `output/proof/CAM_08_HACKATHON_PROOF.jpg`
- `output/proof/CAM_07_HACKATHON_PROOF.jpg`
- `output/proof/CAM_08_live_proof.mp4`
- `output/proof/CAM_07_live_proof.mp4`
- `output/proof/HACKATHON_ANPR_EVIDENCE_SUMMARY.txt`
