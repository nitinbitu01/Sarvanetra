# Can this system read number plates? — measured answer

**Date:** 2026-08-27  **Verdict: no, not at current camera geometry.**

This exists because "add ANPR" keeps being proposed, and arithmetic alone was
not settling it. Two tests were run against real daylight footage.

## Test 1 — vehicle crop → upscale → OCR

`backend/scripts/plate_feasibility.py`, 120 vehicle crops from CAM_04/08/20/19
daylight clips. Vehicles cropped from the **original 1920px frame** (not the
1280 inference downscale), upscaled to 640px wide, EasyOCR.

```
vehicle crops examined       : 120
crops with ANY text          : 2   (2%)
crops with PLATE-FORMAT text : 0   (0%)
median vehicle width         : 120 px (source)
```

## Test 2 — tight plate band → 800px upscale → sharpen → OCR

Test 1 was not a fair test of the cascade: it ran OCR across a whole car and
hoped it would find a small plate. Test 2 crops only the lower-centre band
where a plate sits, upscales it to 800px, and sharpens before OCR.

```
plate bands examined      : 150
bands with ANY text       : 7   (5%)
bands with PLATE format   : 0   (0%)
band height BEFORE upscale: median 56px, p90 143px
  -> the plate itself is ~1/4 of that = ~14px tall
```

## Why it fails — the number that decides it

**A plate is ~14px tall in the source frame. Characters are ~7px.**
OCR needs 20–25px characters.

Upscaling does not fix this. Interpolating a 7px character to 50px produces a
50px-tall blur, not a readable glyph — upscaling makes pixels bigger, it does
not add information that the sensor never captured. The saved crops show this
directly: a plate-shaped white rectangle is clearly visible with character-like
smudges inside it, and the characters are unrecoverable.

This is a **camera geometry limit, not a software limit**. At 57 px/metre the
information required to read a plate was never in the video file. No model,
no OCR engine, and no super-resolution network recovers it, because the
correct characters and a dozen wrong ones are all consistent with those 7px.

## What would change the answer

Only one thing: **more pixels on the plate.** Either
- a dedicated ANPR camera at the lane, angled down, tightly zoomed (this is
  what real ANPR deployments use — plates fill 100–200px), or
- an existing camera's PTZ zoomed onto one approach lane, sacrificing its
  wide-area view.

Nothing achievable in software gets there.

## What the system does instead

The 72 existing `anpr_uncertain` alerts are, correctly, the system saying "a
vehicle was here and its plate could not be read." That is the honest output.
Vehicles are tracked by appearance and trajectory, not identity.

**Do not claim plate reading in the demo or slides.** A judge who asks the
system to read a plate on screen will get nothing, and the credibility cost
of that far exceeds the value of the claim. Stating the pixel limit and
recommending ANPR cameras where plate capture is a requirement is the
stronger position — it is what a real integrator would say.
