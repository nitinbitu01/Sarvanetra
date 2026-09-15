# Site measurement request — camera calibration

**25 cameras.** Two measurements, both doable in one visit of about fifteen minutes. **The per-camera table below says which of the two each camera actually needs** — several already have a height on record and need only the ground points.

## What to measure, per camera

### 1. Mounting height  *(where the table asks for it)*

Vertical distance from the **road surface directly below the camera** to the **centre of the lens**, in metres to the nearest 0.05 m.

A laser rangefinder pointed down from the mount is ideal. A tape from the pole base works if the camera is on a straight pole — note it if the mount is on an arm, because then the lens is not above the pole base and the horizontal offset matters too.

### 2. Four to six ground reference points  *(this is the valuable part)*

Pick features on the **road surface** that are permanent and clearly visible in the camera's view — a lane-marking corner, a kerb corner, the end of a zebra stripe, a manhole rim.

Then measure the **distance between each pair** (or from one chosen origin point to each of the others), in metres to the nearest 0.1 m.

Please **spread the points across the whole area the camera watches**, near and far, left and right. Four points bunched in one corner are worth much less than four spread out.

A photo of the scene with the chosen points marked, alongside the numbers, removes all ambiguity about which feature is which.

### 3. Is the road flat?  *(one line, and it matters)*

The maths behind this maps one **flat plane** to another. A bend in the road is fine - the surface is still flat. What breaks it is the road going up or down within the camera's view: a **crest**, a **dip**, or a strongly **cambered** (banked) surface.

So please add one line per camera: `road: flat` or `road: crest` / `dip` / `camber`.

**If it is not flat, take 7-8 points instead of 4-6**, and keep them on the flattest stretch you can that still spans the view. The extra points let us detect the curvature and fit around it rather than silently absorbing it as error - a sloped road fitted as a flat one gives confident, wrong distances, which is worse than no calibration at all.

This cannot be judged reliably from satellite imagery - a crest and a camber both look flat from directly overhead. Standing there is the only good way to answer it, which is why it is on this list.

### Why both

The height alone unblocks one stage of processing. The ground points complete the calibration outright, and give position accuracy an order of magnitude better. The extra effort is a few tape measurements.

---

## Cameras

| Camera | Location | District / Zone | GPS | Blocker | What to measure |
|---|---|---|---|---|---|
| **CAM_01** | Ahmedabad - Chimanbhai Bridge | Ahmedabad / Central | 23.0395, 72.5797 | `needs_height` | height **and** ground points |
| **CAM_03** | Ahmedabad - O.N.G.C. Office | Ahmedabad / Central | 23.0269, 72.6019 | `no_vp1` | height **and** ground points |
| **CAM_05** | Ahmedabad - Visat Teen Rasta | Ahmedabad / North | 23.0600, 72.5806 | `not_one_road` | height **and** ground points |
| **CAM_07** | Gir Somnath - Hero Showroom | Gir Somnath / Saurashtra | 20.8950, 70.4100 | `needs_height` | height **and** ground points |
| **CAM_10** | Junagadh - Char Chowk Road | Junagadh / Saurashtra | 21.5250, 70.4550 | `needs_height` | height **and** ground points |
| **CAM_12** | Gandhinagar - Tri Mandir Adalaj | Gandhinagar / Capital | 23.1667, 72.5800 | `not_one_road` | height **and** ground points |
| **CAM_13** | Ahmedabad - CN Vidhyalaya | Ahmedabad / Central | 23.0270, 72.5500 | `needs_height` | height **and** ground points |
| **CAM_14** | Ahmedabad - Delight Chowk | Ahmedabad / Central | 23.0150, 72.5600 | `needs_height` | height **and** ground points |
| **CAM_18** | Rajkot - CCTV Station | Rajkot / Saurashtra | 22.3000, 70.8000 | `needs_height` | height **and** ground points |
| **CAM_19** | Navsari - Khaparia Gram Panchayat | Navsari / South Gujarat | 20.8500, 72.9200 | `no_vp1` | height **and** ground points |
| **CAM_20** | Patan - Mohanpura | Patan / North Gujarat | 23.8500, 72.1200 | `no_vp1` | height **and** ground points |
| **CAM_23** | Sabarkantha - Kheram | Sabarkantha / North Gujarat | 23.6000, 72.9500 | `no_vp1` | height **and** ground points |
| **CAM_24** | Gandhinagar - Dehgam Circle | Gandhinagar / Capital | 23.1700, 72.8200 | `no_vp1` | height **and** ground points |
| **CAM_25** | Navsari - Dhanori Junction | Navsari / South Gujarat | 20.9200, 72.9600 | `needs_height` | height **and** ground points |
| **CAM_26** | Navsari - Tankal | Navsari / South Gujarat | 20.9300, 72.9700 | `no_vp1` | height **and** ground points |
| **CAM_27** | Bilimora - Main Chowk | Navsari / South Gujarat | 20.7686, 72.9618 | `no_vp1` | height **and** ground points |
| **CAM_28** | Bilimora - Highway Junction | Navsari / South Gujarat | 20.7700, 72.9630 | `no_vp1` | height **and** ground points |
| **CAM_29** | Bilimora - Railway Overbridge | Navsari / South Gujarat | 20.7720, 72.9600 | `no_vp1` | height **and** ground points |
| **CAM_30** | Gandhidham - Rambaugh P2 Checkpost | Kutch / Kutch | 23.0900, 70.1400 | `no_vp1` | height **and** ground points |
| **CAM_02** | Ahmedabad - Janpath | Ahmedabad / Central | 23.0225, 72.5714 | `needs_legs` | **ground points only** — height already on record |
| **CAM_04** | Ahmedabad - Paldi Circle | Ahmedabad / East | 23.0876, 72.6461 | `needs_legs` | **ground points only** — height already on record |
| **CAM_06** | Junagadh - Timbavadi Gate | Junagadh / Saurashtra | 21.5100, 70.4400 | `needs_legs` | **ground points only** — height already on record |
| **CAM_09** | Junagadh - New Bypass Circle | Junagadh / Saurashtra | 21.5300, 70.4600 | `needs_legs` | **ground points only** — height already on record |
| **CAM_16** | Ahmedabad - Visat P2 | Ahmedabad / North | 23.0650, 72.5850 | `needs_legs` | **ground points only** — height already on record |
| **CAM_21** | Patan - Dethali Char Rasta | Patan / North Gujarat | 23.8600, 72.1300 | `needs_legs` | **ground points only** — height already on record |

For the 6 camera(s) needing **ground points only**, the height is already in our records - please do not spend the visit re-measuring it.
Where both are asked for, the ground points are the part that finishes the camera. If time runs short, skip a whole camera rather than doing half of one - half a camera cannot be calibrated, and we would not know the visit had happened.

## Suggested trips

These cluster geographically, so the list is a handful of journeys rather than 25 separate ones:

| District | Cameras | Count |
|---|---|---:|
| Ahmedabad | CAM_01, CAM_02, CAM_03, CAM_04, CAM_05, CAM_13, CAM_14, CAM_16 | 8 |
| Navsari | CAM_19, CAM_25, CAM_26, CAM_27, CAM_28, CAM_29 | 6 |
| Junagadh | CAM_06, CAM_09, CAM_10 | 3 |
| Gandhinagar | CAM_12, CAM_24 | 2 |
| Patan | CAM_20, CAM_21 | 2 |
| Gir Somnath | CAM_07 | 1 |
| Rajkot | CAM_18 | 1 |
| Sabarkantha | CAM_23 | 1 |
| Kutch | CAM_30 | 1 |

The largest cluster is worth doing first: it is the most cameras finished per journey.

## Sending the results back

Plain text or a spreadsheet is fine. Per camera:

```
CAM_xx
  mounting_height_m: 6.20    # omit if the table says ground points only
  arm_offset_m: 0.0          # 0 if the lens is directly above the pole base
  road: flat                 # or crest / dip / camber -> then 7-8 points
  points:
    A -> B: 3.50            # e.g. lane width
    A -> C: 12.00
    B -> D: 9.20
  photo: CAM_xx_points.jpg   # scene with A, B, C, D marked
```

Points are entered into the calibration studio against the camera's own frame, so their identity in the photo is what matters — not any particular naming.
