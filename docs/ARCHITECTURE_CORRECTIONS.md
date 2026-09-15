# Sentinel Architecture Document — Corrections

Audit of the Sentinel production architecture plan (Sections 0–4), against a
document whose own stated purpose is to remove unsourced figures and show its
derivations.

Every corrected number here is computed, not asserted. The computation lives in
`tests/test_bandwidth_math.py` (15 tests) and can be re-run:

```
python -m pytest tests/test_bandwidth_math.py -v
```

**Scope.** This is a corrections list keyed to section numbers, not a rewritten
document — the source was truncated mid-`tpm_preseal.py`, so Sections after
`_find_segment()` were not reviewed. See *Not reviewed* at the end.

---

## Summary

| # | Section | Claim | Corrected | Severity |
|---|---------|-------|-----------|----------|
| 1 | 1.3 | 1,180 B mean event | **2,175 B** | Changes every figure in 1.3 |
| 2 | 1.3 | 275 Mbps statewide | **428 Mbps** | Headline claim survives |
| 3 | 3.2 | 56,000 Milvus QPS | **160,000 QPS** | Hides a 1.6–3.2× shortfall |
| 4 | 3.3 | ₹212 Cr CAPEX | **₹167.7 Cr** | 3 conversion errors |
| 5 | 3.3 | Bengaluru ₹66,000/cam | **₹662,093/cam** | Understates our own case 10× |
| 6 | 0 vs 3.3 | ₹160–200 Cr vs ₹212 Cr | §0 is correct | Internal contradiction |

Findings 1–2 and 4–5 are arithmetic. Finding 3 is a methodology error that
conceals a capacity problem. Section 4 code defects are listed separately.

---

## 1. Event payload is 1.85× larger than stated — §1.3 Step 1

**Document:** 890 / 1,180 / 1,400 bytes (min/mean/max), justified by the
`SentinelEvent` docstring: *"embedding_int8: 512 int8 values = 512 bytes."*

**Defect:** That is the **binary** size. `to_json()` calls `json.dumps`, which
emits the embedding as a JSON array of integers — each value costs 1–4
characters of text plus a separator. The document reasons in one encoding and
ships another.

**Measured**, constructing the dataclass exactly as §4 defines it:

| | Document | Measured |
|---|---|---|
| Embedding array alone | 512 B implied | **1,659 B** |
| Full event (vehicle + plate) | 1,180 B mean | **2,175 B** |

The embedding alone exceeds the document's stated *maximum* for the whole event.

**Note on method.** Payload size depends on the *distribution* of the int8
values, not just the count — a value in 0..9 costs one character, one in
−128..−10 costs four. A 512-dim L2-normalised vector has components
~N(0, 1/√512); `_quantize_embedding` scales max_abs onto 127, leaving sd ≈ 36
in int8 space. Testing with zeros understates the result by more than half;
a uniform spread over the full range overstates it (≈2,390 B). The figure above
uses the Gaussian.

**Recommended fix:** base64-encode the embedding — 512 bytes → 684 characters,
against 1,659 as a JSON int array. That removes ~975 bytes per event and
returns the statewide figure close to the document's original estimate,
honestly. `test_base64_encoding_would_restore_the_original_estimate` asserts
the remedy is real rather than leaving it as a suggestion.

## 2. Corrected bandwidth — §1.3 Steps 4–6

Same derivation chain, corrected payload:

```
80,000 cameras × 2 events/sec × 2,175 B × 8      = 2,784 Mbps raw
÷ 5    (HTTP/2 + zstd on JSON batches)           =   557 Mbps
÷ 1.3  (Kafka snappy, already-compressed)        =   428 Mbps
```

| | Document | Corrected |
|---|---|---|
| Statewide metadata | 275 Mbps | **428 Mbps** |
| Per district (÷34) | 8.1 Mbps | **12.6 Mbps** |
| Share of 250 Mbps uplink | ~3% | **5.0%** |
| Reduction vs 160 Gbps raw | 99.83% | **99.73%** |

**The headline claim survives.** 99.73% clears the "99.7–99.85%" band §0
commits to, and 5.0% of a shared district uplink is still comfortable under
QoS. Only the specific numbers in §1.3 need replacing — the argument does not.

**§1.1 and §1.2 verify exactly** and need no change: 160 Gbps raw, 2,353
cameras/district, 4,706 Mbps against a 250–500 Mbps uplink, 10–19× overflow.
This is the strongest quantitative argument in the document.

## 3. Deduplication counted twice, concealing a capacity shortfall — §3.2

**Document:** *"80,000 × 2 events/sec = 160,000 searches/sec. After dedup:
~56,000 searches/sec."*

**Defect:** `2 events/sec` is **already** the post-dedup figure. §1.3 Step 3
derives it: `5 × (1 − 0.65) = 1.75`, rounded up to 2. §3.2 applies the same 65%
reduction a second time.

**Consequence:** true Milvus load is **160,000 QPS**, against the document's
own stated capacity of 50,000–100,000 QPS — a **1.6× to 3.2× shortfall**. The
double-count is precisely what makes the central sizing appear adequate.

**Resolution required** — one of:
- scale the Milvus cluster ~3× (more nodes/shards; it is CPU+RAM bound, so
  this is a cost line, not a redesign), or
- reduce events/sec with a justified, measured dedup rate, or
- add a district-tier pre-filter so not every event reaches central ANN search.

§3.2's central insight — that this is a **CPU + RAM workload, not an A100
workload** — is correct and is the most valuable correction in the document.
Only the QPS arithmetic is wrong.

## 4. CAPEX — three Lakh→Crore conversion errors — §3.3

100 Lakh = 1 Crore. Three rows do not follow from unit × quantity:

| Item | Unit | Qty | Stated | Correct | Error |
|---|---|---|---|---|---|
| AGX Orin edge box | ₹2.1 L | 3,500 | ₹73.5 Cr | ₹73.5 Cr | ✓ |
| Orin NX edge box | ₹1.4 L | 3,000 | ₹42.0 Cr | ₹42.0 Cr | ✓ |
| District hubs | ₹18 L | 32 | ₹5.8 Cr | ₹5.76 Cr | rounding |
| Central CPU servers | ₹35 L | 4 | ₹14.0 Cr | **₹1.4 Cr** | 10× |
| Central GPU servers | ₹40 L | 2 | ₹8.0 Cr | **₹0.8 Cr** | 10× |
| Ceph cluster | ₹25 L | 1 | ₹25.0 Cr | **₹0.25 Cr** | 100× |
| Networking | — | — | ₹8.0 Cr | *no basis given* | — |

**Why this survived review:** the stated column sums correctly to ₹176.3 Cr.
Checking the total cannot find it; only unit × quantity can.

```
Hardware   ₹131.7 Cr   (was ₹176.3 Cr)
Software    ₹36.0 Cr
─────────────────────
TOTAL      ₹167.7 Cr   (was ₹212 Cr)
AMC @15%    ₹19.8 Cr/yr (was ₹27 Cr/yr)
```

**Also:** the ₹8 Cr networking line is the only hardware row without a
unit-cost basis. A reviewer who finds the three errors above will ask about it.

**§3.1 verifies** and needs no change (30÷2 = 15, 45÷2 = 22.5; 3,500 + 3,000 =
6,500 boxes). One inconsistency: cameras-per-AGX-box is stated as 16–24 in §0
and §2.1, 15–22 in the §3.1 table, and 16 in the planning figures. Pick one.

## 5. The Bengaluru comparison is 10× against us — §3.3

The document computes Bengaluru correctly at *"~₹6.6 Lakh per camera"*, then
compares against **"₹66,000 Bengaluru greenfield."** ₹6.6 Lakh is ₹660,000.

| | Per camera |
|---|---|
| Bengaluru Safe City (₹496.57 Cr ÷ 7,500) | **₹662,093** |
| Sentinel (₹167.7 Cr ÷ 80,000) | **₹20,963** |

The real advantage is **31.6×**, not the 2.5× the stated comparison implies.
This error works against the document's own case.

## 6. §0 contradicts §3.3

§0 states the corrected CAPEX as **₹160–200 Cr**. §3.3 concludes **₹212 Cr**
with a range of ₹175–260 Cr. These cannot both stand.

**§0 is the one that is right** — the corrected total of ₹167.7 Cr falls inside
it. Update §3.3 to match.

---

## Section 4 — code defects

### Critical

**C1. `_should_send` requires movement AND embedding drift.** Both guards
return `False` early, so a track is re-sent only if it moved *and* its
appearance changed. A person walking steadily in unchanged clothing has a
stable embedding, so after the first sighting they are **never re-sent** —
which silently disables cross-camera journey tracking, the product's core
function. Should be OR.

Secondary effect: real dedup would be ~99%, not 65%, invalidating §1.3 Step 3
in the opposite direction from finding #1.

**C2. The API-mismatch bug §0 claims to have fixed is still present.**
`_load_reid_model` calls `torch.load()` on `models/osnet_x025.engine` — a
serialised TensorRT plan, not a pickle. It will raise. The comment directly
above it reads *"Same Ultralytics API — consistent, no mismatch"* while calling
a method that uses neither Ultralytics nor TensorRT.

§0 correction #8 ("TensorRT + Ultralytics API mixed (code won't run) → Fixed")
holds for the detector and not for the ReID model.

**C3. `strict=False` can yield a silently random model.** If the checkpoint
keys do not match, `load_state_dict(..., strict=False)` leaves layers
randomly initialised and raises nothing — producing garbage embeddings, in a
system whose entire value is embedding similarity. Load strictly, or verify a
known-answer embedding at startup.

### Serious — chain of custody

**C4. The edge cannot know the score that triggers pre-sealing.**
`tpm_preseal.py` says it fires *"the moment a score-threshold event is detected
locally"*, but §2.1 places the danger scorer at Tier 3 and
`detection_pipeline.py` computes no score. The trigger as described does not
exist. Either give the edge a local threshold rule, or pre-seal every buffered
segment unconditionally.

**C5. Dev-mode pre-seals are indistinguishable from TPM pre-seals.** If
`tpm2-pytss` is merely not installed, `_tpm_sign` silently falls back to an
HMAC with a hardcoded key — and `PresealRecord` carries **no field recording
which path signed it**. Central cannot tell a hardware-attested seal from a
development one, which defeats the module's only purpose. Add a `signer` field;
refuse to start in production without a TPM.

**C6. Hash-before-write race.** The pre-seal hashes a ring-buffer segment that
may still be being written. The hash will not reproduce when the segment is
later pulled.

**C7. 72-hour ring buffer versus indefinite pre-seals.** A pre-seal referencing
a segment pulled on day 4 points at overwritten bytes. Creating a pre-seal must
pin its segment against circular overwrite.

**C8. Signature omits `box_id` and `clip_ref_id`.** The signed message is
`{event_id}:{seg_hash}:{camera_id}:{timestamp}`, so a seal from one box can be
replayed as another's.

**C9. Whole-segment read into memory.** `f.read()` on a video segment; hash in
chunks.

### Minor

- `process_camera_frame` is `async`, but YOLO, torch and cv2 calls all block;
  only `enqueue` awaits. The async is decorative and will stall the event loop.
- `track_states` and `activity_scores` are never evicted — unbounded growth on
  a box running for weeks.
- `next_skip` is computed then discarded (*"caller reads this via attribute"* —
  no attribute is set). Adaptive skip is dead code.
- `T.Compose` is rebuilt per crop inside `_extract_reid_embedding`.
- `_quantize_embedding` hardcodes `[0] * 512` on the degenerate path,
  assuming a dimensionality the model is not checked against.

---

## Unsourced figures

§0 commits to removing asserted numbers. These remain, introduced as
*"measured"* with no source:

- **60–85% ReID accuracy** (§2.2) — load-bearing: it is the justification for
  the human review gate.
- **60–70% deduplication rate** (§1.3 Step 3) — load-bearing for all of §1.3.
- **5 events/sec average** (§1.3 Step 2).

Each needs a citation or a "to be measured during PoC" marker.

## Smaller inconsistencies

- Score bands leave **0.00–0.34 undefined** (§2.2 starts at 0.35).
- §2.1 says video *"NEVER crosses any network boundary"* without
  authorisation, while `ring_buffer.write_frame()` runs on every processed
  frame. These are compatible — the buffer is local — but the defensible
  phrasing is *"never leaves the edge box"*, which is what the code guarantees.

---

## What holds up

- **§1.1 / §1.2** — the district-overflow argument verifies exactly and is the
  strongest quantitative claim in the document.
- **§3.1** — box counts and camera-per-box derivations are internally
  consistent.
- **§3.2's classification** of the central workload as CPU+RAM rather than GPU.
- **§2.2 / §2.3** — the Puttaswamy proportionality reasoning, the BSA §63
  certification argument, and the Category A/B/C data model. This is the
  strongest section overall and needs no correction.
- The **99.7%+ bandwidth reduction claim** itself, after finding #1.

## Not reviewed

The source was truncated mid-`tpm_preseal.py::_find_segment()`. Not covered:

- the remainder of `tpm_preseal.py`, `mtls_client.py`, `ota_receiver.py`
- all of `central/` — `kafka_consumer`, `milvus_indexer`, `reid_engine`,
  `geo_filter`, `danger_scorer`, `human_review_queue`, `alert_dispatcher`,
  `vault_manager`, `fleet_manager`, `ota_manager`
- `apps/officer-app` — including the JWT + badge-number auth that §0
  correction #10 claims to have fixed, which is therefore **unverified**
- `infra/`, the `tests/` suite, and any sections beyond §4
