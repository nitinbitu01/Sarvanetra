"""
test_bandwidth_math.py — makes the architecture document's bandwidth
derivation executable.

WHY THIS EXISTS
───────────────
The Sentinel architecture document rests its central claim — ">99.8% WAN
reduction versus streaming video centrally" — on one number: the serialised
size of a SentinelEvent. That number was asserted (890 / 1,180 / 1,400 bytes)
rather than measured, and it was wrong by 1.9x, because the event's docstring
reasoned about the embedding in BINARY ("512 int8 values = 512 bytes") while
`to_json()` emits it as a JSON array of integers (~1,711 bytes).

A prose document cannot catch that. A test can, and only if it constructs the
real dataclass and measures the real serialiser — which is what this does.

It also pins the arithmetic chain. The document's Section 3.2 applied the 65%
deduplication factor a SECOND time to a figure that already had it baked in,
which turned a 1.6-3.2x Milvus capacity shortfall into an apparent surplus.
That class of error survives any number of re-readings and dies immediately
to an assertion.

These tests encode the CORRECTED figures. If someone adds a field to
SentinelEvent, changes the embedding encoding, or re-derives the QPS, the
failure names the section of the document that needs updating with it.

Run:  python -m pytest tests/test_bandwidth_math.py -v
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import pytest

# ─── Deployment constants (architecture document, Sections 1.1-1.3) ────────
CAMERAS                = 80_000
RAW_BITRATE_MBPS       = 2.0     # 1080p H.264, busy traffic, upper bound
RAW_TOTAL_MBPS         = CAMERAS * RAW_BITRATE_MBPS      # 160 Gbps
DISTRICTS              = 34
DISTRICT_UPLINK_MBPS   = 250     # GSWAN lower bound; shared, not dedicated

RAW_EVENTS_PER_SEC     = 5.0     # per camera, before dedup
DEDUP_RATE             = 0.65    # measured 60-70%; 65% used throughout
EVENTS_PER_SEC         = 2       # POST-dedup, rounded up from 1.75

ZSTD_RATIO             = 5.0     # HTTP/2 + zstd on JSON batches
SNAPPY_RATIO           = 1.3     # Kafka, on already-compressed batches


# ─── SentinelEvent, exactly as the document's Section 4 defines it ─────────

@dataclass
class SentinelEvent:
    event_id:        str
    camera_id:       str
    box_id:          str
    department:      str
    district:        str
    timestamp_epoch: float
    location_lat:    float
    location_lng:    float
    zone_risk:       float
    object_type:     str
    track_id:        int
    bbox_norm:       List[float]
    confidence:      float
    embedding_int8:  List[int]
    embedding_scale: float
    plate_number:    Optional[str]
    plate_state:     Optional[str]
    plate_conf:      Optional[float]
    clip_ref_id:     str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(',', ':'))

    def size_bytes(self) -> int:
        return len(self.to_json().encode())


def realistic_embedding() -> List[int]:
    """
    512 int8 values with the distribution a real quantised embedding has —
    deterministic, so the byte counts below are reproducible.

    Getting this distribution right matters, because JSON encodes each integer
    as text: a value in 0..9 costs one character, one in -128..-10 costs four.
    So the payload size depends on the SHAPE of the distribution, not just the
    dimensionality.

    A 512-dim L2-normalised vector has components ~N(0, 1/sqrt(512)), and
    `_quantize_embedding` scales by max_abs/127 — which maps roughly 3.5
    standard deviations onto 127, leaving sd ~36 in int8 space. So:

      - all-zeros would understate the size by more than half
      - a uniform spread over -128..127 overstates it (~2,390 B), because real
        components cluster well inside the range

    Neither is the number to design a WAN around. A seeded Gaussian is.
    """
    import random
    rng = random.Random(7)
    return [max(-128, min(127, int(rng.gauss(0, 36)))) for _ in range(512)]


def sample_event() -> SentinelEvent:
    """A vehicle event with a plate — the largest routine payload."""
    return SentinelEvent(
        event_id="a3f9c1d2e4b5a6c7",
        camera_id="AHM-TRF-01423",
        box_id="BOX-AHM-0087",
        department="traffic_police",
        district="Ahmedabad",
        timestamp_epoch=1755882123.456789,
        location_lat=23.0489123,
        location_lng=72.5714567,
        zone_risk=0.67,
        object_type="vehicle",
        track_id=104857,
        bbox_norm=[0.1234567, 0.2345678, 0.3456789, 0.4567890],
        confidence=0.8734,
        embedding_int8=realistic_embedding(),
        embedding_scale=0.0078431,
        plate_number="GJ01AB1234",
        plate_state="GJ",
        plate_conf=0.91,
        clip_ref_id="AHM-TRF-01423_1755882120_seg0042",
    )


# ─── The finding this file was written for ────────────────────────────────

class TestPayloadSize:
    """Section 1.3 Step 1 and the SentinelEvent docstring."""

    def test_embedding_as_json_is_not_512_bytes(self):
        """
        The document's docstring says "512 int8 values = 512 bytes". That is
        true of the binary form and false of what actually crosses the wire.
        """
        emb = realistic_embedding()
        as_json = len(json.dumps(emb, separators=(',', ':')).encode())
        assert len(emb) == 512
        assert as_json > 3 * 512, (
            f"JSON-encoded embedding is {as_json} B, not 512 B — the document "
            f"reasons about the binary size but ships json.dumps()"
        )

    def test_event_exceeds_the_documented_upper_bound(self):
        """Section 1.3 claims a 1,400-byte maximum. It is not attainable."""
        size = sample_event().size_bytes()
        assert size > 1400, (
            f"Event is {size} B. If this now passes under 1400 B the encoding "
            f"changed — re-derive Section 1.3 Steps 4-6 to match."
        )

    def test_measured_size_matches_the_corrected_figure(self):
        """Corrected Section 1.3 Step 1: ~2,200 bytes, not 1,180."""
        assert 2_100 <= sample_event().size_bytes() <= 2_350

    def test_base64_encoding_would_restore_the_original_estimate(self):
        """
        The recommended fix. Not asserting the system does this — asserting
        the remedy is real, so the tradeoff is checkable rather than asserted.
        """
        emb = realistic_embedding()
        as_json = len(json.dumps(emb, separators=(',', ':')).encode())
        as_b64 = len(base64.b64encode(bytes((v + 256) % 256 for v in emb)))
        assert as_b64 < as_json / 2
        assert as_b64 < 700


# ─── The derivation chain ─────────────────────────────────────────────────

def metadata_mbps(event_bytes: float) -> float:
    """Sections 1.3 Steps 4-6: raw -> zstd -> snappy."""
    raw = CAMERAS * EVENTS_PER_SEC * event_bytes * 8 / 1e6
    return raw / ZSTD_RATIO / SNAPPY_RATIO


class TestBandwidthDerivation:

    def test_dedup_is_already_applied_to_events_per_sec(self):
        """
        EVENTS_PER_SEC is post-dedup. Section 3.2 forgot this and applied the
        factor a second time. Pinning the relationship here is what stops that
        recurring.
        """
        assert RAW_EVENTS_PER_SEC * (1 - DEDUP_RATE) == pytest.approx(1.75)
        assert EVENTS_PER_SEC == 2  # 1.75 rounded up for margin

    def test_corrected_statewide_load(self):
        """Corrected Section 1.3: ~439 Mbps, not the documented 275 Mbps."""
        assert 400 <= metadata_mbps(sample_event().size_bytes()) <= 470

    def test_headline_reduction_claim_still_holds(self):
        """
        The specific numbers in Section 1.3 are wrong; the CLAIM survives.
        This is the assertion that matters for the pitch — it must clear the
        99.7% floor the document commits to in Section 0.
        """
        reduction = 100 - (metadata_mbps(sample_event().size_bytes())
                           / RAW_TOTAL_MBPS * 100)
        assert reduction > 99.7, f"Reduction fell to {reduction:.2f}%"

    def test_per_district_still_fits_the_gswan_uplink(self):
        """
        The real constraint. Section 1.2 establishes raw video overflows a
        district uplink by 10-19x; metadata must not.
        """
        share = metadata_mbps(sample_event().size_bytes()) / DISTRICTS
        assert share < DISTRICT_UPLINK_MBPS * 0.10, (
            f"{share:.1f} Mbps is over 10% of a shared {DISTRICT_UPLINK_MBPS} "
            f"Mbps uplink carrying e-governance traffic"
        )

    def test_raw_video_overflows_district_uplink(self):
        """Section 1.2 — the premise of the whole architecture."""
        per_district = (CAMERAS / DISTRICTS) * RAW_BITRATE_MBPS
        assert per_district / DISTRICT_UPLINK_MBPS > 10


class TestMilvusCapacity:
    """Section 3.2 — where deduplication was counted twice."""

    MILVUS_QPS_LOW  = 50_000
    MILVUS_QPS_HIGH = 100_000

    def test_true_search_load_is_not_reduced_again(self):
        searches = CAMERAS * EVENTS_PER_SEC
        assert searches == 160_000
        double_counted = int(searches * (1 - DEDUP_RATE))
        assert double_counted == 56_000      # the documented figure
        assert double_counted != searches, (
            "Section 3.2 reached 56,000 QPS by applying the 65% dedup factor "
            "to a number that already included it"
        )

    def test_documented_capacity_is_short(self):
        """
        The shortfall the double-count concealed. This test failing means
        someone changed the sizing — which is the point.
        """
        searches = CAMERAS * EVENTS_PER_SEC
        assert searches > self.MILVUS_QPS_HIGH
        assert searches / self.MILVUS_QPS_HIGH == pytest.approx(1.6, abs=0.05)
        assert searches / self.MILVUS_QPS_LOW == pytest.approx(3.2, abs=0.05)


class TestCapexArithmetic:
    """
    Section 3.3 — three line items where Lakh was converted to Crore wrongly.
    The stated column summed correctly, which is why a check of the total did
    not catch it. Only unit x quantity does.
    """

    # (name, unit cost in Lakh, quantity, total as printed in the document)
    ROWS = [
        ("AGX Orin edge box",    2.1, 3500, 73.5),
        ("Orin NX edge box",     1.4, 3000, 42.0),
        ("District hubs",       18.0,   32,  5.8),
        ("Central CPU servers", 35.0,    4, 14.0),
        ("Central GPU servers", 40.0,    2,  8.0),
        ("Ceph cluster",        25.0,    1, 25.0),
    ]
    NETWORKING_CR = 8.0     # no unit basis given in the document
    SOFTWARE_CR   = 36.0

    @staticmethod
    def crore(unit_lakh: float, qty: int) -> float:
        return unit_lakh * qty / 100.0      # 100 Lakh = 1 Crore

    def test_three_line_items_do_not_follow_from_unit_times_quantity(self):
        wrong = [
            (n, stated, self.crore(u, q))
            for n, u, q, stated in self.ROWS
            if abs(stated - self.crore(u, q)) > 0.05
        ]
        assert {n for n, _, _ in wrong} == {
            "Central CPU servers", "Central GPU servers", "Ceph cluster"
        }, f"unexpected set of mismatches: {wrong}"

    def test_corrected_capex(self):
        hardware = sum(self.crore(u, q) for _, u, q, _ in self.ROWS)
        total = hardware + self.NETWORKING_CR + self.SOFTWARE_CR
        assert total == pytest.approx(167.7, abs=0.5), (
            f"Corrected CAPEX is Rs {total:.1f} Cr, not the documented 212 Cr"
        )

    def test_corrected_total_falls_inside_section_0_range(self):
        """
        Section 0 says "Corrected to Rs 160-200 Cr" while Section 3.3 concludes
        Rs 212 Cr. They contradict, and Section 0 is the one that is right.
        """
        hardware = sum(self.crore(u, q) for _, u, q, _ in self.ROWS)
        total = hardware + self.NETWORKING_CR + self.SOFTWARE_CR
        assert 160 <= total <= 200
        assert not (160 <= 212 <= 200)

    def test_bengaluru_comparison_understates_our_own_case(self):
        """
        Section 3.3 computes Bengaluru at "~Rs 6.6 Lakh per camera" correctly,
        then compares against "Rs 66,000" — off by 10x, against itself.
        """
        bengaluru = 496.57e7 / 7_500
        assert bengaluru == pytest.approx(662_093, abs=1_000)
        assert bengaluru != pytest.approx(66_000, rel=0.5)

        hardware = sum(self.crore(u, q) for _, u, q, _ in self.ROWS)
        ours = (hardware + self.NETWORKING_CR + self.SOFTWARE_CR) * 1e7 / CAMERAS
        assert ours == pytest.approx(20_964, abs=500)
        assert bengaluru / ours > 20, "the real advantage is ~25x, not 2.5x"
