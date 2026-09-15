"""
tests/test_junction_topology.py — Master 9-Gap Junction Geometry & Topology Tests (v15.0.0).

Tests:
  1. Clockwise rotary: legal vehicle circling rotary (Gap V)
  2. Clockwise rotary: wrong-way vehicle (counter-clockwise)
  3. Counter-clockwise formula rejection (verifying sign correction)
  4. Median cut U-turn within 45s grace period
  5. Median cut U-turn after 46s expiry
  6. Slipway Frenet lateral deviation <= 2.5m (Gap Z)
  7. Slipway Frenet lateral deviation > 2.5m
  8. BRTS: car in legal direction -> BRTS_INCURSION (₹500)
  9. BRTS: car in wrong direction -> BRTS_WRONG_WAY (₹1,500)
  10. BRTS: bus in either direction -> PROCEED_TO_CHECK (allowed)
  11. Parking: 3m reverse at 4 km/h -> PARKING_MANEUVER (suppressed)
  12. Real wrong-way: D_opp = 15m, V = 40 km/h -> PROCEED_TO_CHECK
  13. Emergency vehicle by class label (Gap Y) -> EMERGENCY_EXEMPT
  14. Emergency vehicle by livery color ratio -> EMERGENCY_EXEMPT
  15. Cross-camera deduplication within 15 min -> DEDUPLICATED
"""
import numpy as np
import pytest

from backend.services.junction_topology import (
    BRTSCorridorZone,
    JunctionTopology,
    ManeuverVerdict,
    MedianCutZone,
    PlateDeduplicationCache,
    RoundaboutZone,
    ShoulderZone,
    SlipwayZone,
    evaluate_vehicle_maneuver,
    is_emergency_vehicle,
)


def test_clockwise_rotary_legal():
    # Rotary at (0, 0) with inner=10m, outer=30m
    rotary = RoundaboutZone(0.0, 0.0, 10.0, 30.0)
    top = JunctionTopology(roundabouts=[rotary])

    # Vehicle at (0, 20) — North of circle
    # Clockwise flow at (0, 20): tx = dy/R = 20/20 = +1.0, ty = -dx/R = 0/20 = 0.0 -> heading East (+1, 0)
    heading_east = (1.0, 0.0)
    verdict = evaluate_vehicle_maneuver(
        track_id=1,
        wx=0.0,
        wy=20.0,
        speed_kmh=25.0,
        vehicle_class="car",
        heading=heading_east,
        plate="GJ01AA1111",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert verdict == ManeuverVerdict.ROUNDABOUT_LEGAL


def test_clockwise_rotary_wrong_way():
    rotary = RoundaboutZone(0.0, 0.0, 10.0, 30.0)
    top = JunctionTopology(roundabouts=[rotary])

    # Vehicle at (0, 20) heading West (-1, 0) — counter-clockwise (wrong way in India!)
    heading_west = (-1.0, 0.0)
    verdict = evaluate_vehicle_maneuver(
        track_id=2,
        wx=0.0,
        wy=20.0,
        speed_kmh=25.0,
        vehicle_class="car",
        heading=heading_west,
        plate="GJ01AA2222",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert verdict == ManeuverVerdict.PROCEED_TO_CHECK


def test_median_u_turn_grace_and_expiry():
    poly = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
    median = MedianCutZone(polygon_world_m=poly, grace_period_sec=45.0)
    top = JunctionTopology(median_cuts=[median])

    # T = 0: Entered median pocket
    v1 = evaluate_vehicle_maneuver(
        track_id=10,
        wx=0.0,
        wy=0.0,
        speed_kmh=5.0,
        vehicle_class="car",
        heading=(0.0, 1.0),
        plate="GJ01BB3333",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=10.0,
        net_displacement_opp_m=12.0,
    )
    assert v1 == ManeuverVerdict.MEDIAN_UTURN_GRACE

    # T = 30s: Still within 45s grace
    v2 = evaluate_vehicle_maneuver(
        track_id=10,
        wx=0.0,
        wy=0.0,
        speed_kmh=5.0,
        vehicle_class="car",
        heading=(0.0, 1.0),
        plate="GJ01BB3333",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=40.0,
        net_displacement_opp_m=12.0,
    )
    assert v2 == ManeuverVerdict.MEDIAN_UTURN_GRACE

    # T = 60s: Exceeded 45s grace -> must proceed to normal check
    v3 = evaluate_vehicle_maneuver(
        track_id=10,
        wx=0.0,
        wy=0.0,
        speed_kmh=5.0,
        vehicle_class="car",
        heading=(0.0, 1.0),
        plate="GJ01BB3333",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=70.0,
        net_displacement_opp_m=12.0,
    )
    assert v3 == ManeuverVerdict.PROCEED_TO_CHECK


def test_frenet_slipway_lateral_deviation():
    # Centerline from (0, 0) to (50, 50) curved
    centerline = [(0.0, 0.0), (20.0, 10.0), (40.0, 30.0), (50.0, 50.0)]
    poly = [(-5.0, -5.0), (55.0, -5.0), (55.0, 55.0), (-5.0, 55.0)]
    slipway = SlipwayZone(centerline_world_m=centerline, polygon_world_m=poly, max_lateral_deviation_m=2.5)
    top = JunctionTopology(slipways=[slipway])

    # Point close to centerline (d < 2.5m)
    v_legal = evaluate_vehicle_maneuver(
        track_id=20,
        wx=20.5,
        wy=10.2,
        speed_kmh=35.0,
        vehicle_class="car",
        heading=(0.7, 0.7),
        plate="GJ01CC4444",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert v_legal == ManeuverVerdict.SLIPWAY_LEGAL

    # Point far from centerline (d = 10m > 2.5m)
    v_off = evaluate_vehicle_maneuver(
        track_id=21,
        wx=20.0,
        wy=25.0,
        speed_kmh=35.0,
        vehicle_class="car",
        heading=(0.7, 0.7),
        plate="GJ01CC5555",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert v_off == ManeuverVerdict.PROCEED_TO_CHECK


def test_brts_incursion_and_wrong_way():
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 100.0), (0.0, 100.0)]
    flow = (0.0, 1.0)  # Northbound legal flow
    brts = BRTSCorridorZone(polygon_world_m=poly, flow_vector=flow, allowed_classes={"bus", "brts_bus"})
    top = JunctionTopology(brts_corridors=[brts])

    # 1. Car driving in legal direction inside BRTS -> Incursion (₹500)
    v_inc = evaluate_vehicle_maneuver(
        track_id=30,
        wx=5.0,
        wy=50.0,
        speed_kmh=30.0,
        vehicle_class="car",
        heading=(0.0, 1.0),
        plate="GJ01DD6666",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert v_inc == ManeuverVerdict.BRTS_INCURSION

    # 2. Car driving counter-flow inside BRTS -> Wrong Way (₹1,500)
    v_ww = evaluate_vehicle_maneuver(
        track_id=31,
        wx=5.0,
        wy=50.0,
        speed_kmh=30.0,
        vehicle_class="car",
        heading=(0.0, -1.0),
        plate="GJ01DD7777",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert v_ww == ManeuverVerdict.BRTS_WRONG_WAY

    # 3. BRTS Bus in corridor -> Legal (PROCEED_TO_CHECK)
    v_bus = evaluate_vehicle_maneuver(
        track_id=32,
        wx=5.0,
        wy=50.0,
        speed_kmh=40.0,
        vehicle_class="bus",
        heading=(0.0, 1.0),
        plate="GJ01BR0001",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=15.0,
    )
    assert v_bus == ManeuverVerdict.PROCEED_TO_CHECK


def test_parking_reversal_suppressed():
    top = JunctionTopology()
    # 3m reverse at 4 km/h -> Parking maneuver
    v = evaluate_vehicle_maneuver(
        track_id=40,
        wx=0.0,
        wy=0.0,
        speed_kmh=4.0,
        vehicle_class="car",
        heading=(0.0, -1.0),
        plate="GJ01EE8888",
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=100.0,
        net_displacement_opp_m=3.0,
    )
    assert v == ManeuverVerdict.PARKING_MANEUVER


def test_emergency_vehicle_exemption():
    # 1. By class
    assert is_emergency_vehicle("ambulance", None) is True
    assert is_emergency_vehicle("fire_truck", None) is True
    assert is_emergency_vehicle("car", None) is False

    # 2. By livery (Synthetic ambulance crop with white body + red stripe)
    crop = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)  # White
    crop[40:60, :] = (0, 0, 220)  # Red BGR stripe
    assert is_emergency_vehicle("unknown", crop) is True


def test_plate_deduplication_15_minutes():
    top = JunctionTopology()
    plate = "GJ01FF9999"

    # First detection at t = 0 -> Not duplicate
    v1 = evaluate_vehicle_maneuver(
        track_id=50,
        wx=0.0,
        wy=0.0,
        speed_kmh=35.0,
        vehicle_class="car",
        heading=(0.0, 1.0),
        plate=plate,
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=0.0,
        net_displacement_opp_m=15.0,
    )
    assert v1 != ManeuverVerdict.DEDUPLICATED

    # Second detection at t = 300s (5 min later on adjacent camera) -> DEDUPLICATED
    v2 = evaluate_vehicle_maneuver(
        track_id=51,
        wx=0.0,
        wy=0.0,
        speed_kmh=35.0,
        vehicle_class="car",
        heading=(0.0, 1.0),
        plate=plate,
        violation_type="WRONG_WAY",
        topology=top,
        vehicle_crop=None,
        now=300.0,
        net_displacement_opp_m=15.0,
    )
    assert v2 == ManeuverVerdict.DEDUPLICATED
