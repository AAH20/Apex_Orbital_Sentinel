#!/usr/bin/env python3
"""
demo_conjunction.py — Conjunction Detection Demo for Apex_Orbital_Sentinel

Demonstrates conjunction detection between two orbiting objects by
propagating both orbits and computing the minimum distance between them
over a time window. Alerts when the distance falls below a configurable
threshold.

Usage:
    python demo_conjunction.py

Requires:
    pip install sgp4
"""

import math
from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec, jday

# ─── Sample TLEs ─────────────────────────────────────────────────────────────

# Object A: ISS-like
OBJ_A_TLE = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008",
    "2 25544  51.6400 208.9163 0006703  35.9063  62.1360 15.49815522  1000",
)

# Object B: Debris-like (close approach scenario)
OBJ_B_TLE = (
    "1 39999U 98067B   24001.50000000  .00016717  00000-0  10270-3 0  9008",
    "2 39999  51.6410 208.9170 0006700  35.9060  62.1365 15.49815522  1000",
)

# Conjunction alert threshold (km)
ALERT_THRESHOLD_KM = 5.0


def distance_km(r1: tuple, r2: tuple) -> float:
    """Euclidean distance between two position vectors in km."""
    return math.sqrt(
        (r1[0] - r2[0])**2 +
        (r1[1] - r2[1])**2 +
        (r1[2] - r2[2])**2
    )


def detect_conjunction(name_a: str, line1_a: str, line2_a: str,
                       name_b: str, line1_b: str, line2_b: str,
                       start: datetime, duration_min: int = 60,
                       step_sec: int = 30) -> dict:
    """Detect closest approach between two objects over a time window."""
    sat_a = Satrec.twoline2rv(line1_a, line2_a)
    sat_b = Satrec.twoline2rv(line1_b, line2_b)

    min_dist = float("inf")
    min_time = None
    min_r_a = None
    min_r_b = None

    print(f"\n{'='*70}")
    print(f"  Conjunction Detection: {name_a} vs {name_b}")
    print(f"  Alert threshold: {ALERT_THRESHOLD_KM} km")
    print(f"{'='*70}")
    print(f"{'Time (UTC)':<22} {'Distance (km)':>14} {'Status':>12}")
    print("-" * 70)

    steps = int(duration_min * 60 / step_sec)
    for i in range(steps + 1):
        t = start + timedelta(seconds=i * step_sec)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                       t.second + t.microsecond * 1e-6)

        e_a, r_a, v_a = sat_a.sgp4(jd, fr)
        e_b, r_b, v_b = sat_b.sgp4(jd, fr)

        if e_a != 0 or e_b != 0:
            continue

        d = distance_km(r_a, r_b)
        status = "⚠️  ALERT" if d < ALERT_THRESHOLD_KM else "OK"

        if i % 10 == 0 or d < ALERT_THRESHOLD_KM:
            print(f"  t={i*step_sec:>5}s       {d:>14.3f} {status:>12}")

        if d < min_dist:
            min_dist = d
            min_time = t
            min_r_a = r_a
            min_r_b = r_b

    print(f"\n  Minimum distance: {min_dist:.3f} km at {min_time.strftime('%H:%M:%S UTC')}")
    if min_dist < ALERT_THRESHOLD_KM:
        print(f"  ⚠️  CONJUNCTION ALERT: Objects within {ALERT_THRESHOLD_KM} km!")
    else:
        print(f"  ✓  No conjunction detected (min distance > {ALERT_THRESHOLD_KM} km)")

    return {
        "min_distance_km": min_dist,
        "min_time": min_time,
        "alert": min_dist < ALERT_THRESHOLD_KM,
    }


def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      Apex_Orbital_Sentinel — Conjunction Detection Demo            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    result = detect_conjunction(
        "ISS (ZARYA)", OBJ_A_TLE[0], OBJ_A_TLE[1],
        "DEBRIS-1", OBJ_B_TLE[0], OBJ_B_TLE[1],
        start_time, duration_min=60, step_sec=30,
    )

    print(f"\n{'='*70}")
    print("  Conjunction detection complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
