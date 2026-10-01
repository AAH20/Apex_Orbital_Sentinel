#!/usr/bin/env python3
"""
demo_collision_avoidance.py — Collision Avoidance Demo for Apex_Orbital_Sentinel

Demonstrates collision avoidance maneuver planning by computing the required
delta-V for a phasing maneuver that increases the miss distance between two
objects. Uses a simplified impulsive maneuver model.

Usage:
    python demo_collision_avoidance.py

Requires:
    pip install sgp4
"""

import math
from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec, jday

# ─── Sample TLEs ─────────────────────────────────────────────────────────────

# Primary satellite (to be maneuvered)
PRIMARY_TLE = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008",
    "2 25544  51.6400 208.9163 0006703  35.9063  62.1360 15.49815522  1000",
)

# Threat object (debris)
THREAT_TLE = (
    "1 39999U 98067B   24001.50000000  .00016717  00000-0  10270-3 0  9008",
    "2 39999  51.6410 208.9170 0006700  35.9060  62.1365 15.49815522  1000",
)

# Constants
MU_EARTH = 398600.4418  # km³/s² (Earth gravitational parameter)
SAFETY_BUFFER_KM = 10.0  # Desired minimum miss distance after maneuver


def distance_km(r1: tuple, r2: tuple) -> float:
    """Euclidean distance between two position vectors in km."""
    return math.sqrt(
        (r1[0] - r2[0])**2 +
        (r1[1] - r2[1])**2 +
        (r1[2] - r2[2])**2
    )


def orbital_velocity(semi_major_axis_km: float) -> float:
    """Circular orbital velocity in km/s."""
    return math.sqrt(MU_EARTH / semi_major_axis_km)


def compute_miss_distance(sat_primary: Satrec, sat_threat: Satrec,
                          start: datetime, duration_min: int = 60,
                          step_sec: int = 30) -> tuple:
    """Compute minimum miss distance between two objects."""
    min_dist = float("inf")
    min_time = None

    steps = int(duration_min * 60 / step_sec)
    for i in range(steps + 1):
        t = start + timedelta(seconds=i * step_sec)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                       t.second + t.microsecond * 1e-6)

        e_p, r_p, v_p = sat_primary.sgp4(jd, fr)
        e_t, r_t, v_t = sat_threat.sgp4(jd, fr)

        if e_p != 0 or e_t != 0:
            continue

        d = distance_km(r_p, r_t)
        if d < min_dist:
            min_dist = d
            min_time = t

    return min_dist, min_time


def plan_avoidance_maneuver(current_miss_km: float, target_miss_km: float,
                            semi_major_axis_km: float) -> dict:
    """
    Plan a collision avoidance maneuver using a simplified phasing model.

    Computes the delta-V required to shift the orbital phase such that
    the miss distance increases to the target value.
    """
    if current_miss_km >= target_miss_km:
        return {
            "delta_v_ms": 0.0,
            "delta_v_kms": 0.0,
            "maneuver_type": "None (already safe)",
            "burn_direction": "N/A",
            "time_to_burn": "N/A",
        }

    # Required phase shift in radians (simplified linear model)
    # For small angles: phase_shift ≈ miss_distance / orbital_radius
    phase_shift_rad = (target_miss_km - current_miss_km) / semi_major_axis_km

    # Orbital period in seconds
    orbital_period_s = 2 * math.pi * math.sqrt(semi_major_axis_km**3 / MU_EARTH)

    # Time shift needed (seconds) — phase shift maps to time shift
    time_shift_s = (phase_shift_rad / (2 * math.pi)) * orbital_period_s

    # Delta-V for a phasing maneuver (simplified)
    # For a Hohmann-like phasing: delta-V ≈ (2/3) * v * phase_shift
    v = orbital_velocity(semi_major_axis_km)
    delta_v_ms = (2.0 / 3.0) * v * phase_shift_rad * 1000  # m/s

    # Determine burn direction
    if time_shift_s > 0:
        burn_direction = "Retrograde (slow down to let threat pass)"
    else:
        burn_direction = "Prograde (speed up to pass threat)"

    return {
        "delta_v_ms": delta_v_ms,
        "delta_v_kms": delta_v_ms / 1000,
        "maneuver_type": "Phasing",
        "burn_direction": burn_direction,
        "time_to_burn": f"{abs(time_shift_s):.0f} s before closest approach",
    }


def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║    Apex_Orbital_Sentinel — Collision Avoidance Maneuver Demo       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    sat_primary = Satrec.twoline2rv(PRIMARY_TLE[0], PRIMARY_TLE[1])
    sat_threat = Satrec.twoline2rv(THREAT_TLE[0], THREAT_TLE[1])

    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Step 1: Compute current miss distance
    print("\n[1] Computing current miss distance...")
    current_miss, miss_time = compute_miss_distance(
        sat_primary, sat_threat, start_time, duration_min=60, step_sec=30
    )
    print(f"    Current miss distance: {current_miss:.3f} km")
    print(f"    Time of closest approach: {miss_time.strftime('%H:%M:%S UTC')}")

    # Step 2: Estimate semi-major axis from SGP4
    jd, fr = jday(start_time.year, start_time.month, start_time.day,
                   start_time.hour, start_time.minute, start_time.second)
    e, r, v = sat_primary.sgp4(jd, fr)
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # Vis-viva: a = 1 / (2/r - v²/μ)
    semi_major_axis = 1.0 / (2.0 / r_mag - v_mag**2 / MU_EARTH)
    print(f"    Semi-major axis: {semi_major_axis:.1f} km")
    print(f"    Orbital velocity: {v_mag:.4f} km/s")

    # Step 3: Plan avoidance maneuver
    print(f"\n[2] Planning avoidance maneuver...")
    print(f"    Target miss distance: {SAFETY_BUFFER_KM} km")

    maneuver = plan_avoidance_maneuver(
        current_miss, SAFETY_BUFFER_KM, semi_major_axis
    )

    print(f"\n    ┌──────────────────────────────────────────────┐")
    print(f"    │  MANEUVER PLAN                              │")
    print(f"    ├──────────────────────────────────────────────┤")
    print(f"    │  Type          : {maneuver['maneuver_type']:<28}│")
    print(f"    │  Delta-V       : {maneuver['delta_v_ms']:.2f} m/s{'':<19}│")
    print(f"    │  Direction     : {maneuver['burn_direction']:<28}│")
    print(f"    │  Time to burn  : {maneuver['time_to_burn']:<28}│")
    print(f"    └──────────────────────────────────────────────┘")

    # Step 4: Summary
    print(f"\n[3] Summary:")
    if maneuver["delta_v_ms"] > 0:
        print(f"    ⚠️  Collision avoidance maneuver REQUIRED")
        print(f"    → Execute {maneuver['delta_v_ms']:.2f} m/s {maneuver['burn_direction'].split('(')[0].strip()} burn")
        print(f"    → Expected miss distance after maneuver: {SAFETY_BUFFER_KM} km")
    else:
        print(f"    ✓  No maneuver required — current miss distance is safe")

    print(f"\n{'='*70}")
    print("  Collision avoidance demo complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
