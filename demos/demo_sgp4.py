#!/usr/bin/env python3
"""
demo_sgp4.py — SGP4 Propagation Demo for Apex_Orbital_Sentinel

Demonstrates Simplified General Perturbations 4 (SGP4) orbit propagation
using sample Two-Line Element sets (TLEs). Computes satellite position and
velocity in TEME (True Equator Mean Equinox) frame at multiple epochs.

Usage:
    python demo_sgp4.py

Requires:
    pip install sgp4
"""

import math
from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec, jday

# ─── Sample TLEs ─────────────────────────────────────────────────────────────

# ISS (ZARYA) — NORAD ID 25544
ISS_TLE = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9008",
    "2 25544  51.6400 208.9163 0006703  35.9063  62.1360 15.49815522  1000",
)

# NOAA 18 — NORAD ID 28654
NOAA18_TLE = (
    "1 28654U 05018A   24001.50000000  .00000023  00000-0  10270-3 0  9000",
    "2 28654  98.7040  55.0000 0014000  90.0000 270.0000 14.12500000  1000",
)


def propagate_tle(name: str, line1: str, line2: str, start: datetime,
                  minutes: int = 10, step: int = 1) -> None:
    """Propagate a TLE and print position/velocity at each time step."""
    sat = Satrec.twoline2rv(line1, line2)

    print(f"\n{'='*70}")
    print(f"  Satellite: {name}")
    print(f"  NORAD ID : {sat.satnum}")
    print(f"  Epoch    : {sat.epochyr:02d}/{sat.epochdays:.8f}")
    print(f"{'='*70}")
    print(f"{'Time (UTC)':<22} {'X (km)':>12} {'Y (km)':>12} {'Z (km)':>12} "
          f"{'|V| (km/s)':>10}")
    print("-" * 70)

    for m in range(0, minutes + 1, step):
        t = start + timedelta(minutes=m)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                       t.second + t.microsecond * 1e-6)
        e, r, v = sat.sgp4(jd, fr)

        if e != 0:
            print(f"  t={m:>3} min  SGP4 error code {e}")
            continue

        r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

        print(f"  t={m:>3} min       "
              f"{r[0]:>12.3f} {r[1]:>12.3f} {r[2]:>12.3f} {v_mag:>10.4f}")

    print(f"\n  Final position magnitude: {r_mag:.3f} km")
    print(f"  Final velocity magnitude: {v_mag:.4f} km/s")


def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        Apex_Orbital_Sentinel — SGP4 Propagation Demo                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    propagate_tle("ISS (ZARYA)", ISS_TLE[0], ISS_TLE[1], start_time,
                  minutes=10, step=2)
    propagate_tle("NOAA 18", NOAA18_TLE[0], NOAA18_TLE[1], start_time,
                  minutes=10, step=2)

    print(f"\n{'='*70}")
    print("  SGP4 propagation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
