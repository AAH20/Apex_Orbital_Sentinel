"""SGP4 orbital propagator."""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.space.tle import TLE


class PropagationError(Exception):
    """Raised when propagation fails."""
    pass


@dataclass
class StateVector:
    """Orbital state vector at a given time."""
    position: tuple  # (x, y, z) in km
    velocity: tuple  # (vx, vy, vz) in km/s
    time: datetime


class SGP4Propagator:
    """Simplified SGP4 propagator for near-Earth orbits."""

    # WGS-84 constants
    MU = 398600.4418  # km^3/s^2
    EARTH_RADIUS = 6378.137  # km
    J2 = 1.08262668e-3

    def propagate(self, tle: TLE, when: datetime) -> StateVector:
        """Propagate TLE to given time."""
        try:
            dt_seconds = (when - tle.epoch).total_seconds()
        except Exception as e:
            raise PropagationError(f"Invalid time: {e}")

        # Convert orbital elements to radians
        inc = math.radians(tle.inclination)
        raan = math.radians(tle.raan)
        argp = math.radians(tle.arg_perigee)
        ma = math.radians(tle.mean_anomaly)
        ecc = tle.eccentricity
        n = tle.mean_motion * 2 * math.pi / 86400.0  # rad/s

        # Semi-major axis from mean motion
        a = (self.MU / (n ** 2)) ** (1.0 / 3.0)

        # J2 perturbations on RAAN and argument of perigee
        p = a * (1 - ecc ** 2)
        raan_dot = -1.5 * n * self.J2 * (self.EARTH_RADIUS / p) ** 2 * math.cos(inc)
        argp_dot = 0.75 * n * self.J2 * (self.EARTH_RADIUS / p) ** 2 * (5 * math.cos(inc) ** 2 - 1)

        # Propagate elements
        raan_t = raan + raan_dot * dt_seconds
        argp_t = argp + argp_dot * dt_seconds
        ma_t = ma + n * dt_seconds

        # Solve Kepler's equation
        E = self._solve_kepler(ma_t, ecc)

        # True anomaly
        nu = 2 * math.atan2(
            math.sqrt(1 + ecc) * math.sin(E / 2),
            math.sqrt(1 - ecc) * math.cos(E / 2)
        )

        # Radius
        r = a * (1 - ecc * math.cos(E))

        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        # Velocity in orbital plane
        h = math.sqrt(self.MU * a * (1 - ecc ** 2))
        vx_orb = -self.MU / h * math.sin(E)
        vy_orb = self.MU / h * math.sqrt(1 - ecc ** 2) * math.cos(E)

        # Rotate to ECI
        cos_raan = math.cos(raan_t)
        sin_raan = math.sin(raan_t)
        cos_i = math.cos(inc)
        sin_i = math.sin(inc)
        cos_argp = math.cos(argp_t)
        sin_argp = math.sin(argp_t)

        # Rotation matrix elements
        R11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
        R12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_i
        R21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
        R22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_i
        R31 = sin_argp * sin_i
        R32 = cos_argp * sin_i

        x = R11 * x_orb + R12 * y_orb
        y = R21 * x_orb + R22 * y_orb
        z = R31 * x_orb + R32 * y_orb

        vx = R11 * vx_orb + R12 * vy_orb
        vy = R21 * vx_orb + R22 * vy_orb
        vz = R31 * vx_orb + R32 * vy_orb

        return StateVector(
            position=(x, y, z),
            velocity=(vx, vy, vz),
            time=when,
        )

    def propagate_many(self, tle: TLE, times: list) -> list:
        """Propagate to multiple times."""
        return [self.propagate(tle, t) for t in times]

    def _solve_kepler(self, M: float, e: float, tol: float = 1e-10) -> float:
        """Solve Kepler's equation M = E - e*sin(E) using Newton-Raphson."""
        E = M if e < 0.8 else math.pi
        for _ in range(50):
            dE = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
            E -= dE
            if abs(dE) < tol:
                break
        return E
