"""Orbital debris tracking: catalog, trajectory prediction, and risk assessment."""
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
EARTH_MU_KM3_S2 = 398600.4418  # km³/s²
EARTH_RADIUS_KM = 6371.0


# ─── Data Model ───────────────────────────────────────────────────────────────
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DebrisObject:
    norad_id: str
    name: str
    size_m: float
    mass_kg: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_periapsis_deg: float
    mean_anomaly_deg: float
    epoch: datetime
    object_type: str = "debris"


# ─── Catalog ──────────────────────────────────────────────────────────────────
class DebrisCatalog:
    def __init__(self):
        self._objects: Dict[str, DebrisObject] = {}

    def add(self, obj: DebrisObject) -> None:
        if obj.norad_id in self._objects:
            raise ValueError(f"Object {obj.norad_id} already exists")
        self._objects[obj.norad_id] = obj

    def get(self, norad_id: str) -> DebrisObject:
        if norad_id not in self._objects:
            raise KeyError(f"Object {norad_id} not found")
        return self._objects[norad_id]

    def remove(self, norad_id: str) -> None:
        if norad_id not in self._objects:
            raise KeyError(f"Object {norad_id} not found")
        del self._objects[norad_id]

    def list_all(self) -> List[DebrisObject]:
        return list(self._objects.values())

    def filter_by_type(self, object_type: str) -> List[DebrisObject]:
        return [o for o in self._objects.values() if o.object_type == object_type]

    def filter_by_size(self, min_m: float, max_m: float) -> List[DebrisObject]:
        return [o for o in self._objects.values() if min_m <= o.size_m <= max_m]

    def count(self) -> int:
        return len(self._objects)


# ─── Trajectory Prediction ────────────────────────────────────────────────────
class TrajectoryPredictor:
    def __init__(self, catalog: DebrisCatalog):
        self.catalog = catalog

    def _solve_kepler(self, M: float, e: float, tol: float = 1e-10) -> float:
        """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
        E = M
        for _ in range(100):
            dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < tol:
                break
        return E

    def _mean_motion_rad_s(self, a_km: float) -> float:
        """Mean motion n = sqrt(mu / a^3) in rad/s."""
        return math.sqrt(EARTH_MU_KM3_S2 / (a_km ** 3))

    def _position_eci(self, obj: DebrisObject, t: datetime) -> Tuple[float, float, float]:
        """Compute ECI position (km) at time t using Keplerian propagation."""
        dt_seconds = (t - obj.epoch).total_seconds()
        n = self._mean_motion_rad_s(obj.semi_major_axis_km)
        M0 = math.radians(obj.mean_anomaly_deg)
        M = M0 + n * dt_seconds
        M = M % (2.0 * math.pi)

        E = self._solve_kepler(M, obj.eccentricity)

        # Position in orbital plane
        a = obj.semi_major_axis_km
        e = obj.eccentricity
        x_orb = a * (math.cos(E) - e)
        y_orb = a * math.sqrt(1.0 - e * e) * math.sin(E)

        # Rotation angles
        raan = math.radians(obj.raan_deg)
        arg_p = math.radians(obj.arg_periapsis_deg)
        inc = math.radians(obj.inclination_deg)

        # Rotate from orbital plane to ECI
        cos_raan, sin_raan = math.cos(raan), math.sin(raan)
        cos_arg, sin_arg = math.cos(arg_p), math.sin(arg_p)
        cos_inc, sin_inc = math.cos(inc), math.sin(inc)

        # R = R3(-raan) * R1(-inc) * R3(-arg_p)
        x = (cos_raan * cos_arg - sin_raan * sin_arg * cos_inc) * x_orb + \
            (-cos_raan * sin_arg - sin_raan * cos_arg * cos_inc) * y_orb
        y = (sin_raan * cos_arg + cos_raan * sin_arg * cos_inc) * x_orb + \
            (-sin_raan * sin_arg + cos_raan * cos_arg * cos_inc) * y_orb
        z = (sin_arg * sin_inc) * x_orb + (cos_arg * sin_inc) * y_orb

        return (x, y, z)

    def predict_position(self, norad_id: str, t: datetime) -> Tuple[float, float, float]:
        obj = self.catalog.get(norad_id)
        return self._position_eci(obj, t)

    def predict_trajectory(
        self,
        norad_id: str,
        start: datetime,
        end: datetime,
        step_seconds: float = 600.0,
    ) -> List[Tuple[float, float, float]]:
        obj = self.catalog.get(norad_id)
        positions = []
        current = start
        while current <= end:
            positions.append(self._position_eci(obj, current))
            current += timedelta(seconds=step_seconds)
        # Ensure endpoint is included
        if positions[-1] != self._position_eci(obj, end):
            positions.append(self._position_eci(obj, end))
        return positions


# ─── Risk Assessment ──────────────────────────────────────────────────────────
class RiskAssessor:
    def __init__(self, catalog: DebrisCatalog):
        self.catalog = catalog
        self.predictor = TrajectoryPredictor(catalog)

    def compute_closest_approach(
        self, id1: str, id2: str, t: datetime
    ) -> float:
        pos1 = self.predictor.predict_position(id1, t)
        pos2 = self.predictor.predict_position(id2, t)
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2
            + (pos1[1] - pos2[1]) ** 2
            + (pos1[2] - pos2[2]) ** 2
        )

    def assess_risk(self, id1: str, id2: str, t: datetime) -> RiskLevel:
        dist = self.compute_closest_approach(id1, id2, t)
        if dist < 1.0:
            return RiskLevel.CRITICAL
        elif dist < 10.0:
            return RiskLevel.HIGH
        elif dist < 100.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def find_conjunctions(
        self, t: datetime, threshold_km: float = 10.0
    ) -> List[Tuple[str, str]]:
        objects = self.catalog.list_all()
        conjunctions = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                dist = self.compute_closest_approach(
                    objects[i].norad_id, objects[j].norad_id, t
                )
                if dist <= threshold_km:
                    conjunctions.append((objects[i].norad_id, objects[j].norad_id))
        return conjunctions

    def compute_collision_probability(
        self, id1: str, id2: str, t: datetime
    ) -> float:
        dist = self.compute_closest_approach(id1, id2, t)
        obj1 = self.catalog.get(id1)
        obj2 = self.catalog.get(id2)
        combined_radius_km = (obj1.size_m + obj2.size_m) / 2000.0
        if dist < combined_radius_km:
            return 1.0
        sigma = max(combined_radius_km, 0.001)
        prob = math.exp(-0.5 * (dist / sigma) ** 2)
        return min(max(prob, 0.0), 1.0)
