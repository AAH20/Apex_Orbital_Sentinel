"""Conjunction detection: closest approach, collision probability, CDM generation."""
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple

from src.space.tle import TLE
from src.space.propagator import SGP4Propagator


@dataclass
class ConjunctionEvent:
    """Result of a closest-approach computation."""
    # New API (km-based, for ConjunctionDetector)
    time: datetime = None
    distance_km: float = 0.0
    sat1_name: str = ""
    sat2_name: str = ""
    relative_velocity_km_s: float = 0.0

    # Legacy API (metres-based, for closest_approach)
    miss_distance: float = 0.0  # metres
    time_of_closest_approach: float = 0.0  # seconds from epoch
    relative_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    relative_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def closest_approach(s1: Dict, s2: Dict) -> ConjunctionEvent:
    """Compute closest approach between two state vectors.

    Args:
        s1: {"position": (x,y,z), "velocity": (vx,vy,vz)} in metres and m/s
        s2: same format

    Returns:
        ConjunctionEvent with miss distance and TCA.
    """
    p1 = s1["position"]
    p2 = s2["position"]
    v1 = s1["velocity"]
    v2 = s2["velocity"]

    # Relative position and velocity
    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    rz = p2[2] - p1[2]
    vx = v2[0] - v1[0]
    vy = v2[1] - v1[1]
    vz = v2[2] - v1[2]

    # TCA = -dot(r, v) / dot(v, v)
    vv = vx * vx + vy * vy + vz * vz
    if vv == 0.0:
        tca = 0.0
    else:
        tca = -(rx * vx + ry * vy + rz * vz) / vv

    # Position at TCA
    dx = rx + vx * tca
    dy = ry + vy * tca
    dz = rz + vz * tca
    miss = math.sqrt(dx * dx + dy * dy + dz * dz)

    return ConjunctionEvent(
        miss_distance=miss,
        time_of_closest_approach=tca,
        relative_position=(dx, dy, dz),
        relative_velocity=(vx, vy, vz),
    )


def collision_probability(miss_distance: float, radius1: float, radius2: float) -> float:
    """Compute collision probability using a simplified 2-D Gaussian model.

    Args:
        miss_distance: distance of closest approach (metres)
        radius1: hard-body radius of object 1 (metres)
        radius2: hard-body radius of object 2 (metres)

    Returns:
        Probability in [0, 1].
    """
    combined_radius = radius1 + radius2
    if combined_radius <= 0.0:
        return 0.0
    if miss_distance <= 0.0:
        return 1.0

    # Simplified 2-D Gaussian: P_c = exp(-d^2 / (2 * R^2))
    exponent = -(miss_distance * miss_distance) / (2.0 * combined_radius * combined_radius)
    return math.exp(exponent)


def generate_cdm(
    object1_name: str,
    object2_name: str,
    s1: Dict,
    s2: Dict,
    ca: ConjunctionEvent,
    radius1: float = 10.0,
    radius2: float = 10.0,
) -> Dict:
    """Generate a Conjunction Data Message (CDM) in CCSDS-style format.

    Args:
        object1_name: identifier for object 1
        object2_name: identifier for object 2
        s1: state vector for object 1
        s2: state vector for object 2
        ca: ConjunctionEvent from closest_approach()
        radius1: hard-body radius of object 1 (metres)
        radius2: hard-body radius of object 2 (metres)

    Returns:
        Dictionary representing the CDM.
    """
    p_collision = collision_probability(ca.miss_distance, radius1, radius2)

    message_id = f"CDM-{object1_name}-{object2_name}-{uuid.uuid4().hex[:8].upper()}"
    creation_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    return {
        "header": {
            "message_id": message_id,
            "creation_date": creation_date,
            "originator": "APEX-OS",
            "message_for": f"{object1_name}/{object2_name}",
        },
        "relative_metadata": {
            "tca": ca.time_of_closest_approach,
            "miss_distance": ca.miss_distance,
            "collision_probability": p_collision,
            "combined_radius": radius1 + radius2,
        },
        "object1": {
            "object_name": object1_name,
            "position": list(s1["position"]),
            "velocity": list(s1["velocity"]),
            "radius": radius1,
        },
        "object2": {
            "object_name": object2_name,
            "position": list(s2["position"]),
            "velocity": list(s2["velocity"]),
            "radius": radius2,
        },
    }


class ConjunctionDetector:
    """Detects conjunctions between orbital objects."""

    def __init__(self, threshold_km: float = 10.0, time_step_seconds: float = 60.0):
        self.threshold_km = threshold_km
        self.time_step_seconds = time_step_seconds
        self.propagator = SGP4Propagator()

    def detect(self, primary: TLE, secondary: TLE, start: datetime, duration_minutes: float = 90) -> list:
        """Detect conjunctions between two objects over a time period."""
        events = []
        steps = int(duration_minutes * 60 / self.time_step_seconds)

        for i in range(steps + 1):
            t = start + timedelta(seconds=i * self.time_step_seconds)
            sv1 = self.propagator.propagate(primary, t)
            sv2 = self.propagator.propagate(secondary, t)

            dist = self._distance(sv1.position, sv2.position)

            if dist <= self.threshold_km:
                # Calculate relative velocity
                rel_vel = math.sqrt(
                    (sv1.velocity[0] - sv2.velocity[0]) ** 2 +
                    (sv1.velocity[1] - sv2.velocity[1]) ** 2 +
                    (sv1.velocity[2] - sv2.velocity[2]) ** 2
                )
                events.append(ConjunctionEvent(
                    time=t,
                    distance_km=dist,
                    sat1_name=primary.name,
                    sat2_name=secondary.name,
                    relative_velocity_km_s=rel_vel,
                ))

        return events

    def _distance(self, pos1: tuple, pos2: tuple) -> float:
        """Euclidean distance between two positions."""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )
