"""Collision risk assessment: miss distance, Pc calculation, risk levels."""
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RiskAssessment:
    """Result of a collision risk assessment."""
    probability_of_collision: float
    miss_distance_km: float
    risk_level: str
    risk_score: float
    combined_radius_km: float
    relative_velocity_km_s: float = 0.0


def compute_miss_distance(s1: dict, s2: dict) -> float:
    """Compute miss distance between two state vectors.

    Args:
        s1: {"position": (x,y,z), "velocity": (vx,vy,vz)} in km and km/s
        s2: same format

    Returns:
        Miss distance in km.
    """
    p1 = s1["position"]
    p2 = s2["position"]
    v1 = s1["velocity"]
    v2 = s2["velocity"]

    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    rz = p2[2] - p1[2]
    vx = v2[0] - v1[0]
    vy = v2[1] - v1[1]
    vz = v2[2] - v1[2]

    vv = vx * vx + vy * vy + vz * vz
    if vv == 0.0:
        tca = 0.0
    else:
        tca = -(rx * vx + ry * vy + rz * vz) / vv

    dx = rx + vx * tca
    dy = ry + vy * tca
    dz = rz + vz * tca
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_pc(
    miss_distance_km: float,
    combined_radius_km: float,
    sigma_x: float = None,
    sigma_y: float = None,
) -> float:
    """Compute probability of collision using 2-D Gaussian model.

    Args:
        miss_distance_km: distance of closest approach in km
        combined_radius_km: combined hard-body radius in km
        sigma_x: 1-sigma uncertainty in x (defaults to combined_radius)
        sigma_y: 1-sigma uncertainty in y (defaults to combined_radius)

    Returns:
        Probability in [0, 1].
    """
    if combined_radius_km <= 0.0:
        return 0.0
    if miss_distance_km <= 0.0:
        return 1.0

    if sigma_x is None:
        sigma_x = combined_radius_km
    if sigma_y is None:
        sigma_y = combined_radius_km

    if sigma_x <= 0.0 or sigma_y <= 0.0:
        return 0.0

    # 2-D Gaussian: Pc = exp(-d^2 / (2 * sigma^2))
    # Use geometric mean of sigmas for the effective sigma
    sigma_eff = math.sqrt(sigma_x * sigma_y)
    exponent = -(miss_distance_km * miss_distance_km) / (2.0 * sigma_eff * sigma_eff)
    return math.exp(exponent)


def assess_risk(
    miss_distance_km: float,
    combined_radius_km: float,
    pc: float,
    relative_velocity_km_s: float = 0.0,
) -> RiskAssessment:
    """Assess collision risk level.

    Args:
        miss_distance_km: miss distance in km
        combined_radius_km: combined hard-body radius in km
        pc: probability of collision
        relative_velocity_km_s: relative velocity in km/s

    Returns:
        RiskAssessment with risk level and score.
    """
    # Risk score: weighted combination of Pc and proximity
    # Pc contributes up to 70 points, proximity up to 30
    pc_score = min(pc * 70.0, 70.0)

    # Proximity score: closer = higher (scale: 100x combined radius)
    if combined_radius_km > 0.0:
        proximity_ratio = max(0.0, 1.0 - miss_distance_km / (100.0 * combined_radius_km))
    else:
        proximity_ratio = 0.0
    proximity_score = proximity_ratio * 30.0

    # Velocity factor: higher velocity = slightly higher risk
    velocity_factor = min(relative_velocity_km_s / 20.0, 1.0) * 5.0

    risk_score = pc_score + proximity_score + velocity_factor
    risk_score = min(risk_score, 100.0)

    # Risk level thresholds
    if risk_score >= 80.0:
        risk_level = "CRITICAL"
    elif risk_score >= 50.0:
        risk_level = "HIGH"
    elif risk_score >= 20.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return RiskAssessment(
        probability_of_collision=pc,
        miss_distance_km=miss_distance_km,
        risk_level=risk_level,
        risk_score=risk_score,
        combined_radius_km=combined_radius_km,
        relative_velocity_km_s=relative_velocity_km_s,
    )
