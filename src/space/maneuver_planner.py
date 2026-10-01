"""Maneuver planning: selection, timing optimization, fuel minimization."""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Optional


class ManeuverType(Enum):
    """Types of collision-avoidance maneuvers."""
    PROGRADE = "prograde"
    RETROGRADE = "retrograde"
    RADIAL = "radial"
    NORMAL = "normal"


@dataclass
class Maneuver:
    """A single maneuver burn."""
    maneuver_type: ManeuverType
    execution_time: datetime
    direction: int  # +1 or -1
    estimated_dv: float  # m/s
    fuel_estimate: float  # kg

    def __post_init__(self):
        if self.direction not in (1, -1):
            raise ValueError("Direction must be +1 or -1")
        if self.estimated_dv < 0:
            raise ValueError("Estimated delta-v must be non-negative")


@dataclass
class ManeuverPlan:
    """A complete maneuver plan consisting of one or more burns."""
    maneuvers: List[Maneuver] = field(default_factory=list)
    safety_margin_km: float = 1.0

    @property
    def total_estimated_dv(self) -> float:
        return sum(m.estimated_dv for m in self.maneuvers)

    @property
    def total_fuel_estimate(self) -> float:
        return sum(m.fuel_estimate for m in self.maneuvers)


@dataclass
class ManeuverConstraints:
    """Constraints for maneuver planning."""
    max_lead_time_hours: float = 48.0
    min_lead_time_hours: float = 1.0
    safety_margin_km: float = 1.0
    max_dv_budget: float = 100.0  # m/s


class ManeuverPlanner:
    """Plans collision-avoidance maneuvers."""

    # Reference epoch for timing calculations
    _EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Fraction of TCA to use as optimal lead time (before clamping)
    _OPTIMAL_LEAD_FRACTION = 0.25

    # Minimum lead time in seconds
    _MIN_LEAD_SECONDS = 60.0

    def select_maneuver_type(self, conj: dict) -> ManeuverType:
        """Select maneuver type based on dominant relative position component.

        The relative position vector components map to:
          x → along-track (prograde/retrograde)
          y → cross-track (normal)
          z → radial (radial)
        """
        rel_pos = conj.get("relative_position", (0.0, 0.0, 0.0))
        ax, ay, az = abs(rel_pos[0]), abs(rel_pos[1]), abs(rel_pos[2])

        if ax >= ay and ax >= az:
            # Along-track dominant — choose prograde or retrograde based on sign
            return ManeuverType.PROGRADE if rel_pos[0] >= 0 else ManeuverType.RETROGRADE
        elif ay >= ax and ay >= az:
            # Cross-track dominant
            return ManeuverType.NORMAL
        else:
            # Radial dominant
            return ManeuverType.RADIAL

    def optimize_timing(self, conj: dict, constraints: Optional[ManeuverConstraints] = None) -> datetime:
        """Compute optimal maneuver execution time.

        The optimal time is a fraction of the time-to-TCA, clamped by
        min/max lead time constraints.
        """
        tca_seconds = conj.get("time_of_closest_approach", 0.0)
        tca = self._EPOCH + timedelta(seconds=tca_seconds)

        if constraints is None:
            constraints = ManeuverConstraints()

        # Optimal lead time as fraction of TCA
        optimal_lead = tca_seconds * self._OPTIMAL_LEAD_FRACTION

        # Clamp to constraints
        min_lead = max(constraints.min_lead_time_hours * 3600, self._MIN_LEAD_SECONDS)
        max_lead = constraints.max_lead_time_hours * 3600

        lead = max(min_lead, min(optimal_lead, max_lead))

        return tca - timedelta(seconds=lead)

    def minimize_fuel(self, conj: dict, constraints: Optional[ManeuverConstraints] = None) -> ManeuverPlan:
        """Generate a fuel-minimizing maneuver plan.

        Strategy: single impulsive burn at the optimal time.
        """
        mtype = self.select_maneuver_type(conj)
        exec_time = self.optimize_timing(conj, constraints)

        # Direction from relative position sign
        rel_pos = conj.get("relative_position", (0.0, 0.0, 0.0))
        direction = self._direction_from_type(mtype, rel_pos)

        # Estimate delta-v from miss distance (simplified model)
        miss_m = conj.get("miss_distance", 100.0)
        estimated_dv = self._estimate_dv(miss_m, mtype)

        # Fuel estimate: proportional to delta-v (simplified)
        fuel = self._estimate_fuel(estimated_dv)

        maneuver = Maneuver(
            maneuver_type=mtype,
            execution_time=exec_time,
            direction=direction,
            estimated_dv=estimated_dv,
            fuel_estimate=fuel,
        )

        safety_margin = constraints.safety_margin_km if constraints else 1.0
        return ManeuverPlan(maneuvers=[maneuver], safety_margin_km=safety_margin)

    def plan_maneuver(
        self, conj: dict, constraints: Optional[ManeuverConstraints] = None
    ) -> ManeuverPlan:
        """Generate a complete maneuver plan for a conjunction.

        This is the main entry point: selects maneuver type, optimizes
        timing, and minimizes fuel.
        """
        return self.minimize_fuel(conj, constraints)

    def _direction_from_type(self, mtype: ManeuverType, rel_pos: tuple) -> int:
        """Determine burn direction from maneuver type and relative position."""
        if mtype == ManeuverType.PROGRADE:
            return 1 if rel_pos[0] >= 0 else -1
        elif mtype == ManeuverType.RETROGRADE:
            return -1 if rel_pos[0] >= 0 else 1
        elif mtype == ManeuverType.NORMAL:
            return 1 if rel_pos[1] >= 0 else -1
        else:  # RADIAL
            return 1 if rel_pos[2] >= 0 else -1

    def _estimate_dv(self, miss_distance_m: float, mtype: ManeuverType) -> float:
        """Estimate required delta-v (simplified model).

        This is a placeholder for actual delta-v computation.
        The real implementation would use orbital mechanics.
        """
        # Base delta-v proportional to miss distance
        base_dv = math.sqrt(miss_distance_m) * 0.5

        # Scale by maneuver type (plane changes are more expensive)
        if mtype == ManeuverType.NORMAL:
            base_dv *= 2.0
        elif mtype == ManeuverType.RADIAL:
            base_dv *= 1.5

        return max(base_dv, 0.1)  # Minimum 0.1 m/s

    def _estimate_fuel(self, dv: float) -> float:
        """Estimate fuel consumption from delta-v (simplified).

        Uses a linear approximation: fuel ∝ delta-v.
        """
        # Specific impulse ~300s, g0=9.81 m/s²
        # m_fuel = m_dry * (exp(dv / (Isp * g0)) - 1)
        # Simplified: assume 1000 kg spacecraft
        isp = 300.0
        g0 = 9.81
        m_dry = 1000.0
        return m_dry * (math.exp(dv / (isp * g0)) - 1.0)