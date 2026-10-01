"""Tests for maneuver planning: selection, timing optimization, fuel minimization."""
import math
import pytest
from datetime import datetime, timezone, timedelta

from src.space.maneuver_planner import (
    ManeuverType,
    Maneuver,
    ManeuverPlan,
    ManeuverPlanner,
    ManeuverConstraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conjunction(
    miss_distance_m=500.0,
    tca_seconds=3600.0,
    rel_pos=(100.0, 200.0, 50.0),
    rel_vel=(10.0, 5.0, 2.0),
):
    """Build a conjunction event dict."""
    return {
        "miss_distance": miss_distance_m,
        "time_of_closest_approach": tca_seconds,
        "relative_position": rel_pos,
        "relative_velocity": rel_vel,
    }


def _constraints(
    max_lead_time_hours=48,
    min_lead_time_hours=1,
    safety_margin_km=1.0,
    max_dv_budget=100.0,
):
    return ManeuverConstraints(
        max_lead_time_hours=max_lead_time_hours,
        min_lead_time_hours=min_lead_time_hours,
        safety_margin_km=safety_margin_km,
        max_dv_budget=max_dv_budget,
    )


# ---------------------------------------------------------------------------
# ManeuverType
# ---------------------------------------------------------------------------

class TestManeuverType:
    def test_has_four_types(self):
        assert len(ManeuverType) == 4

    def test_prograde_exists(self):
        assert ManeuverType.PROGRADE.value == "prograde"

    def test_retrograde_exists(self):
        assert ManeuverType.RETROGRADE.value == "retrograde"

    def test_radial_exists(self):
        assert ManeuverType.RADIAL.value == "radial"

    def test_normal_exists(self):
        assert ManeuverType.NORMAL.value == "normal"


# ---------------------------------------------------------------------------
# Maneuver dataclass
# ---------------------------------------------------------------------------

class TestManeuver:
    def test_creation(self):
        m = Maneuver(
            maneuver_type=ManeuverType.PROGRADE,
            execution_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            direction=1,
            estimated_dv=5.0,
            fuel_estimate=1.0,
        )
        assert m.maneuver_type == ManeuverType.PROGRADE
        assert m.direction == 1
        assert m.estimated_dv == 5.0

    def test_direction_must_be_plus_or_minus_one(self):
        with pytest.raises(ValueError):
            Maneuver(
                maneuver_type=ManeuverType.PROGRADE,
                execution_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                direction=0,
                estimated_dv=5.0,
                fuel_estimate=1.0,
            )

    def test_negative_dv_raises(self):
        with pytest.raises(ValueError):
            Maneuver(
                maneuver_type=ManeuverType.PROGRADE,
                execution_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                direction=1,
                estimated_dv=-1.0,
                fuel_estimate=1.0,
            )


# ---------------------------------------------------------------------------
# ManeuverPlan
# ---------------------------------------------------------------------------

class TestManeuverPlan:
    def test_empty_plan(self):
        plan = ManeuverPlan(maneuvers=[])
        assert plan.maneuvers == []
        assert plan.total_estimated_dv == 0.0
        assert plan.total_fuel_estimate == 0.0

    def test_plan_totals(self):
        m1 = Maneuver(ManeuverType.PROGRADE, datetime(2026, 1, 1, tzinfo=timezone.utc), 1, 5.0, 1.0)
        m2 = Maneuver(ManeuverType.NORMAL, datetime(2026, 1, 1, tzinfo=timezone.utc), -1, 3.0, 0.6)
        plan = ManeuverPlan(maneuvers=[m1, m2])
        assert plan.total_estimated_dv == pytest.approx(8.0)
        assert plan.total_fuel_estimate == pytest.approx(1.6)

    def test_plan_safety_margin(self):
        plan = ManeuverPlan(maneuvers=[], safety_margin_km=2.5)
        assert plan.safety_margin_km == 2.5


# ---------------------------------------------------------------------------
# ManeuverPlanner — selection
# ---------------------------------------------------------------------------

class TestManeuverSelection:
    def test_large_along_track_selects_prograde(self):
        """Large along-track separation → prograde/retrograde to change period."""
        conj = _conjunction(rel_pos=(5000.0, 100.0, 50.0))
        planner = ManeuverPlanner()
        mtype = planner.select_maneuver_type(conj)
        assert mtype in (ManeuverType.PROGRADE, ManeuverType.RETROGRADE)

    def test_large_cross_track_selects_normal(self):
        """Large cross-track separation → normal burn to change plane."""
        conj = _conjunction(rel_pos=(100.0, 100.0, 5000.0))
        planner = ManeuverPlanner()
        mtype = planner.select_maneuver_type(conj)
        assert mtype == ManeuverType.NORMAL

    def test_large_radial_selects_radial(self):
        """Large radial separation → radial burn."""
        conj = _conjunction(rel_pos=(5000.0, 100.0, 100.0))
        planner = ManeuverPlanner()
        mtype = planner.select_maneuver_type(conj)
        assert mtype == ManeuverType.RADIAL

    def test_dominant_component_determines_type(self):
        """The largest relative position component determines maneuver type."""
        # Along-track dominant
        conj = _conjunction(rel_pos=(10000.0, 200.0, 100.0))
        planner = ManeuverPlanner()
        assert planner.select_maneuver_type(conj) in (ManeuverType.PROGRADE, ManeuverType.RETROGRADE)

        # Cross-track dominant
        conj = _conjunction(rel_pos=(100.0, 200.0, 10000.0))
        assert planner.select_maneuver_type(conj) == ManeuverType.NORMAL

        # Radial dominant
        conj = _conjunction(rel_pos=(10000.0, 100.0, 100.0))
        assert planner.select_maneuver_type(conj) == ManeuverType.RADIAL


# ---------------------------------------------------------------------------
# ManeuverPlanner — timing optimization
# ---------------------------------------------------------------------------

class TestTimingOptimization:
    def test_optimal_time_before_tca(self):
        """Optimal maneuver time should be before TCA."""
        conj = _conjunction(tca_seconds=3600.0)
        planner = ManeuverPlanner()
        opt_time = planner.optimize_timing(conj)
        tca = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=3600)
        assert opt_time < tca

    def test_respects_min_lead_time(self):
        """Maneuver should not be too close to TCA."""
        conj = _conjunction(tca_seconds=3600.0)
        planner = ManeuverPlanner()
        opt_time = planner.optimize_timing(conj)
        tca = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=3600)
        lead_time = (tca - opt_time).total_seconds()
        assert lead_time >= 60  # At least 1 minute

    def test_respects_max_lead_time(self):
        """Maneuver should not be too far from TCA."""
        conj = _conjunction(tca_seconds=86400.0 * 5)  # 5 days
        planner = ManeuverPlanner()
        opt_time = planner.optimize_timing(conj)
        tca = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=86400.0 * 5)
        lead_time = (tca - opt_time).total_seconds()
        assert lead_time <= 86400 * 2  # At most 2 days

    def test_earlier_tca_gets_earlier_maneuver(self):
        """Shorter time-to-TCA → maneuver closer to TCA."""
        conj_short = _conjunction(tca_seconds=1800.0)
        conj_long = _conjunction(tca_seconds=7200.0)
        planner = ManeuverPlanner()
        t_short = planner.optimize_timing(conj_short)
        t_long = planner.optimize_timing(conj_long)
        tca_short = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=1800)
        tca_long = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=7200)
        lead_short = (tca_short - t_short).total_seconds()
        lead_long = (tca_long - t_long).total_seconds()
        assert lead_short < lead_long


# ---------------------------------------------------------------------------
# ManeuverPlanner — fuel minimization
# ---------------------------------------------------------------------------

class TestFuelMinimization:
    def test_prefers_single_burn_over_multiple(self):
        """Single burn should be preferred over multiple burns for fuel."""
        planner = ManeuverPlanner()
        conj = _conjunction()
        plan = planner.minimize_fuel(conj)
        assert len(plan.maneuvers) == 1

    def test_fuel_estimate_positive(self):
        """Fuel estimate should be positive."""
        planner = ManeuverPlanner()
        conj = _conjunction()
        plan = planner.minimize_fuel(conj)
        assert plan.total_fuel_estimate > 0

    def test_larger_miss_distance_needs_less_fuel(self):
        """Larger miss distance → smaller correction → less fuel."""
        planner = ManeuverPlanner()
        conj_small = _conjunction(miss_distance_m=100.0)
        conj_large = _conjunction(miss_distance_m=5000.0)
        plan_small = planner.minimize_fuel(conj_small)
        plan_large = planner.minimize_fuel(conj_large)
        assert plan_large.total_fuel_estimate <= plan_small.total_fuel_estimate


# ---------------------------------------------------------------------------
# ManeuverPlanner — full plan
# ---------------------------------------------------------------------------

class TestPlanManeuver:
    def test_plan_has_maneuver(self):
        """A plan should contain at least one maneuver."""
        planner = ManeuverPlanner()
        conj = _conjunction()
        plan = planner.plan_maneuver(conj)
        assert len(plan.maneuvers) >= 1

    def test_plan_maneuver_type_matches_selection(self):
        """Plan maneuver type should match the selected type."""
        planner = ManeuverPlanner()
        conj = _conjunction(rel_pos=(100.0, 100.0, 5000.0))
        plan = planner.plan_maneuver(conj)
        assert plan.maneuvers[0].maneuver_type == ManeuverType.NORMAL

    def test_plan_total_dv_positive(self):
        """Total estimated delta-v should be positive."""
        planner = ManeuverPlanner()
        conj = _conjunction()
        plan = planner.plan_maneuver(conj)
        assert plan.total_estimated_dv > 0

    def test_plan_with_constraints(self):
        """Plan should respect constraints."""
        planner = ManeuverPlanner()
        conj = _conjunction(tca_seconds=3600.0)
        cons = _constraints(max_lead_time_hours=2, min_lead_time_hours=0.5)
        plan = planner.plan_maneuver(conj, constraints=cons)
        assert len(plan.maneuvers) >= 1

    def test_plan_safety_margin_from_constraints(self):
        """Plan safety margin should come from constraints."""
        planner = ManeuverPlanner()
        conj = _conjunction()
        cons = _constraints(safety_margin_km=5.0)
        plan = planner.plan_maneuver(conj, constraints=cons)
        assert plan.safety_margin_km == 5.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_miss_distance(self):
        """Zero miss distance (direct collision) should still produce a plan."""
        planner = ManeuverPlanner()
        conj = _conjunction(miss_distance_m=0.0)
        plan = planner.plan_maneuver(conj)
        assert len(plan.maneuvers) >= 1

    def test_very_small_miss_distance(self):
        """Very small miss distance should produce a plan."""
        planner = ManeuverPlanner()
        conj = _conjunction(miss_distance_m=1.0)
        plan = planner.plan_maneuver(conj)
        assert len(plan.maneuvers) >= 1

    def test_very_large_miss_distance(self):
        """Very large miss distance should produce a plan."""
        planner = ManeuverPlanner()
        conj = _conjunction(miss_distance_m=100000.0)
        plan = planner.plan_maneuver(conj)
        assert len(plan.maneuvers) >= 1

    def test_tca_at_zero(self):
        """TCA at zero should still produce a plan."""
        planner = ManeuverPlanner()
        conj = _conjunction(tca_seconds=0.0)
        plan = planner.plan_maneuver(conj)
        assert len(plan.maneuvers) >= 1

    def test_negative_tca(self):
        """Negative TCA (past conjunction) should still produce a plan."""
        planner = ManeuverPlanner()
        conj = _conjunction(tca_seconds=-100.0)
        plan = planner.plan_maneuver(conj)
        assert len(plan.maneuvers) >= 1

    def test_symmetric_relative_position(self):
        """Symmetric relative position should select along-track maneuver."""
        planner = ManeuverPlanner()
        conj = _conjunction(rel_pos=(5000.0, 5000.0, 100.0))
        mtype = planner.select_maneuver_type(conj)
        assert mtype in (ManeuverType.PROGRADE, ManeuverType.RETROGRADE)

    def test_all_components_equal(self):
        """All equal components — should still select a valid type."""
        planner = ManeuverPlanner()
        conj = _conjunction(rel_pos=(1000.0, 1000.0, 1000.0))
        mtype = planner.select_maneuver_type(conj)
        assert mtype in ManeuverType