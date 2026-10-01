"""Tests for collision risk: miss distance, Pc calculation, risk assessment."""
import math
import pytest
from datetime import datetime, timezone

from src.space.collision_risk import (
    compute_miss_distance,
    compute_pc,
    assess_risk,
    RiskAssessment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(x, y, z, vx, vy, vz):
    """Build a state vector dict (km, km/s)."""
    return {"position": (x, y, z), "velocity": (vx, vy, vz)}


# ---------------------------------------------------------------------------
# compute_miss_distance
# ---------------------------------------------------------------------------

class TestComputeMissDistance:
    def test_head_on_collision_zero_miss(self):
        """Head-on collision course → miss distance ~0."""
        s1 = _state(0, 0, 0, 1.0, 0, 0)
        s2 = _state(10, 0, 0, -1.0, 0, 0)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_parallel_tracks_constant_separation(self):
        """Parallel tracks at same velocity → constant separation."""
        s1 = _state(0, 0, 0, 1.0, 0, 0)
        s2 = _state(0, 5.0, 0, 1.0, 0, 0)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(5.0, abs=0.01)

    def test_crossing_orbits_minimum_separation(self):
        """Perpendicular crossing → minimum separation at crossing point."""
        s1 = _state(0, 0, 0, 1.0, 0, 0)
        s2 = _state(5.0, -5.0, 0, 0, 1.0, 0)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_stationary_objects_separation(self):
        """Stationary objects → miss distance is their separation."""
        s1 = _state(0, 0, 0, 0, 0, 0)
        s2 = _state(3.0, 4.0, 0, 0, 0, 0)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(5.0, abs=0.01)

    def test_miss_distance_non_negative(self):
        """Miss distance must never be negative."""
        s1 = _state(0, 0, 0, 0.5, 0.5, 0.5)
        s2 = _state(10, 20, 30, -0.5, -0.5, -0.5)
        d = compute_miss_distance(s1, s2)
        assert d >= 0

    def test_identical_states_zero_miss(self):
        """Identical state vectors → zero miss distance."""
        s1 = _state(1, 2, 3, 0.1, 0.2, 0.3)
        s2 = _state(1, 2, 3, 0.1, 0.2, 0.3)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_miss_distance_in_km(self):
        """Verify units are in km (not metres)."""
        s1 = _state(0, 0, 0, 0, 0, 0)
        s2 = _state(1.0, 0, 0, 0, 0, 0)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(1.0, abs=0.001)

    def test_typical_leo_conjunction(self):
        """Typical LEO conjunction: ~100m miss distance."""
        s1 = _state(6878.0, 0, 0, 0, 7.5, 0)
        s2 = _state(6878.0, 0.1, 0, 0, 7.5, 0)
        d = compute_miss_distance(s1, s2)
        assert d == pytest.approx(0.1, abs=0.01)


# ---------------------------------------------------------------------------
# compute_pc
# ---------------------------------------------------------------------------

class TestComputePc:
    def test_zero_miss_distance_max_pc(self):
        """Zero miss distance → Pc = 1.0."""
        p = compute_pc(0.0, 10.0)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_large_miss_distance_zero_pc(self):
        """Very large miss distance → Pc ≈ 0."""
        p = compute_pc(1000.0, 10.0)
        assert p < 1e-10

    def test_pc_decreases_with_miss_distance(self):
        """Larger miss distance → lower Pc."""
        p_near = compute_pc(1.0, 10.0)
        p_far = compute_pc(100.0, 10.0)
        assert p_near > p_far

    def test_pc_increases_with_combined_radius(self):
        """Larger combined radius → higher Pc for same miss distance."""
        p_small = compute_pc(10.0, 5.0)
        p_large = compute_pc(10.0, 20.0)
        assert p_large > p_small

    def test_pc_bounds_zero_to_one(self):
        """Pc must always be in [0, 1]."""
        for miss in [0, 0.1, 1, 10, 100, 1000]:
            for radius in [1, 5, 10, 50]:
                p = compute_pc(float(miss), float(radius))
                assert 0.0 <= p <= 1.0

    def test_pc_symmetric_in_radii(self):
        """Swapping radii should not change Pc."""
        p1 = compute_pc(5.0, 10.0, 20.0)
        p2 = compute_pc(5.0, 20.0, 10.0)
        assert p1 == pytest.approx(p2)

    def test_pc_with_custom_sigma(self):
        """Custom sigma values should affect Pc."""
        p_default = compute_pc(10.0, 10.0)
        p_large_sigma = compute_pc(10.0, 10.0, sigma_x=50.0, sigma_y=50.0)
        assert p_large_sigma > p_default

    def test_pc_with_small_sigma(self):
        """Small sigma → lower Pc for same miss distance."""
        p_default = compute_pc(10.0, 10.0)
        p_small_sigma = compute_pc(10.0, 10.0, sigma_x=1.0, sigma_y=1.0)
        assert p_small_sigma < p_default

    def test_pc_typical_leo_conjunction(self):
        """Typical LEO: 100m miss, 10m combined radius."""
        p = compute_pc(0.1, 0.01)
        assert 0.0 < p < 1.0

    def test_pc_zero_radius_zero(self):
        """Zero combined radius → zero Pc."""
        p = compute_pc(0.0, 0.0)
        assert p == pytest.approx(0.0, abs=1e-15)

    def test_pc_with_sigma_larger_than_radius(self):
        """Sigma larger than radius → higher Pc."""
        p = compute_pc(5.0, 1.0, sigma_x=10.0, sigma_y=10.0)
        assert p > 0.5


# ---------------------------------------------------------------------------
# assess_risk
# ---------------------------------------------------------------------------

class TestAssessRisk:
    def test_low_risk_large_miss_distance(self):
        """Large miss distance → LOW risk."""
        r = assess_risk(miss_distance_km=100.0, combined_radius_km=0.01, pc=1e-8)
        assert r.risk_level == "LOW"

    def test_high_risk_small_miss_distance(self):
        """Small miss distance → HIGH or CRITICAL risk."""
        r = assess_risk(miss_distance_km=0.05, combined_radius_km=0.01, pc=0.5)
        assert r.risk_level in ("HIGH", "CRITICAL")

    def test_critical_risk_very_small_miss(self):
        """Very small miss distance → CRITICAL risk."""
        r = assess_risk(miss_distance_km=0.001, combined_radius_km=0.01, pc=0.9)
        assert r.risk_level == "CRITICAL"

    def test_risk_score_bounds(self):
        """Risk score must be in [0, 100]."""
        for miss in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            for radius in [0.001, 0.01, 0.1]:
                for pc in [0.0, 0.1, 0.5, 0.9, 1.0]:
                    r = assess_risk(miss, radius, pc)
                    assert 0.0 <= r.risk_score <= 100.0

    def test_risk_score_increases_with_pc(self):
        """Higher Pc → higher risk score."""
        r_low = assess_risk(0.1, 0.01, 0.01)
        r_high = assess_risk(0.1, 0.01, 0.9)
        assert r_high.risk_score > r_low.risk_score

    def test_risk_score_increases_with_smaller_miss(self):
        """Smaller miss distance → higher risk score."""
        r_far = assess_risk(10.0, 0.01, 0.5)
        r_near = assess_risk(0.01, 0.01, 0.5)
        assert r_near.risk_score > r_far.risk_score

    def test_risk_level_thresholds(self):
        """Risk levels follow expected thresholds."""
        # Very low Pc → LOW
        r = assess_risk(100.0, 0.01, 1e-10)
        assert r.risk_level == "LOW"

        # Moderate Pc and proximity → MEDIUM
        r = assess_risk(0.1, 0.01, 0.1)
        assert r.risk_level == "MEDIUM"

        # High Pc → HIGH
        r = assess_risk(0.1, 0.01, 0.5)
        assert r.risk_level == "HIGH"

    def test_risk_with_relative_velocity(self):
        """Higher relative velocity → higher risk."""
        r_slow = assess_risk(0.1, 0.01, 0.5, relative_velocity_km_s=1.0)
        r_fast = assess_risk(0.1, 0.01, 0.5, relative_velocity_km_s=15.0)
        assert r_fast.risk_score >= r_slow.risk_score

    def test_risk_returns_assessment_object(self):
        """assess_risk returns a RiskAssessment dataclass."""
        r = assess_risk(0.1, 0.01, 0.5)
        assert isinstance(r, RiskAssessment)

    def test_risk_includes_all_fields(self):
        """RiskAssessment contains all expected fields."""
        r = assess_risk(0.1, 0.01, 0.5, relative_velocity_km_s=10.0)
        assert hasattr(r, "probability_of_collision")
        assert hasattr(r, "miss_distance_km")
        assert hasattr(r, "risk_level")
        assert hasattr(r, "risk_score")
        assert hasattr(r, "combined_radius_km")
        assert hasattr(r, "relative_velocity_km_s")

    def test_risk_level_ordering(self):
        """Risk levels have a clear semantic ordering."""
        level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for i in range(len(levels) - 1):
            assert level_order[levels[i]] < level_order[levels[i + 1]]


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_head_on(self):
        """Full pipeline: head-on collision → CRITICAL risk."""
        s1 = _state(0, 0, 0, 7.0, 0, 0)
        s2 = _state(100, 0, 0, -7.0, 0, 0)
        miss = compute_miss_distance(s1, s2)
        pc = compute_pc(miss, 0.01)
        risk = assess_risk(miss, 0.01, pc, relative_velocity_km_s=14.0)
        assert risk.risk_level == "CRITICAL"
        assert risk.probability_of_collision == pytest.approx(1.0, abs=0.01)

    def test_full_pipeline_distant(self):
        """Full pipeline: distant objects → LOW risk."""
        s1 = _state(0, 0, 0, 7.0, 0, 0)
        s2 = _state(1000, 0, 0, 7.0, 0, 0)
        miss = compute_miss_distance(s1, s2)
        pc = compute_pc(miss, 0.01)
        risk = assess_risk(miss, 0.01, pc, relative_velocity_km_s=0.0)
        assert risk.risk_level == "LOW"

    def test_full_pipeline_moderate(self):
        """Full pipeline: moderate miss → MEDIUM or HIGH risk."""
        s1 = _state(0, 0, 0, 7.0, 0, 0)
        s2 = _state(0.02, 0, 0, 7.0, 0, 0)
        miss = compute_miss_distance(s1, s2)
        pc = compute_pc(miss, 0.01)
        risk = assess_risk(miss, 0.01, pc, relative_velocity_km_s=0.0)
        assert risk.risk_level in ("MEDIUM", "HIGH")
