"""Tests for conjunction detection: closest approach, collision probability, CDM."""
import math
import pytest
from datetime import datetime, timezone

from src.space.conjunction import (
    closest_approach,
    collision_probability,
    generate_cdm,
    ConjunctionEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(x, y, z, vx, vy, vz):
    """Build a state vector dict."""
    return {"position": (x, y, z), "velocity": (vx, vy, vz)}


# ---------------------------------------------------------------------------
# closest_approach
# ---------------------------------------------------------------------------

class TestClosestApproach:
    def test_head_on_collision_miss_distance_zero(self):
        """Two objects on collision course — miss distance should be ~0."""
        s1 = _state(0, 0, 0, 1000, 0, 0)
        s2 = _state(10000, 0, 0, -1000, 0, 0)
        result = closest_approach(s1, s2)
        assert result.miss_distance < 1.0  # metres

    def test_head_on_collision_tca_halfway(self):
        """TCA should be at the midpoint for symmetric head-on."""
        s1 = _state(0, 0, 0, 1000, 0, 0)
        s2 = _state(10000, 0, 0, -1000, 0, 0)
        result = closest_approach(s1, s2)
        assert result.time_of_closest_approach == pytest.approx(5.0, abs=0.01)

    def test_parallel_same_velocity_no_approach(self):
        """Objects moving in parallel at same velocity — constant separation."""
        s1 = _state(0, 0, 0, 100, 0, 0)
        s2 = _state(0, 5000, 0, 100, 0, 0)
        result = closest_approach(s1, s2)
        assert result.miss_distance == pytest.approx(5000.0, abs=0.1)
        assert result.time_of_closest_approach == pytest.approx(0.0, abs=0.01)

    def test_crossing_orbits_perpendicular(self):
        """Perpendicular crossing — miss distance is the minimum separation."""
        s1 = _state(0, 0, 0, 1000, 0, 0)
        s2 = _state(5000, -5000, 0, 0, 1000, 0)
        result = closest_approach(s1, s2)
        # They cross at (5000, 0, 0) at t=5
        assert result.miss_distance < 1.0
        assert result.time_of_closest_approach == pytest.approx(5.0, abs=0.01)

    def test_tca_in_the_past(self):
        """If closest approach was in the past, tca should be negative."""
        s1 = _state(0, 0, 0, 1000, 0, 0)
        s2 = _state(10000, 0, 0, 2000, 0, 0)
        result = closest_approach(s1, s2)
        # s2 is ahead and faster — they were closest in the past
        assert result.time_of_closest_approach < 0

    def test_identical_states_zero_miss(self):
        """Identical state vectors — zero miss distance at t=0."""
        s1 = _state(100, 200, 300, 1, 2, 3)
        s2 = _state(100, 200, 300, 1, 2, 3)
        result = closest_approach(s1, s2)
        assert result.miss_distance == pytest.approx(0.0, abs=0.001)
        assert result.time_of_closest_approach == pytest.approx(0.0, abs=0.001)

    def test_stationary_objects(self):
        """Two stationary objects — miss distance is their separation."""
        s1 = _state(0, 0, 0, 0, 0, 0)
        s2 = _state(3000, 4000, 0, 0, 0, 0)
        result = closest_approach(s1, s2)
        assert result.miss_distance == pytest.approx(5000.0, abs=0.1)

    def test_returns_conjunction_event(self):
        """Result should be a ConjunctionEvent dataclass."""
        s1 = _state(0, 0, 0, 100, 0, 0)
        s2 = _state(1000, 0, 0, -100, 0, 0)
        result = closest_approach(s1, s2)
        assert isinstance(result, ConjunctionEvent)

    def test_miss_distance_always_non_negative(self):
        """Miss distance must never be negative."""
        s1 = _state(0, 0, 0, 500, 500, 500)
        s2 = _state(10000, 20000, 30000, -500, -500, -500)
        result = closest_approach(s1, s2)
        assert result.miss_distance >= 0

    def test_relative_velocity_zero_tca_zero(self):
        """When relative velocity is zero, TCA defaults to 0."""
        s1 = _state(0, 0, 0, 100, 0, 0)
        s2 = _state(5000, 0, 0, 100, 0, 0)
        result = closest_approach(s1, s2)
        assert result.time_of_closest_approach == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# collision_probability
# ---------------------------------------------------------------------------

class TestCollisionProbability:
    def test_zero_miss_distance_max_probability(self):
        """Zero miss distance should give maximum probability."""
        p = collision_probability(0.0, 10.0, 10.0)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_large_miss_distance_zero_probability(self):
        """Very large miss distance should give ~0 probability."""
        p = collision_probability(1e6, 10.0, 10.0)
        assert p < 1e-10

    def test_probability_increases_with_combined_radius(self):
        """Larger combined radius → higher probability for same miss distance."""
        p_small = collision_probability(100.0, 5.0, 5.0)
        p_large = collision_probability(100.0, 20.0, 20.0)
        assert p_large > p_small

    def test_probability_decreases_with_miss_distance(self):
        """Larger miss distance → lower probability."""
        p_near = collision_probability(10.0, 10.0, 10.0)
        p_far = collision_probability(1000.0, 10.0, 10.0)
        assert p_near > p_far

    def test_probability_bounds_zero_to_one(self):
        """Probability must always be in [0, 1]."""
        for miss in [0, 1, 10, 100, 1000, 1e4, 1e5]:
            for r1, r2 in [(5, 5), (10, 20), (50, 50)]:
                p = collision_probability(float(miss), float(r1), float(r2))
                assert 0.0 <= p <= 1.0

    def test_symmetric_radii(self):
        """Swapping radii should not change probability."""
        p1 = collision_probability(50.0, 10.0, 20.0)
        p2 = collision_probability(50.0, 20.0, 10.0)
        assert p1 == pytest.approx(p2)

    def test_zero_radii_zero_probability(self):
        """Zero combined radius → zero probability."""
        p = collision_probability(0.0, 0.0, 0.0)
        assert p == pytest.approx(0.0, abs=1e-15)

    def test_typical_leo_conjunction(self):
        """Typical LEO conjunction: 100m miss, 10m combined radius."""
        p = collision_probability(100.0, 5.0, 5.0)
        # Should be small but non-zero
        assert 0.0 < p < 0.1


# ---------------------------------------------------------------------------
# generate_cdm
# ---------------------------------------------------------------------------

class TestGenerateCDM:
    def test_cdm_contains_required_fields(self):
        """CDM must contain all mandatory CCSDS fields."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        assert "header" in cdm
        assert "relative_metadata" in cdm
        assert "object1" in cdm
        assert "object2" in cdm

    def test_cdm_header_fields(self):
        """CDM header must have message ID and creation date."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        header = cdm["header"]
        assert "message_id" in header
        assert "creation_date" in header
        assert "originator" in header

    def test_cdm_relative_metadata(self):
        """Relative metadata must include TCA, miss distance, and collision probability."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        rel = cdm["relative_metadata"]
        assert "tca" in rel
        assert "miss_distance" in rel
        assert "collision_probability" in rel

    def test_cdm_object_data_present(self):
        """Both object sections must exist with state data."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        assert "object1" in cdm
        assert "object2" in cdm

    def test_cdm_miss_distance_matches_ca(self):
        """CDM miss distance should match the closest approach result."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        assert cdm["relative_metadata"]["miss_distance"] == pytest.approx(ca.miss_distance)

    def test_cdm_tca_matches_ca(self):
        """CDM TCA should match the closest approach result."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        assert cdm["relative_metadata"]["tca"] == pytest.approx(ca.time_of_closest_approach)

    def test_cdm_with_custom_radii(self):
        """CDM should use custom combined radius for probability."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca, radius1=15.0, radius2=15.0)
        p = cdm["relative_metadata"]["collision_probability"]
        assert 0.0 <= p <= 1.0

    def test_cdm_message_id_contains_object_names(self):
        """Message ID should reference both objects."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("ALPHA", "BETA", s1, s2, ca)
        mid = cdm["header"]["message_id"]
        assert "ALPHA" in mid
        assert "BETA" in mid

    def test_cdm_creation_date_is_iso_format(self):
        """Creation date should be ISO 8601 format."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        date_str = cdm["header"]["creation_date"]
        # Should be parseable as ISO format
        datetime.fromisoformat(date_str.replace("Z", "+00:00"))

    def test_cdm_collision_probability_present(self):
        """CDM must include a collision probability value."""
        s1 = _state(0, 0, 0, 7000, 0, 0)
        s2 = _state(5000, 0, 0, -7000, 0, 0)
        ca = closest_approach(s1, s2)
        cdm = generate_cdm("SAT-A", "SAT-B", s1, s2, ca)
        p = cdm["relative_metadata"]["collision_probability"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
