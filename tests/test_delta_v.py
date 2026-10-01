"""Tests for delta-v computation: Hohmann, bi-elliptic, plane change."""
import math
import pytest

from src.space.delta_v import (
    hohmann_transfer,
    bielliptic_transfer,
    plane_change,
    hohmann_plane_change,
    circular_orbit_velocity,
)

MU_EARTH = 3.986004418e14  # m^3/s^2


# ---------------------------------------------------------------------------
# circular_orbit_velocity
# ---------------------------------------------------------------------------

class TestCircularOrbitVelocity:
    def test_leo_velocity(self):
        """LEO at 400 km altitude should be ~7.67 km/s."""
        r = 6371e3 + 400e3  # Earth radius + 400 km
        v = circular_orbit_velocity(r)
        assert v == pytest.approx(7668, rel=0.001)

    def test_geo_velocity(self):
        """GEO orbit velocity should be ~3.07 km/s."""
        r = 42164e3
        v = circular_orbit_velocity(r)
        assert v == pytest.approx(3075, rel=0.001)

    def test_velocity_decreases_with_radius(self):
        """Higher orbit → lower velocity."""
        v_low = circular_orbit_velocity(7000e3)
        v_high = circular_orbit_velocity(42000e3)
        assert v_high < v_low

    def test_velocity_proportional_to_inverse_sqrt(self):
        """v ∝ 1/sqrt(r)."""
        v1 = circular_orbit_velocity(10000e3)
        v2 = circular_orbit_velocity(40000e3)
        assert v1 / v2 == pytest.approx(2.0, rel=0.001)

    def test_zero_radius_raises(self):
        """Zero radius should raise ValueError."""
        with pytest.raises(ValueError):
            circular_orbit_velocity(0)

    def test_negative_radius_raises(self):
        """Negative radius should raise ValueError."""
        with pytest.raises(ValueError):
            circular_orbit_velocity(-1000)


# ---------------------------------------------------------------------------
# hohmann_transfer
# ---------------------------------------------------------------------------

class TestHohmannTransfer:
    def test_leo_to_geo(self):
        """LEO (400 km) to GEO Hohmann transfer ~3.9 km/s total."""
        r1 = 6371e3 + 400e3
        r2 = 42164e3
        result = hohmann_transfer(r1, r2)
        # Known value: ~3.9 km/s
        assert result["total"] == pytest.approx(3900, rel=0.01)
        assert result["dv1"] > 0
        assert result["dv2"] > 0

    def test_same_orbit_zero_delta_v(self):
        """Transfer to same orbit should require zero delta-v."""
        result = hohmann_transfer(7000e3, 7000e3)
        assert result["dv1"] == pytest.approx(0.0, abs=1e-10)
        assert result["dv2"] == pytest.approx(0.0, abs=1e-10)
        assert result["total"] == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_transfer(self):
        """r1→r2 should give same total as r2→r1."""
        r1, r2 = 7000e3, 42000e3
        fwd = hohmann_transfer(r1, r2)
        rev = hohmann_transfer(r2, r1)
        assert fwd["total"] == pytest.approx(rev["total"], rel=1e-10)

    def test_dv1_always_positive(self):
        """First burn always increases velocity (prograde)."""
        result = hohmann_transfer(7000e3, 42000e3)
        assert result["dv1"] > 0

    def test_dv2_always_positive(self):
        """Second burn always increases velocity (prograde)."""
        result = hohmann_transfer(7000e3, 42000e3)
        assert result["dv2"] > 0

    def test_total_is_sum_of_burns(self):
        """Total delta-v must equal dv1 + dv2."""
        result = hohmann_transfer(7000e3, 42000e3)
        assert result["total"] == pytest.approx(result["dv1"] + result["dv2"])

    def test_small_transfer(self):
        """Small orbit-to-orbit transfer."""
        r1, r2 = 7000e3, 8000e3
        result = hohmann_transfer(r1, r2)
        assert result["total"] > 0
        assert result["total"] < 1000  # Less than 1 km/s

    def test_large_transfer(self):
        """Large transfer (LEO to Moon distance)."""
        r1 = 7000e3
        r2 = 384400e3  # Moon distance
        result = hohmann_transfer(r1, r2)
        assert result["total"] > 3000  # > 3 km/s

    def test_zero_radius_raises(self):
        """Zero radius should raise ValueError."""
        with pytest.raises(ValueError):
            hohmann_transfer(0, 7000e3)

    def test_negative_radius_raises(self):
        """Negative radius should raise ValueError."""
        with pytest.raises(ValueError):
            hohmann_transfer(-1000, 7000e3)

    def test_r2_less_than_r1_raises(self):
        """r2 must be >= r1."""
        with pytest.raises(ValueError):
            hohmann_transfer(42000e3, 7000e3)

    def test_returns_dict_with_required_keys(self):
        """Result must contain dv1, dv2, total."""
        result = hohmann_transfer(7000e3, 42000e3)
        assert "dv1" in result
        assert "dv2" in result
        assert "total" in result

    def test_very_small_transfer(self):
        """Very small orbit change."""
        r1, r2 = 7000e3, 7001e3
        result = hohmann_transfer(r1, r2)
        assert result["total"] > 0
        assert result["total"] < 10  # Very small


# ---------------------------------------------------------------------------
# bielliptic_transfer
# ---------------------------------------------------------------------------

class TestBiellipticTransfer:
    def test_basic_bielliptic(self):
        """Basic bi-elliptic transfer LEO to GEO."""
        r1 = 6371e3 + 400e3
        r2 = 42164e3
        rb = 100000e3  # Intermediate apoapsis
        result = bielliptic_transfer(r1, r2, rb)
        assert result["dv1"] > 0
        assert result["dv2"] > 0
        assert result["dv3"] > 0
        assert result["total"] > 0

    def test_same_orbit_zero_delta_v(self):
        """Same orbit bi-elliptic should be zero."""
        result = bielliptic_transfer(7000e3, 7000e3, 100000e3)
        assert result["total"] == pytest.approx(0.0, abs=1e-10)

    def test_total_is_sum_of_burns(self):
        """Total must equal dv1 + dv2 + dv3."""
        result = bielliptic_transfer(7000e3, 42000e3, 100000e3)
        assert result["total"] == pytest.approx(
            result["dv1"] + result["dv2"] + result["dv3"]
        )

    def test_large_intermediate_radius(self):
        """Very large intermediate radius."""
        r1, r2 = 7000e3, 42000e3
        rb = 500000e3
        result = bielliptic_transfer(r1, r2, rb)
        assert result["total"] > 0

    def test_bielliptic_can_be_cheaper_than_hohmann(self):
        """For large radius ratios (r2/r1 > ~12), bi-elliptic can be cheaper."""
        r1 = 7000e3
        r2 = 100000e3  # ratio ≈ 14.3, above crossover
        rb = 500000e3  # Very large intermediate
        hoh = hohmann_transfer(r1, r2)
        bie = bielliptic_transfer(r1, r2, rb)
        # With very large rb and r2/r1 > 12, bi-elliptic should be cheaper
        assert bie["total"] < hoh["total"]

    def test_rb_must_exceed_r2(self):
        """Intermediate radius must be > r2."""
        with pytest.raises(ValueError):
            bielliptic_transfer(7000e3, 42000e3, 40000e3)

    def test_zero_radius_raises(self):
        """Zero radius should raise ValueError."""
        with pytest.raises(ValueError):
            bielliptic_transfer(0, 7000e3, 100000e3)

    def test_negative_radius_raises(self):
        """Negative radius should raise ValueError."""
        with pytest.raises(ValueError):
            bielliptic_transfer(7000e3, -1000, 100000e3)

    def test_returns_dict_with_required_keys(self):
        """Result must contain dv1, dv2, dv3, total."""
        result = bielliptic_transfer(7000e3, 42000e3, 100000e3)
        assert "dv1" in result
        assert "dv2" in result
        assert "dv3" in result
        assert "total" in result

    def test_symmetric_transfer(self):
        """r1→r2 should give same total as r2→r1."""
        r1, r2, rb = 7000e3, 42000e3, 100000e3
        fwd = bielliptic_transfer(r1, r2, rb)
        rev = bielliptic_transfer(r2, r1, rb)
        assert fwd["total"] == pytest.approx(rev["total"], rel=1e-10)


# ---------------------------------------------------------------------------
# plane_change
# ---------------------------------------------------------------------------

class TestPlaneChange:
    def test_zero_angle_zero_delta_v(self):
        """Zero plane change angle → zero delta-v."""
        result = plane_change(7000.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_180_degree_plane_change(self):
        """180° plane change at 7 km/s → 14 km/s."""
        result = plane_change(7000.0, math.pi)
        assert result == pytest.approx(14000.0, rel=0.001)

    def test_small_angle(self):
        """Small plane change angle."""
        result = plane_change(7000.0, math.radians(10))
        assert result > 0
        assert result < 2000  # Less than 2 km/s

    def test_90_degree_plane_change(self):
        """90° plane change."""
        v = 7000.0
        result = plane_change(v, math.radians(90))
        assert result == pytest.approx(v * math.sqrt(2), rel=0.001)

    def test_delta_v_proportional_to_velocity(self):
        """Double velocity → double delta-v."""
        dv1 = plane_change(5000.0, math.radians(30))
        dv2 = plane_change(10000.0, math.radians(30))
        assert dv2 / dv1 == pytest.approx(2.0, rel=0.001)

    def test_negative_angle_raises(self):
        """Negative angle should raise ValueError."""
        with pytest.raises(ValueError):
            plane_change(7000.0, -0.1)

    def test_zero_velocity_raises(self):
        """Zero velocity should raise ValueError."""
        with pytest.raises(ValueError):
            plane_change(0.0, math.radians(30))

    def test_negative_velocity_raises(self):
        """Negative velocity should raise ValueError."""
        with pytest.raises(ValueError):
            plane_change(-1000.0, math.radians(30))

    def test_large_angle(self):
        """Large plane change angle (120°)."""
        v = 7000.0
        result = plane_change(v, math.radians(120))
        assert result > v  # More than 1x velocity


# ---------------------------------------------------------------------------
# hohmann_plane_change (combined)
# ---------------------------------------------------------------------------

class TestHohmannPlaneChange:
    def test_zero_angle_equals_hohmann(self):
        """Zero plane change should equal pure Hohmann."""
        r1, r2 = 7000e3, 42000e3
        hoh = hohmann_transfer(r1, r2)
        combined = hohmann_plane_change(r1, r2, 0.0)
        assert combined["total"] == pytest.approx(hoh["total"], rel=1e-10)

    def test_with_plane_change_more_expensive(self):
        """Adding plane change should increase total delta-v."""
        r1, r2 = 7000e3, 42000e3
        hoh = hohmann_transfer(r1, r2)
        combined = hohmann_plane_change(r1, r2, math.radians(30))
        assert combined["total"] > hoh["total"]

    def test_180_degree_plane_change(self):
        """180° plane change combined with Hohmann."""
        r1, r2 = 7000e3, 42000e3
        result = hohmann_plane_change(r1, r2, math.pi)
        assert result["total"] > 0

    def test_same_orbit_with_plane_change(self):
        """Same orbit with plane change."""
        result = hohmann_plane_change(7000e3, 7000e3, math.radians(30))
        assert result["total"] > 0

    def test_negative_angle_raises(self):
        """Negative angle should raise ValueError."""
        with pytest.raises(ValueError):
            hohmann_plane_change(7000e3, 42000e3, -0.1)

    def test_zero_radius_raises(self):
        """Zero radius should raise ValueError."""
        with pytest.raises(ValueError):
            hohmann_plane_change(0, 42000e3, 0.0)

    def test_r2_less_than_r1_raises(self):
        """r2 must be >= r1."""
        with pytest.raises(ValueError):
            hohmann_plane_change(42000e3, 7000e3, 0.0)

    def test_returns_dict_with_required_keys(self):
        """Result must contain dv1, dv2, total."""
        result = hohmann_plane_change(7000e3, 42000e3, math.radians(15))
        assert "dv1" in result
        assert "dv2" in result
        assert "total" in result

    def test_total_is_sum_of_burns(self):
        """Total must equal dv1 + dv2."""
        result = hohmann_plane_change(7000e3, 42000e3, math.radians(20))
        assert result["total"] == pytest.approx(result["dv1"] + result["dv2"])
