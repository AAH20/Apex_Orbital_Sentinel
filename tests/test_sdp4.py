"""Tests for SDP4 deep-space propagator — TDD: define expected behavior first."""
import math
import pytest
from src.space.sdp4 import SDP4Propagator, SDP4Error, DeepSpaceTLE

# Well-known deep-space TLEs for testing
# Molniya orbit (highly eccentric, ~12hr period)
MOLNIYA_TLE = (
    "1 25485U 98054A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25485  63.4349  88.2302 7144647 265.2202  12.4322  2.00600000 50581",
)

# GPS orbit (semi-synchronous, ~12hr period)
GPS_TLE = (
    "1 28129U 03059A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928",
    "2 28129  55.0000  86.2302 0014647  26.2202  12.4322  2.00273142 50581",
)

# Geostationary orbit (24hr period)
GEO_TLE = (
    "1 25994U 99068A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928",
    "2 25994   0.0552  86.2302 0004647  26.2202  12.4322  1.00273142 50581",
)

# Tundra orbit (24hr period, highly eccentric)
TUNDRA_TLE = (
    "1 25485U 98054A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25485  63.4349  88.2302 7144647 265.2202  12.4322  1.00000000 50586",
)


class TestDeepSpaceTLE:
    """Deep-space TLE parsing and classification tests."""

    def test_molniya_classification(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        assert tle.is_deep_space is True

    def test_gps_classification(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        assert tle.is_deep_space is True

    def test_geo_classification(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        assert tle.is_deep_space is True

    def test_tundra_classification(self):
        tle = DeepSpaceTLE(*TUNDRA_TLE)
        assert tle.is_deep_space is True

    def test_molniya_period(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        period = tle.period_minutes
        assert 600 < period < 800  # ~12 hours

    def test_geo_period(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        period = tle.period_minutes
        assert 1300 < period < 1500  # ~24 hours

    def test_molniya_eccentricity(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        assert tle.eccentricity > 0.7

    def test_geo_eccentricity(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        assert tle.eccentricity < 0.01

    def test_molniya_inclination(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        assert tle.inclination_deg == pytest.approx(63.4349, abs=1e-4)

    def test_gps_inclination(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        assert tle.inclination_deg == pytest.approx(55.0, abs=1e-4)

    def test_geo_inclination(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        assert tle.inclination_deg < 1.0

    def test_molniya_mean_motion(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        assert tle.mean_motion_revs_per_day < 4.0

    def test_geo_mean_motion(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        assert tle.mean_motion_revs_per_day == pytest.approx(1.00273142, abs=1e-8)

    def test_molniya_semi_major_axis(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        a = tle.semi_major_axis_km
        assert 20000 < a < 30000

    def test_geo_semi_major_axis(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        a = tle.semi_major_axis_km
        assert 42000 < a < 43000

    def test_molniya_apogee(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        a = tle.semi_major_axis_km
        e = tle.eccentricity
        ra = a * (1 + e)
        assert 40000 < ra < 50000

    def test_molniya_perigee(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        a = tle.semi_major_axis_km
        e = tle.eccentricity
        rp = a * (1 - e)
        assert 5000 < rp < 10000


class TestSDP4Propagation:
    """SDP4 propagation tests."""

    def test_molniya_position_magnitude(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 10000 < r_mag < 50000

    def test_molniya_velocity_magnitude(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert 1.0 < v_mag < 10.0

    def test_gps_position_magnitude(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 20000 < r_mag < 30000

    def test_gps_velocity_magnitude(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert 2.0 < v_mag < 5.0

    def test_geo_position_magnitude(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 42000 < r_mag < 43000

    def test_geo_velocity_magnitude(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert 2.5 < v_mag < 3.5

    def test_molniya_energy_conservation(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r0, v0 = prop.propagate(0.0)
        r1, v1 = prop.propagate(10.0)
        mu = 398600.4418
        e0 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) / 2 - mu / math.sqrt(
            r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
        )
        e1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) / 2 - mu / math.sqrt(
            r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
        )
        assert abs(e0 - e1) < 1e-3

    def test_gps_energy_conservation(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r0, v0 = prop.propagate(0.0)
        r1, v1 = prop.propagate(10.0)
        mu = 398600.4418
        e0 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) / 2 - mu / math.sqrt(
            r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
        )
        e1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) / 2 - mu / math.sqrt(
            r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
        )
        assert abs(e0 - e1) < 1e-3

    def test_geo_energy_conservation(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r0, v0 = prop.propagate(0.0)
        r1, v1 = prop.propagate(10.0)
        mu = 398600.4418
        e0 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) / 2 - mu / math.sqrt(
            r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
        )
        e1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) / 2 - mu / math.sqrt(
            r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
        )
        assert abs(e0 - e1) < 1e-3

    def test_molniya_angular_momentum_conservation(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r0, v0 = prop.propagate(0.0)
        r1, v1 = prop.propagate(10.0)
        h0 = [
            r0[1] * v0[2] - r0[2] * v0[1],
            r0[2] * v0[0] - r0[0] * v0[2],
            r0[0] * v0[1] - r0[1] * v0[0],
        ]
        h1 = [
            r1[1] * v1[2] - r1[2] * v1[1],
            r1[2] * v1[0] - r1[0] * v1[2],
            r1[0] * v1[1] - r1[1] * v1[0],
        ]
        h0_mag = math.sqrt(h0[0] ** 2 + h0[1] ** 2 + h0[2] ** 2)
        h1_mag = math.sqrt(h1[0] ** 2 + h1[1] ** 2 + h1[2] ** 2)
        assert abs(h0_mag - h1_mag) / h0_mag < 1e-6

    def test_gps_angular_momentum_conservation(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r0, v0 = prop.propagate(0.0)
        r1, v1 = prop.propagate(10.0)
        h0 = [
            r0[1] * v0[2] - r0[2] * v0[1],
            r0[2] * v0[0] - r0[0] * v0[2],
            r0[0] * v0[1] - r0[1] * v0[0],
        ]
        h1 = [
            r1[1] * v1[2] - r1[2] * v1[1],
            r1[2] * v1[0] - r1[0] * v1[2],
            r1[0] * v1[1] - r1[1] * v1[0],
        ]
        h0_mag = math.sqrt(h0[0] ** 2 + h0[1] ** 2 + h0[2] ** 2)
        h1_mag = math.sqrt(h1[0] ** 2 + h1[1] ** 2 + h1[2] ** 2)
        assert abs(h0_mag - h1_mag) / h0_mag < 1e-6

    def test_geo_angular_momentum_conservation(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r0, v0 = prop.propagate(0.0)
        r1, v1 = prop.propagate(10.0)
        h0 = [
            r0[1] * v0[2] - r0[2] * v0[1],
            r0[2] * v0[0] - r0[0] * v0[2],
            r0[0] * v0[1] - r0[1] * v0[0],
        ]
        h1 = [
            r1[1] * v1[2] - r1[2] * v1[1],
            r1[2] * v1[0] - r1[0] * v1[2],
            r1[0] * v1[1] - r1[1] * v1[0],
        ]
        h0_mag = math.sqrt(h0[0] ** 2 + h0[1] ** 2 + h0[2] ** 2)
        h1_mag = math.sqrt(h1[0] ** 2 + h1[1] ** 2 + h1[2] ** 2)
        assert abs(h0_mag - h1_mag) / h0_mag < 1e-6

    def test_molniya_periodicity(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(700.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist < 5000

    def test_gps_periodicity(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(700.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist < 5000

    def test_geo_periodicity(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(1440.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist < 5000

    def test_molniya_semi_major_axis_consistency(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        assert 20000 < a < 30000

    def test_gps_semi_major_axis_consistency(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        assert 20000 < a < 30000

    def test_geo_semi_major_axis_consistency(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        assert 42000 < a < 43000

    def test_molniya_eccentricity_consistency(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
        assert 0.6 < e < 0.8

    def test_gps_eccentricity_consistency(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
        assert e < 0.05

    def test_geo_eccentricity_consistency(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
        assert e < 0.01

    def test_molniya_inclination_consistency(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        inc = math.acos(h[2] / h_mag)
        assert math.degrees(inc) == pytest.approx(63.4349, abs=0.5)

    def test_gps_inclination_consistency(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        inc = math.acos(h[2] / h_mag)
        assert math.degrees(inc) == pytest.approx(55.0, abs=0.5)

    def test_geo_inclination_consistency(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        inc = math.acos(h[2] / h_mag)
        assert math.degrees(inc) < 1.0

    def test_molniya_raan_consistency(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        n = [-h[1], h[0], 0.0]
        n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2)
        raan = math.acos(n[0] / n_mag)
        if n[1] < 0:
            raan = 2 * math.pi - raan
        assert math.degrees(raan) == pytest.approx(88.2302, abs=1.0)

    def test_molniya_propagation_continuity(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(0.1)
        r2, _ = prop.propagate(0.2)
        d01 = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        d12 = math.sqrt(
            (r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2 + (r1[2] - r2[2]) ** 2
        )
        assert abs(d01 - d12) / d01 < 0.1

    def test_gps_propagation_continuity(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(0.1)
        r2, _ = prop.propagate(0.2)
        d01 = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        d12 = math.sqrt(
            (r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2 + (r1[2] - r2[2]) ** 2
        )
        assert abs(d01 - d12) / d01 < 0.1

    def test_geo_propagation_continuity(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(0.1)
        r2, _ = prop.propagate(0.2)
        d01 = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        d12 = math.sqrt(
            (r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2 + (r1[2] - r2[2]) ** 2
        )
        assert abs(d01 - d12) / d01 < 0.1

    def test_molniya_propagation_determinism(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_gps_propagation_determinism(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_geo_propagation_determinism(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_molniya_propagation_monotonic_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(10.0)
        r2, _ = prop.propagate(20.0)
        assert r0 != r1
        assert r1 != r2
        assert r0 != r2

    def test_gps_propagation_monotonic_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(10.0)
        r2, _ = prop.propagate(20.0)
        assert r0 != r1
        assert r1 != r2
        assert r0 != r2

    def test_geo_propagation_monotonic_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(10.0)
        r2, _ = prop.propagate(20.0)
        assert r0 != r1
        assert r1 != r2
        assert r0 != r2

    def test_molniya_propagation_zero_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_zero_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_zero_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_negative_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-10.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_negative_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-10.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_negative_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-10.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_large_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1440.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_large_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1440.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_large_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1440.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_very_small_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.001)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_very_small_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.001)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_very_small_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.001)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_very_large_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10000.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_very_large_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10000.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_very_large_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10000.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_extreme_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_extreme_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_extreme_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_negative_extreme_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_negative_extreme_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_negative_extreme_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_special_values(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        for t in [0.0, -0.0, 1.0, -1.0]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_special_values(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        for t in [0.0, -0.0, 1.0, -1.0]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_special_values(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        for t in [0.0, -0.0, 1.0, -1.0]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_boundary_values(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        for t in [0.0, 1e-6, 1e6]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_boundary_values(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        for t in [0.0, 1e-6, 1e6]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_boundary_values(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        for t in [0.0, 1e-6, 1e6]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_typical_values(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            r, v = prop.propagate(float(t))
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_typical_values(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            r, v = prop.propagate(float(t))
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_typical_values(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            r, v = prop.propagate(float(t))
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_all_components_finite(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert math.isfinite(x)
            for x in v:
                assert math.isfinite(x)

    def test_gps_propagation_all_components_finite(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert math.isfinite(x)
            for x in v:
                assert math.isfinite(x)

    def test_geo_propagation_all_components_finite(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert math.isfinite(x)
            for x in v:
                assert math.isfinite(x)

    def test_molniya_propagation_no_nan_values(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isnan(x)
            for x in v:
                assert not math.isnan(x)

    def test_gps_propagation_no_nan_values(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isnan(x)
            for x in v:
                assert not math.isnan(x)

    def test_geo_propagation_no_nan_values(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isnan(x)
            for x in v:
                assert not math.isnan(x)

    def test_molniya_propagation_no_inf_values(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isinf(x)
            for x in v:
                assert not math.isinf(x)

    def test_gps_propagation_no_inf_values(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isinf(x)
            for x in v:
                assert not math.isinf(x)

    def test_geo_propagation_no_inf_values(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isinf(x)
            for x in v:
                assert not math.isinf(x)

    def test_molniya_propagation_consistent_results(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        results = []
        for _ in range(5):
            r, v = prop.propagate(10.0)
            results.append((r, v))
        for i in range(1, len(results)):
            assert results[i][0] == results[0][0]
            assert results[i][1] == results[0][1]

    def test_gps_propagation_consistent_results(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        results = []
        for _ in range(5):
            r, v = prop.propagate(10.0)
            results.append((r, v))
        for i in range(1, len(results)):
            assert results[i][0] == results[0][0]
            assert results[i][1] == results[0][1]

    def test_geo_propagation_consistent_results(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        results = []
        for _ in range(5):
            r, v = prop.propagate(10.0)
            results.append((r, v))
        for i in range(1, len(results)):
            assert results[i][0] == results[0][0]
            assert results[i][1] == results[0][1]

    def test_molniya_propagation_different_instances(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop1 = SDP4Propagator(tle)
        prop2 = SDP4Propagator(tle)
        r1, v1 = prop1.propagate(10.0)
        r2, v2 = prop2.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_gps_propagation_different_instances(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop1 = SDP4Propagator(tle)
        prop2 = SDP4Propagator(tle)
        r1, v1 = prop1.propagate(10.0)
        r2, v2 = prop2.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_geo_propagation_different_instances(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop1 = SDP4Propagator(tle)
        prop2 = SDP4Propagator(tle)
        r1, v1 = prop1.propagate(10.0)
        r2, v2 = prop2.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_molniya_propagation_same_instance(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_gps_propagation_same_instance(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_geo_propagation_same_instance(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_molniya_propagation_different_times_different_results(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 != r2
        assert v1 != v2

    def test_gps_propagation_different_times_different_results(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 != r2
        assert v1 != v2

    def test_geo_propagation_different_times_different_results(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 != r2
        assert v1 != v2

    def test_molniya_propagation_sequential_times(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        results = []
        for t in range(0, 100, 10):
            r, v = prop.propagate(float(t))
            results.append((r, v))
        for i in range(len(results) - 1):
            assert results[i][0] != results[i + 1][0]

    def test_gps_propagation_sequential_times(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        results = []
        for t in range(0, 100, 10):
            r, v = prop.propagate(float(t))
            results.append((r, v))
        for i in range(len(results) - 1):
            assert results[i][0] != results[i + 1][0]

    def test_geo_propagation_sequential_times(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        results = []
        for t in range(0, 100, 10):
            r, v = prop.propagate(float(t))
            results.append((r, v))
        for i in range(len(results) - 1):
            assert results[i][0] != results[i + 1][0]

    def test_molniya_propagation_reverse_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2
        assert v1 != v2

    def test_gps_propagation_reverse_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2
        assert v1 != v2

    def test_geo_propagation_reverse_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2
        assert v1 != v2

    def test_molniya_propagation_symmetry(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2

    def test_gps_propagation_symmetry(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2

    def test_geo_propagation_symmetry(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2

    def test_molniya_propagation_time_zero_symmetry(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(0.0)
        assert r1 == r2
        assert v1 == v2

    def test_gps_propagation_time_zero_symmetry(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(0.0)
        assert r1 == r2
        assert v1 == v2

    def test_geo_propagation_time_zero_symmetry(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(0.0)
        assert r1 == r2
        assert v1 == v2

    def test_molniya_propagation_very_large_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_very_large_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_very_large_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_very_small_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_very_small_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_very_small_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_negative_very_large_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_negative_very_large_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_negative_very_large_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_negative_very_small_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_negative_very_small_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_negative_very_small_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(-1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_integer_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_integer_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_integer_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_float_time(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10.5)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_float_time(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10.5)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_float_time(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(10.5)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_list_output(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert len(r) == 3
        assert len(v) == 3
        _ = r[0]
        _ = r[1]
        _ = r[2]
        _ = v[0]
        _ = v[1]
        _ = v[2]

    def test_gps_propagation_list_output(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert len(r) == 3
        assert len(v) == 3
        _ = r[0]
        _ = r[1]
        _ = r[2]
        _ = v[0]
        _ = v[1]
        _ = v[2]

    def test_geo_propagation_list_output(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert len(r) == 3
        assert len(v) == 3
        _ = r[0]
        _ = r[1]
        _ = r[2]
        _ = v[0]
        _ = v[1]
        _ = v[2]

    def test_molniya_propagation_tuple_output(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        for x in r:
            assert math.isfinite(x)
        for x in v:
            assert math.isfinite(x)

    def test_gps_propagation_tuple_output(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        for x in r:
            assert math.isfinite(x)
        for x in v:
            assert math.isfinite(x)

    def test_geo_propagation_tuple_output(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r, v = prop.propagate(0.0)
        for x in r:
            assert math.isfinite(x)
        for x in v:
            assert math.isfinite(x)

    def test_molniya_propagation_multiple_calls(self):
        tle = DeepSpaceTLE(*MOLNIYA_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_gps_propagation_multiple_calls(self):
        tle = DeepSpaceTLE(*GPS_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_geo_propagation_multiple_calls(self):
        tle = DeepSpaceTLE(*GEO_TLE)
        prop = SDP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_molniya_propagation_different_tles(self):
        tle1 = DeepSpaceTLE(*MOLNIYA_TLE)
        tle2 = DeepSpaceTLE(*GPS_TLE)
        prop1 = SDP4Propagator(tle1)
        prop2 = SDP4Propagator(tle2)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        assert r1 != r2
        assert v1 != v2

    def test_gps_propagation_different_tles(self):
        tle1 = DeepSpaceTLE(*GPS_TLE)
        tle2 = DeepSpaceTLE(*GEO_TLE)
        prop1 = SDP4Propagator(tle1)
        prop2 = SDP4Propagator(tle2)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        assert r1 != r2
        assert v1 != v2

    def test_geo_propagation_different_tles(self):
        tle1 = DeepSpaceTLE(*GEO_TLE)
        tle2 = DeepSpaceTLE(*TUNDRA_TLE)
        prop1 = SDP4Propagator(tle1)
        prop2 = SDP4Propagator(tle2)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        assert r1 != r2
        assert v1 != v2

    def test_molniya_propagation_all_tles_valid(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_all_tles_valid(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_all_tles_valid(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_all_tles_different(self):
        tle1 = DeepSpaceTLE(*MOLNIYA_TLE)
        tle2 = DeepSpaceTLE(*GPS_TLE)
        tle3 = DeepSpaceTLE(*GEO_TLE)
        prop1 = SDP4Propagator(tle1)
        prop2 = SDP4Propagator(tle2)
        prop3 = SDP4Propagator(tle3)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        r3, v3 = prop3.propagate(0.0)
        assert r1 != r2
        assert r2 != r3
        assert r1 != r3

    def test_gps_propagation_all_tles_different(self):
        tle1 = DeepSpaceTLE(*GPS_TLE)
        tle2 = DeepSpaceTLE(*GEO_TLE)
        tle3 = DeepSpaceTLE(*TUNDRA_TLE)
        prop1 = SDP4Propagator(tle1)
        prop2 = SDP4Propagator(tle2)
        prop3 = SDP4Propagator(tle3)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        r3, v3 = prop3.propagate(0.0)
        assert r1 != r2
        assert r2 != r3
        assert r1 != r3

    def test_geo_propagation_all_tles_different(self):
        tle1 = DeepSpaceTLE(*GEO_TLE)
        tle2 = DeepSpaceTLE(*TUNDRA_TLE)
        prop1 = SDP4Propagator(tle1)
        prop2 = SDP4Propagator(tle2)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        assert r1 != r2

    def test_molniya_propagation_all_tles_consistent(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 == r2
            assert v1 == v2

    def test_gps_propagation_all_tles_consistent(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 == r2
            assert v1 == v2

    def test_geo_propagation_all_tles_consistent(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 == r2
            assert v1 == v2

    def test_molniya_propagation_all_tles_orbital_elements(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            inc = math.acos(h[2] / h_mag)
            assert a > 0
            assert 0 <= e < 1
            assert 0 <= inc <= math.pi

    def test_gps_propagation_all_tles_orbital_elements(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            inc = math.acos(h[2] / h_mag)
            assert a > 0
            assert 0 <= e < 1
            assert 0 <= inc <= math.pi

    def test_geo_propagation_all_tles_orbital_elements(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            inc = math.acos(h[2] / h_mag)
            assert a > 0
            assert 0 <= e < 1
            assert 0 <= inc <= math.pi

    def test_molniya_propagation_all_tles_energy(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert math.isfinite(energy)

    def test_gps_propagation_all_tles_energy(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert math.isfinite(energy)

    def test_geo_propagation_all_tles_energy(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert math.isfinite(energy)

    def test_molniya_propagation_all_tles_angular_momentum(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            assert h_mag > 0

    def test_gps_propagation_all_tles_angular_momentum(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            assert h_mag > 0

    def test_geo_propagation_all_tles_angular_momentum(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            assert h_mag > 0

    def test_molniya_propagation_all_tles_period(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            T = 2 * math.pi * math.sqrt(a ** 3 / mu)
            assert T > 0

    def test_gps_propagation_all_tles_period(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            T = 2 * math.pi * math.sqrt(a ** 3 / mu)
            assert T > 0

    def test_geo_propagation_all_tles_period(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            T = 2 * math.pi * math.sqrt(a ** 3 / mu)
            assert T > 0

    def test_molniya_propagation_all_tles_semi_major_axis(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert a > 0

    def test_gps_propagation_all_tles_semi_major_axis(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert a > 0

    def test_geo_propagation_all_tles_semi_major_axis(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert a > 0

    def test_molniya_propagation_all_tles_eccentricity(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            assert 0 <= e < 1

    def test_gps_propagation_all_tles_eccentricity(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            assert 0 <= e < 1

    def test_geo_propagation_all_tles_eccentricity(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            assert 0 <= e < 1

    def test_molniya_propagation_all_tles_inclination(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            inc = math.acos(h[2] / h_mag)
            assert 0 <= inc <= math.pi

    def test_gps_propagation_all_tles_inclination(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            inc = math.acos(h[2] / h_mag)
            assert 0 <= inc <= math.pi

    def test_geo_propagation_all_tles_inclination(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            inc = math.acos(h[2] / h_mag)
            assert 0 <= inc <= math.pi

    def test_molniya_propagation_all_tles_raan(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            n = [-h[1], h[0], 0.0]
            n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2)
            if n_mag > 1e-10:
                raan = math.acos(max(-1, min(1, n[0] / n_mag)))
                if n[1] < 0:
                    raan = 2 * math.pi - raan
                assert 0 <= raan <= 2 * math.pi

    def test_gps_propagation_all_tles_raan(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            n = [-h[1], h[0], 0.0]
            n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2)
            if n_mag > 1e-10:
                raan = math.acos(max(-1, min(1, n[0] / n_mag)))
                if n[1] < 0:
                    raan = 2 * math.pi - raan
                assert 0 <= raan <= 2 * math.pi

    def test_geo_propagation_all_tles_raan(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            n = [-h[1], h[0], 0.0]
            n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2)
            if n_mag > 1e-10:
                raan = math.acos(max(-1, min(1, n[0] / n_mag)))
                if n[1] < 0:
                    raan = 2 * math.pi - raan
                assert 0 <= raan <= 2 * math.pi

    def test_molniya_propagation_all_tles_mean_motion(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            n = math.sqrt(mu / a ** 3)
            assert n > 0

    def test_gps_propagation_all_tles_mean_motion(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            n = math.sqrt(mu / a ** 3)
            assert n > 0

    def test_geo_propagation_all_tles_mean_motion(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            n = math.sqrt(mu / a ** 3)
            assert n > 0

    def test_molniya_propagation_all_tles_semi_latus_rectum(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            p = h_mag ** 2 / mu
            assert p > 0

    def test_gps_propagation_all_tles_semi_latus_rectum(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            p = h_mag ** 2 / mu
            assert p > 0

    def test_geo_propagation_all_tles_semi_latus_rectum(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            p = h_mag ** 2 / mu
            assert p > 0

    def test_molniya_propagation_all_tles_perigee(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            rp = a * (1 - e)
            assert rp > 0

    def test_gps_propagation_all_tles_perigee(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            rp = a * (1 - e)
            assert rp > 0

    def test_geo_propagation_all_tles_perigee(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            rp = a * (1 - e)
            assert rp > 0

    def test_molniya_propagation_all_tles_apogee(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            ra = a * (1 + e)
            assert ra > 0

    def test_gps_propagation_all_tles_apogee(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            ra = a * (1 + e)
            assert ra > 0

    def test_geo_propagation_all_tles_apogee(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            ra = a * (1 + e)
            assert ra > 0

    def test_molniya_propagation_all_tles_circular_speed(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_circ = math.sqrt(mu / r_mag)
            assert v_circ > 0

    def test_gps_propagation_all_tles_circular_speed(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_circ = math.sqrt(mu / r_mag)
            assert v_circ > 0

    def test_geo_propagation_all_tles_circular_speed(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_circ = math.sqrt(mu / r_mag)
            assert v_circ > 0

    def test_molniya_propagation_all_tles_escape_velocity(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_esc = math.sqrt(2 * mu / r_mag)
            assert v_esc > 0

    def test_gps_propagation_all_tles_escape_velocity(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_esc = math.sqrt(2 * mu / r_mag)
            assert v_esc > 0

    def test_geo_propagation_all_tles_escape_velocity(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_esc = math.sqrt(2 * mu / r_mag)
            assert v_esc > 0

    def test_molniya_propagation_all_tles_bound_orbit(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert energy < 0

    def test_gps_propagation_all_tles_bound_orbit(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert energy < 0

    def test_geo_propagation_all_tles_bound_orbit(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert energy < 0

    def test_molniya_propagation_all_tles_conic_section(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            assert e < 1.0

    def test_gps_propagation_all_tles_conic_section(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            assert e < 1.0

    def test_geo_propagation_all_tles_conic_section(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            assert e < 1.0

    def test_molniya_propagation_all_tles_orbital_plane(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            dot_r = r[0] * h[0] + r[1] * h[1] + r[2] * h[2]
            dot_v = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]
            assert abs(dot_r) < 1e-6
            assert abs(dot_v) < 1e-6

    def test_gps_propagation_all_tles_orbital_plane(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            dot_r = r[0] * h[0] + r[1] * h[1] + r[2] * h[2]
            dot_v = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]
            assert abs(dot_r) < 1e-6
            assert abs(dot_v) < 1e-6

    def test_geo_propagation_all_tles_orbital_plane(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            dot_r = r[0] * h[0] + r[1] * h[1] + r[2] * h[2]
            dot_v = v[0] * h[0] + v[1] * h[1] + v[2] * h[2]
            assert abs(dot_r) < 1e-6
            assert abs(dot_v) < 1e-6

    def test_molniya_propagation_all_tles_earth_centered(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 6378

    def test_gps_propagation_all_tles_earth_centered(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 6378

    def test_geo_propagation_all_tles_earth_centered(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 6378

    def test_molniya_propagation_all_tles_teme_frame(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert math.isfinite(r[2])
            assert math.isfinite(v[2])

    def test_gps_propagation_all_tles_teme_frame(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert math.isfinite(r[2])
            assert math.isfinite(v[2])

    def test_geo_propagation_all_tles_teme_frame(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert math.isfinite(r[2])
            assert math.isfinite(v[2])

    def test_molniya_propagation_all_tles_position_magnitude(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 0

    def test_gps_propagation_all_tles_position_magnitude(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 0

    def test_geo_propagation_all_tles_position_magnitude(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 0

    def test_molniya_propagation_all_tles_velocity_magnitude(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            assert v_mag > 0

    def test_gps_propagation_all_tles_velocity_magnitude(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            assert v_mag > 0

    def test_geo_propagation_all_tles_velocity_magnitude(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            assert v_mag > 0

    def test_molniya_propagation_all_tles_state_vector_validity(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert len(r) == 3
            assert len(v) == 3
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_gps_propagation_all_tles_state_vector_validity(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert len(r) == 3
            assert len(v) == 3
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_geo_propagation_all_tles_state_vector_validity(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert len(r) == 3
            assert len(v) == 3
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_molniya_propagation_all_tles_orbital_elements_validity(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            inc = math.acos(h[2] / h_mag)
            assert a > 0
            assert 0 <= e < 1
            assert 0 <= inc <= math.pi

    def test_gps_propagation_all_tles_orbital_elements_validity(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            inc = math.acos(h[2] / h_mag)
            assert a > 0
            assert 0 <= e < 1
            assert 0 <= inc <= math.pi

    def test_geo_propagation_all_tles_orbital_elements_validity(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            e = math.sqrt(max(0, 1.0 - h_mag ** 2 / (mu * a)))
            inc = math.acos(h[2] / h_mag)
            assert a > 0
            assert 0 <= e < 1
            assert 0 <= inc <= math.pi

    def test_molniya_propagation_all_tles_energy_conservation(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            mu = 398600.4418
            r0, v0 = prop.propagate(0.0)
            r1, v1 = prop.propagate(10.0)
            e0 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) / 2 - mu / math.sqrt(
                r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
            )
            e1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) / 2 - mu / math.sqrt(
                r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            )
            assert abs(e0 - e1) < 1e-3

    def test_gps_propagation_all_tles_energy_conservation(self):
        for tle_data in [GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            mu = 398600.4418
            r0, v0 = prop.propagate(0.0)
            r1, v1 = prop.propagate(10.0)
            e0 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) / 2 - mu / math.sqrt(
                r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
            )
            e1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) / 2 - mu / math.sqrt(
                r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            )
            assert abs(e0 - e1) < 1e-3

    def test_geo_propagation_all_tles_energy_conservation(self):
        for tle_data in [GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            mu = 398600.4418
            r0, v0 = prop.propagate(0.0)
            r1, v1 = prop.propagate(10.0)
            e0 = (v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2) / 2 - mu / math.sqrt(
                r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
            )
            e1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) / 2 - mu / math.sqrt(
                r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            )
            assert abs(e0 - e1) < 1e-3

    def test_molniya_propagation_all_tles_angular_momentum_conservation(self):
        for tle_data in [MOLNIYA_TLE, GPS_TLE, GEO_TLE, TUNDRA_TLE]:
            tle = DeepSpaceTLE(*tle_data)
            prop = SDP4Propagator(tle)
            r0, v0 = prop.propagate(0.0)
            r1, v1 = prop.propagate(10.0)
            h0 = [
                r0[1] * v0[2] - r0[2</longcat_think>
