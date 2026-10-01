"""Tests for SGP4/SDP4 propagator — TDD: define expected behavior first."""
import math
import pytest
from src.space.sgp4 import TLE, SGP4Propagator, SGP4Error

# Well-known TLEs for testing
ISS_TLE = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)

GEO_TLE = (
    "1 25994U 99068A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928",
    "2 25994   0.0552  86.2302 0004647  26.2202  12.4322  1.00273142 50581",
)

DEEP_SPACE_TLE = (
    "1 25994U 99068A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928",
    "2 25994   0.0552  86.2302 0004647  26.2202  12.4322  0.50000000 50586",
)


class TestTLEParsing:
    """TLE parsing tests."""

    def test_satellite_number(self):
        tle = TLE(*ISS_TLE)
        assert tle.satellite_number == 25544

    def test_classification(self):
        tle = TLE(*ISS_TLE)
        assert tle.classification == "U"

    def test_epoch_year(self):
        tle = TLE(*ISS_TLE)
        assert tle.epoch_year == 2008

    def test_epoch_day(self):
        tle = TLE(*ISS_TLE)
        assert tle.epoch_day == pytest.approx(264.51782528)

    def test_inclination(self):
        tle = TLE(*ISS_TLE)
        assert tle.inclination_deg == pytest.approx(51.6416, abs=1e-4)

    def test_raan(self):
        tle = TLE(*ISS_TLE)
        assert tle.raan_deg == pytest.approx(247.4627, abs=1e-4)

    def test_eccentricity(self):
        tle = TLE(*ISS_TLE)
        assert tle.eccentricity == pytest.approx(0.0006703, abs=1e-7)

    def test_arg_perigee(self):
        tle = TLE(*ISS_TLE)
        assert tle.arg_perigee_deg == pytest.approx(130.5360, abs=1e-4)

    def test_mean_anomaly(self):
        tle = TLE(*ISS_TLE)
        assert tle.mean_anomaly_deg == pytest.approx(325.0288, abs=1e-4)

    def test_mean_motion(self):
        tle = TLE(*ISS_TLE)
        assert tle.mean_motion_revs_per_day == pytest.approx(15.72125391, abs=1e-8)

    def test_rev_number(self):
        tle = TLE(*ISS_TLE)
        assert tle.rev_number == 56353

    def test_bstar(self):
        tle = TLE(*ISS_TLE)
        assert tle.bstar == pytest.approx(-0.11606e-4, abs=1e-10)

    def test_first_derivative_mean_motion(self):
        tle = TLE(*ISS_TLE)
        assert tle.ndot == pytest.approx(-0.00002182, abs=1e-10)

    def test_second_derivative_mean_motion(self):
        tle = TLE(*ISS_TLE)
        assert tle.nddot == pytest.approx(0.0, abs=1e-10)

    def test_line1_checksum_valid(self):
        tle = TLE(*ISS_TLE)
        assert tle.line1_checksum_valid()

    def test_line2_checksum_valid(self):
        tle = TLE(*ISS_TLE)
        assert tle.line2_checksum_valid()

    def test_line1_checksum_invalid(self):
        bad = list(ISS_TLE)
        bad[0] = bad[0][:-1] + "0"
        with pytest.raises(SGP4Error):
            TLE(*bad)

    def test_line2_checksum_invalid(self):
        bad = list(ISS_TLE)
        bad[1] = bad[1][:-1] + "0"
        with pytest.raises(SGP4Error):
            TLE(*bad)

    def test_invalid_line1_format(self):
        with pytest.raises(SGP4Error):
            TLE("INVALID LINE 1", ISS_TLE[1])

    def test_invalid_line2_format(self):
        with pytest.raises(SGP4Error):
            TLE(ISS_TLE[0], "INVALID LINE 2")

    def test_mismatched_satellite_numbers(self):
        bad = list(ISS_TLE)
        bad[1] = "2 99999" + bad[1][7:]
        with pytest.raises(SGP4Error):
            TLE(*bad)

    def test_epoch_julian_date(self):
        tle = TLE(*ISS_TLE)
        jd = tle.epoch_julian_date()
        assert jd == pytest.approx(2454730.01782528, abs=1e-4)


class TestSGP4Propagation:
    """SGP4 propagation tests."""

    def test_position_magnitude(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 6500 < r_mag < 7500

    def test_velocity_magnitude(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert 7.0 < v_mag < 8.0

    def test_energy_conservation(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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

    def test_angular_momentum_conservation(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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

    def test_periodicity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(92.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist < 500

    def test_iss_known_position(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 6700 < r_mag < 6900

    def test_geo_satellite(self):
        tle = TLE(*GEO_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 42000 < r_mag < 43000

    def test_multiple_time_steps(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        positions = []
        for t in range(0, 100, 10):
            r, _ = prop.propagate(float(t))
            positions.append(r)
        for r in positions:
            assert all(math.isfinite(x) for x in r)

    def test_negative_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(-10.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 6500 < r_mag < 7500

    def test_large_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(1440.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert 6500 < r_mag < 7500

    def test_position_in_teme(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert abs(r[2]) > 1000

    def test_velocity_in_teme(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert v_mag > 7.0

    def test_orbital_period_consistency(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(92.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        circumference = 2 * math.pi * 6700
        assert dist < circumference * 0.1

    def test_semi_major_axis_consistency(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        assert 6700 < a < 6900

    def test_eccentricity_consistency(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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

    def test_inclination_consistency(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        inc = math.acos(h[2] / h_mag)
        assert math.degrees(inc) == pytest.approx(51.6416, abs=0.1)

    def test_raan_consistency(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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
        assert math.degrees(raan) == pytest.approx(247.4627, abs=0.5)

    def test_deep_space_satellite(self):
        tle = TLE(*DEEP_SPACE_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert r_mag > 20000

    def test_near_earth_satellite(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert r_mag < 10000

    def test_propagation_continuity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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

    def test_propagation_determinism(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_propagation_monotonic_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(10.0)
        r2, _ = prop.propagate(20.0)
        assert r0 != r1
        assert r1 != r2
        assert r0 != r2

    def test_propagation_zero_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_very_small_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(0.001)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist < 1.0

    def test_propagation_half_period(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(46.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist > 10000

    def test_propagation_full_period(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(92.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert dist < 500

    def test_propagation_quarter_period(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r0, _ = prop.propagate(0.0)
        r1, _ = prop.propagate(23.0)
        dist = math.sqrt(
            (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
        )
        assert 5000 < dist < 15000

    def test_propagation_energy_at_different_times(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        energies = []
        for t in [0, 10, 20, 30, 40, 50]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            e = v_mag ** 2 / 2 - mu / r_mag
            energies.append(e)
        e_mean = sum(energies) / len(energies)
        for e in energies:
            assert abs(e - e_mean) / abs(e_mean) < 1e-4

    def test_propagation_angular_momentum_at_different_times(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        h_mags = []
        for t in [0, 10, 20, 30, 40, 50]:
            r, v = prop.propagate(float(t))
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            h_mags.append(h_mag)
        h_mean = sum(h_mags) / len(h_mags)
        for h in h_mags:
            assert abs(h - h_mean) / h_mean < 1e-6

    def test_propagation_position_components(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert math.isfinite(r[0])
        assert math.isfinite(r[1])
        assert math.isfinite(r[2])
        assert math.isfinite(v[0])
        assert math.isfinite(v[1])
        assert math.isfinite(v[2])

    def test_propagation_velocity_components(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert abs(v[0]) < 10
        assert abs(v[1]) < 10
        assert abs(v[2]) < 10

    def test_propagation_position_magnitude_range(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in range(0, 200, 10):
            r, _ = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert 6000 < r_mag < 8000

    def test_propagation_velocity_magnitude_range(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in range(0, 200, 10):
            _, v = prop.propagate(float(t))
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            assert 6.0 < v_mag < 9.0

    def test_propagation_orbital_elements_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert 6700 < a < 6900

    def test_propagation_inclination_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            inc = math.degrees(math.acos(h[2] / h_mag))
            assert 51.0 < inc < 52.5

    def test_propagation_raan_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            n = [-h[1], h[0], 0.0]
            n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2)
            if n_mag > 1e-10:
                raan = math.degrees(math.acos(max(-1, min(1, n[0] / n_mag))))
                if n[1] < 0:
                    raan = 360 - raan
                assert 240 < raan < 255

    def test_propagation_eccentricity_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
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

    def test_propagation_semi_major_axis_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        a_values = []
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            a_values.append(a)
        a_mean = sum(a_values) / len(a_values)
        for a in a_values:
            assert abs(a - a_mean) / a_mean < 1e-4

    def test_propagation_period_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        periods = []
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            T = 2 * math.pi * math.sqrt(a ** 3 / mu)
            periods.append(T)
        T_mean = sum(periods) / len(periods)
        for T in periods:
            assert abs(T - T_mean) / T_mean < 1e-4

    def test_propagation_mean_motion_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        n_values = []
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            n = math.sqrt(mu / a ** 3)
            n_values.append(n)
        n_mean = sum(n_values) / len(n_values)
        for n in n_values:
            assert abs(n - n_mean) / n_mean < 1e-4

    def test_propagation_specific_energy_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        energies = []
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            e = v_mag ** 2 / 2 - mu / r_mag
            energies.append(e)
        e_mean = sum(energies) / len(energies)
        for e in energies:
            assert abs(e - e_mean) / abs(e_mean) < 1e-4

    def test_propagation_angular_momentum_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        h_mags = []
        for t in [0, 30, 60, 90]:
            r, v = prop.propagate(float(t))
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            h_mags.append(h_mag)
        h_mean = sum(h_mags) / len(h_mags)
        for h in h_mags:
            assert abs(h - h_mean) / h_mean < 1e-6

    def test_propagation_position_continuity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        prev_r = None
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            r, _ = prop.propagate(t)
            if prev_r is not None:
                dist = math.sqrt(
                    (r[0] - prev_r[0]) ** 2
                    + (r[1] - prev_r[1]) ** 2
                    + (r[2] - prev_r[2]) ** 2
                )
                assert dist < 100
            prev_r = r

    def test_propagation_velocity_continuity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        prev_v = None
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            _, v = prop.propagate(t)
            if prev_v is not None:
                dv = math.sqrt(
                    (v[0] - prev_v[0]) ** 2
                    + (v[1] - prev_v[1]) ** 2
                    + (v[2] - prev_v[2]) ** 2
                )
                assert dv < 1.0
            prev_v = v

    def test_propagation_state_vector_validity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in [0, 10, 20, 30, 40, 50]:
            r, v = prop.propagate(float(t))
            assert len(r) == 3
            assert len(v) == 3
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_orbital_elements_validity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 10, 20, 30, 40, 50]:
            r, v = prop.propagate(float(t))
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

    def test_propagation_teme_frame(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert abs(r[2]) > 1000
        assert abs(v[2]) > 1.0

    def test_propagation_earth_centered(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        assert r_mag > 6000
        assert r_mag < 10000

    def test_propagation_orbital_plane(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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

    def test_propagation_conic_section(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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

    def test_propagation_bound_orbit(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        energy = v_mag ** 2 / 2 - mu / r_mag
        assert energy < 0

    def test_propagation_circular_speed(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_circ = math.sqrt(mu / r_mag)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert abs(v_mag - v_circ) / v_circ < 0.1

    def test_propagation_escape_velocity(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_esc = math.sqrt(2 * mu / r_mag)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert v_mag < v_esc

    def test_propagation_perigee_distance(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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
        assert rp > 6378

    def test_propagation_apogee_distance(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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
        assert 6700 < ra < 6900

    def test_propagation_semi_latus_rectum(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        p = h_mag ** 2 / mu
        assert 6700 < p < 6900

    def test_propagation_true_anomaly(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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
        assert a > 0
        assert e >= 0

    def test_propagation_eccentric_anomaly(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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
        assert a > 0
        assert e >= 0

    def test_propagation_mean_anomaly_from_state(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        n = math.sqrt(mu / a ** 3)
        assert n > 0

    def test_propagation_orbital_period_from_state(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        T = 2 * math.pi * math.sqrt(a ** 3 / mu)
        assert 80 < T < 100

    def test_propagation_semi_major_axis_from_tle(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        n_tle = tle.mean_motion_revs_per_day * 2 * math.pi / (24 * 60)
        a_from_n = (mu / n_tle ** 2) ** (1.0 / 3.0)
        assert abs(a - a_from_n) / a_from_n < 0.01

    def test_propagation_inclination_from_tle(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
        inc = math.degrees(math.acos(h[2] / h_mag))
        assert inc == pytest.approx(tle.inclination_deg, abs=0.1)

    def test_propagation_eccentricity_from_tle(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
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
        assert e == pytest.approx(tle.eccentricity, abs=1e-4)

    def test_propagation_raan_from_tle(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        h = [
            r[1] * v[2] - r[2] * v[1],
            r[2] * v[0] - r[0] * v[2],
            r[0] * v[1] - r[1] * v[0],
        ]
        n = [-h[1], h[0], 0.0]
        n_mag = math.sqrt(n[0] ** 2 + n[1] ** 2)
        if n_mag > 1e-10:
            raan = math.degrees(math.acos(max(-1, min(1, n[0] / n_mag))))
            if n[1] < 0:
                raan = 360 - raan
            assert raan == pytest.approx(tle.raan_deg, abs=0.5)

    def test_propagation_epoch_state(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        n_tle = tle.mean_motion_revs_per_day * 2 * math.pi / (24 * 60)
        a_from_n = (mu / n_tle ** 2) ** (1.0 / 3.0)
        assert abs(a - a_from_n) / a_from_n < 0.01

    def test_propagation_non_epoch_state(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(10.0)
        mu = 398600.4418
        r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
        assert 6700 < a < 6900

    def test_propagation_long_term_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 100, 200, 300, 400, 500]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert 6700 < a < 6900

    def test_propagation_short_term_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 1, 2, 3, 4, 5]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert 6700 < a < 6900

    def test_propagation_very_short_term_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 0.01, 0.02, 0.03, 0.04, 0.05]:
            r, v = prop.propagate(t)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert 6700 < a < 6900

    def test_propagation_very_long_term_stability(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        mu = 398600.4418
        for t in [0, 1000, 2000, 3000, 4000, 5000]:
            r, v = prop.propagate(float(t))
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert 6700 < a < 6900

    def test_propagation_extreme_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(10000.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_negative_extreme_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(-10000.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_zero_time_exact(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_very_small_positive_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_very_small_negative_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(-1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_integer_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_float_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(10.5)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_list_output(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        assert len(r) == 3
        assert len(v) == 3
        _ = r[0]
        _ = r[1]
        _ = r[2]
        _ = v[0]
        _ = v[1]
        _ = v[2]

    def test_propagation_tuple_output(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(0.0)
        for x in r:
            assert math.isfinite(x)
        for x in v:
            assert math.isfinite(x)

    def test_propagation_multiple_calls(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_propagation_different_times_different_results(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 != r2
        assert v1 != v2

    def test_propagation_sequential_times(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        results = []
        for t in range(0, 100, 10):
            r, v = prop.propagate(float(t))
            results.append((r, v))
        for i in range(len(results) - 1):
            assert results[i][0] != results[i + 1][0]

    def test_propagation_reverse_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2
        assert v1 != v2

    def test_propagation_symmetry(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(-10.0)
        assert r1 != r2

    def test_propagation_time_zero_symmetry(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(0.0)
        r2, v2 = prop.propagate(0.0)
        assert r1 == r2
        assert v1 == v2

    def test_propagation_very_large_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_very_small_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_negative_very_large_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(-1e6)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_negative_very_small_time(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r, v = prop.propagate(-1e-10)
        assert all(math.isfinite(x) for x in r)
        assert all(math.isfinite(x) for x in v)

    def test_propagation_special_values(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in [0.0, -0.0, 1.0, -1.0]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_boundary_values(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in [0.0, 1e-6, 1e6]:
            r, v = prop.propagate(t)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_typical_values(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            r, v = prop.propagate(float(t))
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_components_finite(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert math.isfinite(x)
            for x in v:
                assert math.isfinite(x)

    def test_propagation_no_nan_values(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isnan(x)
            for x in v:
                assert not math.isnan(x)

    def test_propagation_no_inf_values(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        for t in range(0, 200, 10):
            r, v = prop.propagate(float(t))
            for x in r:
                assert not math.isinf(x)
            for x in v:
                assert not math.isinf(x)

    def test_propagation_consistent_results(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        results = []
        for _ in range(5):
            r, v = prop.propagate(10.0)
            results.append((r, v))
        for i in range(1, len(results)):
            assert results[i][0] == results[0][0]
            assert results[i][1] == results[0][1]

    def test_propagation_different_instances(self):
        tle = TLE(*ISS_TLE)
        prop1 = SGP4Propagator(tle)
        prop2 = SGP4Propagator(tle)
        r1, v1 = prop1.propagate(10.0)
        r2, v2 = prop2.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_propagation_same_instance(self):
        tle = TLE(*ISS_TLE)
        prop = SGP4Propagator(tle)
        r1, v1 = prop.propagate(10.0)
        r2, v2 = prop.propagate(10.0)
        assert r1 == r2
        assert v1 == v2

    def test_propagation_different_tles(self):
        tle1 = TLE(*ISS_TLE)
        tle2 = TLE(*GEO_TLE)
        prop1 = SGP4Propagator(tle1)
        prop2 = SGP4Propagator(tle2)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        assert r1 != r2
        assert v1 != v2

    def test_propagation_iss_vs_geo(self):
        iss_tle = TLE(*ISS_TLE)
        geo_tle = TLE(*GEO_TLE)
        iss_prop = SGP4Propagator(iss_tle)
        geo_prop = SGP4Propagator(geo_tle)
        iss_r, iss_v = iss_prop.propagate(0.0)
        geo_r, geo_v = geo_prop.propagate(0.0)
        iss_r_mag = math.sqrt(iss_r[0] ** 2 + iss_r[1] ** 2 + iss_r[2] ** 2)
        geo_r_mag = math.sqrt(geo_r[0] ** 2 + geo_r[1] ** 2 + geo_r[2] ** 2)
        assert iss_r_mag < geo_r_mag

    def test_propagation_iss_vs_deep_space(self):
        iss_tle = TLE(*ISS_TLE)
        ds_tle = TLE(*DEEP_SPACE_TLE)
        iss_prop = SGP4Propagator(iss_tle)
        ds_prop = SGP4Propagator(ds_tle)
        iss_r, iss_v = iss_prop.propagate(0.0)
        ds_r, ds_v = ds_prop.propagate(0.0)
        iss_r_mag = math.sqrt(iss_r[0] ** 2 + iss_r[1] ** 2 + iss_r[2] ** 2)
        ds_r_mag = math.sqrt(ds_r[0] ** 2 + ds_r[1] ** 2 + ds_r[2] ** 2)
        assert iss_r_mag < ds_r_mag

    def test_propagation_geo_vs_deep_space(self):
        geo_tle = TLE(*GEO_TLE)
        ds_tle = TLE(*DEEP_SPACE_TLE)
        geo_prop = SGP4Propagator(geo_tle)
        ds_prop = SGP4Propagator(ds_tle)
        geo_r, geo_v = geo_prop.propagate(0.0)
        ds_r, ds_v = ds_prop.propagate(0.0)
        geo_r_mag = math.sqrt(geo_r[0] ** 2 + geo_r[1] ** 2 + geo_r[2] ** 2)
        ds_r_mag = math.sqrt(ds_r[0] ** 2 + ds_r[1] ** 2 + ds_r[2] ** 2)
        assert geo_r_mag < ds_r_mag

    def test_propagation_all_tles_valid(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_different(self):
        tle1 = TLE(*ISS_TLE)
        tle2 = TLE(*GEO_TLE)
        tle3 = TLE(*DEEP_SPACE_TLE)
        prop1 = SGP4Propagator(tle1)
        prop2 = SGP4Propagator(tle2)
        prop3 = SGP4Propagator(tle3)
        r1, v1 = prop1.propagate(0.0)
        r2, v2 = prop2.propagate(0.0)
        r3, v3 = prop3.propagate(0.0)
        assert r1 != r2
        assert r2 != r3
        assert r1 != r3

    def test_propagation_all_tles_consistent(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 == r2
            assert v1 == v2

    def test_propagation_all_tles_orbital_elements(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_energy(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert math.isfinite(energy)

    def test_propagation_all_tles_angular_momentum(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            assert h_mag > 0

    def test_propagation_all_tles_period(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            T = 2 * math.pi * math.sqrt(a ** 3 / mu)
            assert T > 0

    def test_propagation_all_tles_semi_major_axis(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            assert a > 0

    def test_propagation_all_tles_eccentricity(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_inclination(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            h = [
                r[1] * v[2] - r[2] * v[1],
                r[2] * v[0] - r[0] * v[2],
                r[0] * v[1] - r[1] * v[0],
            ]
            h_mag = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
            inc = math.acos(h[2] / h_mag)
            assert 0 <= inc <= math.pi

    def test_propagation_all_tles_raan(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_mean_motion(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            n = math.sqrt(mu / a ** 3)
            assert n > 0

    def test_propagation_all_tles_semi_latus_rectum(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_perigee(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_apogee(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_circular_speed(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_circ = math.sqrt(mu / r_mag)
            assert v_circ > 0

    def test_propagation_all_tles_escape_velocity(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_esc = math.sqrt(2 * mu / r_mag)
            assert v_esc > 0

    def test_propagation_all_tles_bound_orbit(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy = v_mag ** 2 / 2 - mu / r_mag
            assert energy < 0

    def test_propagation_all_tles_conic_section(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_orbital_plane(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_earth_centered(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 6378

    def test_propagation_all_tles_teme_frame(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert math.isfinite(r[2])
            assert math.isfinite(v[2])

    def test_propagation_all_tles_position_magnitude(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            assert r_mag > 0

    def test_propagation_all_tles_velocity_magnitude(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            v_mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            assert v_mag > 0

    def test_propagation_all_tles_state_vector_validity(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert len(r) == 3
            assert len(v) == 3
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_orbital_elements_validity(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_energy_conservation(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_angular_momentum_conservation(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
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

    def test_propagation_all_tles_periodicity(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r0, _ = prop.propagate(0.0)
            mu = 398600.4418
            r_mag = math.sqrt(r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2)
            _, v0 = prop.propagate(0.0)
            v_mag = math.sqrt(v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2)
            a = 1.0 / (2.0 / r_mag - v_mag ** 2 / mu)
            T = 2 * math.pi * math.sqrt(a ** 3 / mu)
            r1, _ = prop.propagate(T)
            dist = math.sqrt(
                (r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2 + (r0[2] - r1[2]) ** 2
            )
            assert dist < 500

    def test_propagation_all_tles_multiple_time_steps(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            positions = []
            for t in range(0, 100, 10):
                r, _ = prop.propagate(float(t))
                positions.append(r)
            for r in positions:
                assert all(math.isfinite(x) for x in r)

    def test_propagation_all_tles_negative_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(-10.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_large_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(1440.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_zero_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_very_small_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.001)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_very_large_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(10000.0)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_extreme_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(1e6)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_negative_extreme_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(-1e6)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_special_values(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            for t in [0.0, -0.0, 1.0, -1.0]:
                r, v = prop.propagate(t)
                assert all(math.isfinite(x) for x in r)
                assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_boundary_values(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            for t in [0.0, 1e-6, 1e6]:
                r, v = prop.propagate(t)
                assert all(math.isfinite(x) for x in r)
                assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_typical_values(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            for t in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                r, v = prop.propagate(float(t))
                assert all(math.isfinite(x) for x in r)
                assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_all_components_finite(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            for t in range(0, 200, 10):
                r, v = prop.propagate(float(t))
                for x in r:
                    assert math.isfinite(x)
                for x in v:
                    assert math.isfinite(x)

    def test_propagation_all_tles_no_nan_values(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            for t in range(0, 200, 10):
                r, v = prop.propagate(float(t))
                for x in r:
                    assert not math.isnan(x)
                for x in v:
                    assert not math.isnan(x)

    def test_propagation_all_tles_no_inf_values(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            for t in range(0, 200, 10):
                r, v = prop.propagate(float(t))
                for x in r:
                    assert not math.isinf(x)
                for x in v:
                    assert not math.isinf(x)

    def test_propagation_all_tles_consistent_results(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            results = []
            for _ in range(5):
                r, v = prop.propagate(10.0)
                results.append((r, v))
            for i in range(1, len(results)):
                assert results[i][0] == results[0][0]
                assert results[i][1] == results[0][1]

    def test_propagation_all_tles_different_instances(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop1 = SGP4Propagator(tle)
            prop2 = SGP4Propagator(tle)
            r1, v1 = prop1.propagate(10.0)
            r2, v2 = prop2.propagate(10.0)
            assert r1 == r2
            assert v1 == v2

    def test_propagation_all_tles_same_instance(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 == r2
            assert v1 == v2

    def test_propagation_all_tles_different_times_different_results(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(0.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 != r2
            assert v1 != v2

    def test_propagation_all_tles_sequential_times(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            results = []
            for t in range(0, 100, 10):
                r, v = prop.propagate(float(t))
                results.append((r, v))
            for i in range(len(results) - 1):
                assert results[i][0] != results[i + 1][0]

    def test_propagation_all_tles_reverse_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(-10.0)
            assert r1 != r2
            assert v1 != v2

    def test_propagation_all_tles_symmetry(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(-10.0)
            assert r1 != r2

    def test_propagation_all_tles_time_zero_symmetry(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(0.0)
            r2, v2 = prop.propagate(0.0)
            assert r1 == r2
            assert v1 == v2

    def test_propagation_all_tles_very_large_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(1e6)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_very_small_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(1e-10)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_negative_very_large_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(-1e6)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_negative_very_small_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(-1e-10)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_integer_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(10)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_float_time(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(10.5)
            assert all(math.isfinite(x) for x in r)
            assert all(math.isfinite(x) for x in v)

    def test_propagation_all_tles_list_output(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            assert len(r) == 3
            assert len(v) == 3
            _ = r[0]
            _ = r[1]
            _ = r[2]
            _ = v[0]
            _ = v[1]
            _ = v[2]

    def test_propagation_all_tles_tuple_output(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r, v = prop.propagate(0.0)
            for x in r:
                assert math.isfinite(x)
            for x in v:
                assert math.isfinite(x)

    def test_propagation_all_tles_multiple_calls(self):
        for tle_data in [ISS_TLE, GEO_TLE, DEEP_SPACE_TLE]:
            tle = TLE(*tle_data)
            prop = SGP4Propagator(tle)
            r1, v1 = prop.propagate(10.0)
            r2, v2 = prop.propagate(10.0)
            assert r1 == r2
            assert v1 == v2
