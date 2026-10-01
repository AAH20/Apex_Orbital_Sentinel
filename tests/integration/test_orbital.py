"""Integration tests for the full space domain pipeline.

Tests the complete flow: TLE parsing → SGP4 propagation → state vectors → conjunction detection.
"""
import math
import pytest
from datetime import datetime, timedelta, timezone

from src.space.tle import TLE, TLEParseError
from src.space.propagator import SGP4Propagator, PropagationError
from src.space.conjunction import ConjunctionDetector, ConjunctionEvent
from src.space.pipeline import SpaceDomainPipeline


# ---------------------------------------------------------------------------
# Test data — real TLEs for ISS and a debris object
# ---------------------------------------------------------------------------

ISS_TLE = TLE(
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   20239.50991228  .00000806  00000-0  19740-3 0  9990",
    line2="2 25544  51.6443 322.2412 0004028  69.9862  25.2906 15.49515014253715",
)

DEBRIS_TLE = TLE(
    name="DEBRIS OBJECT",
    line1="1 40000U 20001A   20239.50000000  .00001000  00000-0  20000-3 0  9991",
    line2="2 40000  51.6400 322.2000 0005000  70.0000  25.3000 15.50000000000001",
)

FAR_DEBRIS_TLE = TLE(
    name="FAR DEBRIS",
    line1="1 40001U 20001B   20239.50000000  .00001000  00000-0  20000-3 0  9992",
    line2="2 40001  90.0000 180.0000 0005000  70.0000  25.3000 15.50000000000002",
)


# ---------------------------------------------------------------------------
# 1. TLE Parsing
# ---------------------------------------------------------------------------

class TestTLEParsing:
    def test_parse_valid_tle(self):
        tle = ISS_TLE
        assert tle.name == "ISS (ZARYA)"
        assert tle.satellite_number == 25544
        assert tle.inclination == pytest.approx(51.6443, abs=1e-4)
        assert tle.raan == pytest.approx(322.2412, abs=1e-4)
        assert tle.eccentricity == pytest.approx(0.0004028, abs=1e-7)
        assert tle.arg_perigee == pytest.approx(69.9862, abs=1e-4)
        assert tle.mean_anomaly == pytest.approx(25.2906, abs=1e-4)
        assert tle.mean_motion == pytest.approx(15.49515014, abs=1e-8)

    def test_parse_tle_from_lines(self):
        tle = TLE.from_lines(
            "1 25544U 98067A   20239.50991228  .00000806  00000-0  19740-3 0  9990",
            "2 25544  51.6443 322.2412 0004028  69.9862  25.2906 15.49515014253715",
            name="ISS",
        )
        assert tle.satellite_number == 25544
        assert tle.inclination == pytest.approx(51.6443, abs=1e-4)

    def test_parse_invalid_tle_raises(self):
        with pytest.raises(TLEParseError):
            TLE.from_lines("invalid line 1", "invalid line 2")

    def test_tle_epoch_parsing(self):
        tle = ISS_TLE
        # Epoch 20239.50991228 → day 239 of 2020 + fraction
        assert tle.epoch_year == 2020
        assert tle.epoch_day == pytest.approx(239.50991228, abs=1e-6)

    def test_tle_period_seconds(self):
        tle = ISS_TLE
        # Period = 86400 / mean_motion
        expected_period = 86400.0 / 15.49515014
        assert tle.period_seconds == pytest.approx(expected_period, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. SGP4 Propagation
# ---------------------------------------------------------------------------

class TestSGP4Propagation:
    def test_propagate_returns_state_vector(self):
        prop = SGP4Propagator()
        sv = prop.propagate(ISS_TLE, ISS_TLE.epoch)
        assert sv is not None
        assert len(sv.position) == 3
        assert len(sv.velocity) == 3

    def test_propagate_at_epoch_matches_initial(self):
        prop = SGP4Propagator()
        sv = prop.propagate(ISS_TLE, ISS_TLE.epoch)
        # At epoch, position should be reasonable (LEO altitude)
        r = math.sqrt(sum(x**2 for x in sv.position))
        assert 6500 < r < 7500  # km

    def test_propagate_future_state(self):
        prop = SGP4Propagator()
        future = ISS_TLE.epoch + timedelta(minutes=30)
        sv = prop.propagate(ISS_TLE, future)
        r = math.sqrt(sum(x**2 for x in sv.position))
        assert 6500 < r < 7500

    def test_propagate_past_state(self):
        prop = SGP4Propagator()
        past = ISS_TLE.epoch - timedelta(minutes=30)
        sv = prop.propagate(ISS_TLE, past)
        r = math.sqrt(sum(x**2 for x in sv.position))
        assert 6500 < r < 7500

    def test_propagate_invalid_tle_raises(self):
        prop = SGP4Propagator()
        bad_tle = TLE(
            name="BAD",
            line1="1 00000U 00000A   20239.50000000  .00000000  00000-0  00000-0 0  9990",
            line2="2 00000   0.0000   0.0000 0.0000000   0.0000   0.0000  0.00000000000000",
        )
        with pytest.raises(PropagationError):
            prop.propagate(bad_tle, bad_tle.epoch)

    def test_propagate_multiple_times(self):
        prop = SGP4Propagator()
        times = [ISS_TLE.epoch + timedelta(minutes=m) for m in range(0, 60, 10)]
        states = prop.propagate_many(ISS_TLE, times)
        assert len(states) == 6
        for sv in states:
            r = math.sqrt(sum(x**2 for x in sv.position))
            assert 6500 < r < 7500


# ---------------------------------------------------------------------------
# 3. Conjunction Detection
# ---------------------------------------------------------------------------

class TestConjunctionDetection:
    def test_detect_conjunction_near_objects(self):
        detector = ConjunctionDetector(threshold_km=10.0)
        events = detector.detect(ISS_TLE, DEBRIS_TLE, start=ISS_TLE.epoch, duration_minutes=90)
        assert isinstance(events, list)
        # These two objects are close in orbital elements — should find at least one event
        assert len(events) >= 1

    def test_no_conjunction_for_distant_objects(self):
        detector = ConjunctionDetector(threshold_km=5.0)
        events = detector.detect(ISS_TLE, FAR_DEBRIS_TLE, start=ISS_TLE.epoch, duration_minutes=90)
        assert len(events) == 0

    def test_conjunction_event_has_required_fields(self):
        detector = ConjunctionDetector(threshold_km=50.0)
        events = detector.detect(ISS_TLE, DEBRIS_TLE, start=ISS_TLE.epoch, duration_minutes=90)
        if events:
            ev = events[0]
            assert hasattr(ev, 'time')
            assert hasattr(ev, 'distance_km')
            assert hasattr(ev, 'sat1_name')
            assert hasattr(ev, 'sat2_name')
            assert ev.distance_km <= 50.0

    def test_conjunction_threshold_affects_results(self):
        detector_loose = ConjunctionDetector(threshold_km=100.0)
        detector_tight = ConjunctionDetector(threshold_km=1.0)
        events_loose = detector_loose.detect(ISS_TLE, DEBRIS_TLE, start=ISS_TLE.epoch, duration_minutes=90)
        events_tight = detector_tight.detect(ISS_TLE, DEBRIS_TLE, start=ISS_TLE.epoch, duration_minutes=90)
        assert len(events_loose) >= len(events_tight)


# ---------------------------------------------------------------------------
# 4. Full Pipeline Integration
# ---------------------------------------------------------------------------

class TestSpaceDomainPipeline:
    def test_full_pipeline_single_object(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.process_single(ISS_TLE)
        assert result is not None
        assert 'state_vector' in result
        assert 'position' in result['state_vector']
        assert 'velocity' in result['state_vector']

    def test_full_pipeline_conjunction_analysis(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.analyze_conjunctions(
            primary=ISS_TLE,
            secondary_objects=[DEBRIS_TLE, FAR_DEBRIS_TLE],
            duration_minutes=90,
        )
        assert 'conjunctions' in result
        assert isinstance(result['conjunctions'], list)
        # Should find conjunction with DEBRIS_TLE but not FAR_DEBRIS_TLE
        conj_sats = [c.sat2_name for c in result['conjunctions']]
        assert "DEBRIS OBJECT" in conj_sats
        assert "FAR DEBRIS" not in conj_sats

    def test_full_pipeline_batch_processing(self):
        pipeline = SpaceDomainPipeline()
        results = pipeline.process_batch([ISS_TLE, DEBRIS_TLE, FAR_DEBRIS_TLE])
        assert len(results) == 3
        for r in results:
            assert 'state_vector' in r
            assert 'position' in r['state_vector']

    def test_full_pipeline_with_maneuver_planning(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.plan_avoidance_maneuver(
            primary=ISS_TLE,
            threat=DEBRIS_TLE,
            duration_minutes=90,
        )
        assert 'maneuver' in result
        assert 'delta_v' in result['maneuver']
        assert 'burn_time' in result['maneuver']
        assert result['maneuver']['delta_v'] > 0

    def test_full_pipeline_end_to_end(self):
        """Complete end-to-end: parse → propagate → detect → plan."""
        pipeline = SpaceDomainPipeline()
        # Step 1: Parse
        tle = TLE.from_lines(
            "1 25544U 98067A   20239.50991228  .00000806  00000-0  19740-3 0  9990",
            "2 25544  51.6443 322.2412 0004028  69.9862  25.2906 15.49515014253715",
            name="ISS",
        )
        # Step 2: Propagate
        sv = pipeline.propagator.propagate(tle, tle.epoch)
        assert sv is not None
        # Step 3: Detect conjunctions
        events = pipeline.detector.detect(tle, DEBRIS_TLE, start=tle.epoch, duration_minutes=90)
        assert isinstance(events, list)
        # Step 4: Plan maneuver if needed
        if events:
            maneuver = pipeline.plan_avoidance_maneuver(tle, DEBRIS_TLE, duration_minutes=90)
            assert maneuver['maneuver']['delta_v'] > 0

    def test_pipeline_handles_empty_secondary_list(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.analyze_conjunctions(
            primary=ISS_TLE,
            secondary_objects=[],
            duration_minutes=90,
        )
        assert result['conjunctions'] == []

    def test_pipeline_propagation_accuracy(self):
        """Verify propagation produces physically reasonable results."""
        pipeline = SpaceDomainPipeline()
        sv = pipeline.propagator.propagate(ISS_TLE, ISS_TLE.epoch)
        # Check velocity magnitude is reasonable for LEO (~7.6 km/s)
        v = math.sqrt(sum(x**2 for x in sv.velocity))
        assert 7.0 < v < 8.0  # km/s
        # Check position magnitude
        r = math.sqrt(sum(x**2 for x in sv.position))
        assert 6500 < r < 7500  # km

    def test_pipeline_orbital_elements_derived(self):
        """Verify pipeline can derive orbital elements from state vectors."""
        pipeline = SpaceDomainPipeline()
        sv = pipeline.propagator.propagate(ISS_TLE, ISS_TLE.epoch)
        elements = pipeline.derive_orbital_elements(sv)
        assert 'semi_major_axis' in elements
        assert 'eccentricity' in elements
        assert 'inclination' in elements
        # Semi-major axis should be reasonable for LEO
        assert 6500 < elements['semi_major_axis'] < 7500
        # Inclination should match TLE
        assert elements['inclination'] == pytest.approx(51.6443, abs=0.1)
