"""Unit tests for space weather module."""
import pytest
from src.space.space_weather import (
    SolarFlare,
    classify_flare,
    detect_flares,
    GeomagneticStorm,
    storm_severity,
    predict_geomagnetic_storm,
    SatelliteImpact,
    assess_satellite_impact,
)


# ── Solar Flare Classification ──────────────────────────────────────────

class TestClassifyFlare:
    """Tests for solar flare classification by X-ray flux."""

    def test_class_a_flare(self):
        assert classify_flare(1e-8) == "A"

    def test_class_b_flare(self):
        assert classify_flare(1e-7) == "B"

    def test_class_c_flare(self):
        assert classify_flare(1e-6) == "C"

    def test_class_m_flare(self):
        assert classify_flare(1e-5) == "M"

    def test_class_x_flare(self):
        assert classify_flare(1e-4) == "X"

    def test_boundary_a_to_b(self):
        assert classify_flare(9.9e-8) == "A"
        assert classify_flare(1e-7) == "B"

    def test_boundary_c_to_m(self):
        assert classify_flare(9.9e-6) == "C"
        assert classify_flare(1e-5) == "M"

    def test_boundary_m_to_x(self):
        assert classify_flare(9.9e-5) == "M"
        assert classify_flare(1e-4) == "X"

    def test_above_x_class(self):
        assert classify_flare(1e-3) == "X"

    def test_zero_flux(self):
        assert classify_flare(0.0) == "A"

    def test_negative_flux_raises(self):
        with pytest.raises(ValueError):
            classify_flare(-1e-6)


# ── Solar Flare Detection ───────────────────────────────────────────────

class TestDetectFlares:
    """Tests for detecting flare events from X-ray flux time series."""

    def test_detects_single_flare(self):
        flux = [1e-8, 1e-8, 5e-5, 1e-8, 1e-8]
        flares = detect_flares(flux)
        assert len(flares) == 1
        assert flares[0].flare_class == "M"

    def test_detects_multiple_flares(self):
        flux = [1e-8, 5e-5, 1e-8, 2e-4, 1e-8]
        flares = detect_flares(flux)
        assert len(flares) == 2
        assert flares[0].flare_class == "M"
        assert flares[1].flare_class == "X"

    def test_no_flare_in_quiet_data(self):
        flux = [1e-8, 1.1e-8, 0.9e-8, 1e-8]
        flares = detect_flares(flux)
        assert len(flares) == 0

    def test_flare_has_peak_flux(self):
        flux = [1e-8, 3e-5, 1e-8]
        flares = detect_flares(flux)
        assert flares[0].peak_flux == 3e-5

    def test_flare_has_start_index(self):
        flux = [1e-8, 1e-8, 5e-6, 1e-8]
        flares = detect_flares(flux)
        assert flares[0].start_index == 2

    def test_empty_flux_returns_empty(self):
        assert detect_flares([]) == []


# ── Geomagnetic Storm Severity ──────────────────────────────────────────

class TestStormSeverity:
    """Tests for geomagnetic storm severity classification by Kp index."""

    def test_kp0_no_storm(self):
        assert storm_severity(0) == "None"

    def test_kp4_quiet(self):
        assert storm_severity(4) == "Quiet"

    def test_kp5_g1_minor(self):
        assert storm_severity(5) == "G1-Minor"

    def test_kp6_g2_moderate(self):
        assert storm_severity(6) == "G2-Moderate"

    def test_kp7_g3_strong(self):
        assert storm_severity(7) == "G3-Strong"

    def test_kp8_g4_severe(self):
        assert storm_severity(8) == "G4-Severe"

    def test_kp9_g5_extreme(self):
        assert storm_severity(9) == "G5-Extreme"

    def test_kp_above_9(self):
        assert storm_severity(10) == "G5-Extreme"

    def test_negative_kp_raises(self):
        with pytest.raises(ValueError):
            storm_severity(-1)


# ── Geomagnetic Storm Prediction ────────────────────────────────────────

class TestPredictGeomagneticStorm:
    """Tests for predicting geomagnetic storms from solar wind data."""

    def test_high_kp_predicts_storm(self):
        storm = predict_geomagnetic_storm(kp_index=7, dst_index=-80)
        assert storm.severity == "G3-Strong"

    def test_low_kp_no_storm(self):
        storm = predict_geomagnetic_storm(kp_index=2, dst_index=-10)
        assert storm.severity == "None"

    def test_storm_has_kp_and_dst(self):
        storm = predict_geomagnetic_storm(kp_index=6, dst_index=-50)
        assert storm.kp_index == 6
        assert storm.dst_index == -50

    def test_severe_dst_escalates_severity(self):
        storm = predict_geomagnetic_storm(kp_index=5, dst_index=-150)
        assert storm.severity in ("G2-Moderate", "G3-Strong", "G4-Severe")

    def test_positive_dst_no_storm(self):
        storm = predict_geomagnetic_storm(kp_index=2, dst_index=20)
        assert storm.severity == "None"


# ── Satellite Impact Assessment ─────────────────────────────────────────

class TestAssessSatelliteImpact:
    """Tests for satellite impact assessment from space weather."""

    def test_x_flare_high_altitude_radiation_risk(self):
        impact = assess_satellite_impact("X", "G5-Extreme", 35786)
        assert impact.radiation_risk == "High"

    def test_m_flare_low_altitude_drag_increase(self):
        impact = assess_satellite_impact("M", "G1-Minor", 400)
        assert impact.drag_increase == "Moderate"

    def test_no_flare_no_impact(self):
        impact = assess_satellite_impact("A", "None", 500)
        assert impact.radiation_risk == "Low"
        assert impact.drag_increase == "Low"
        assert impact.comm_disruption == "Low"

    def test_g5_storm_comm_disruption(self):
        impact = assess_satellite_impact("C", "G5-Extreme", 20000)
        assert impact.comm_disruption == "High"

    def test_impact_has_overall_risk(self):
        impact = assess_satellite_impact("X", "G4-Severe", 1000)
        assert impact.overall_risk in ("Low", "Moderate", "High", "Critical")

    def test_invalid_flare_class_raises(self):
        with pytest.raises(ValueError):
            assess_satellite_impact("Z", "None", 500)

    def test_invalid_altitude_raises(self):
        with pytest.raises(ValueError):
            assess_satellite_impact("C", "None", -100)
