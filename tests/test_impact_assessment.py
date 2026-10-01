"""Unit tests for satellite impact assessment, radiation modeling, and mission risk."""
import math
import pytest

from src.space.impact_assessment import (
    EARTH_MU_KM3_S2,
    EARTH_RADIUS_KM,
    ImpactAssessment,
    RadiationEnvironment,
    SatelliteSpec,
    assess_charging_impact,
    assess_comm_impact,
    assess_drag_impact,
    assess_radiation_impact,
    assess_satellite_mission_risk,
    assess_thermal_impact,
    atmospheric_density,
    compute_impact_score,
    cumulative_mission_risk,
    displacement_damage_dose,
    dose_behind_shielding,
    drag_acceleration,
    estimate_dose,
    galactic_cosmic_ray_flux,
    mission_lifetime_risk,
    mission_risk_score,
    orbit_decay_rate,
    orbit_regime,
    orbital_velocity,
    recommend_mitigations,
    risk_category,
    seu_probability,
    seu_rate,
    solar_proton_event_flux,
    total_radiation_environment,
    trapped_electron_flux,
    trapped_proton_flux,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _satellite(name="TEST-SAT", alt=500.0, inc=90.0, shielding=1.0, duration=365.0):
    return SatelliteSpec(
        name=name,
        altitude_km=alt,
        inclination_deg=inc,
        shielding_g_cm2=shielding,
        mission_duration_days=duration,
    )


def _env(flare="A", storm="None", solar="minimum"):
    return RadiationEnvironment(
        flare_class=flare,
        storm_severity=storm,
        solar_activity=solar,
    )


def _impact(radiation="Low", thermal="Low", drag="Low", comm="Low", charging="Low"):
    return ImpactAssessment(
        radiation=radiation,
        thermal=thermal,
        drag=drag,
        comm=comm,
        charging=charging,
        overall="Low",
        dose_rate_rad_s=0.0,
        total_dose_rad=0.0,
        seu_rate_per_s=0.0,
    )


# ── Trapped Proton Flux ─────────────────────────────────────────────────────

class TestTrappedProtonFlux:
    def test_peak_at_500km_polar(self):
        flux = trapped_proton_flux(500, 90)
        assert flux == pytest.approx(1e4, rel=0.01)

    def test_zero_at_equator(self):
        assert trapped_proton_flux(500, 0) == 0.0

    def test_decreases_above_peak(self):
        assert trapped_proton_flux(1000, 90) < trapped_proton_flux(500, 90)

    def test_decreases_below_peak(self):
        assert trapped_proton_flux(300, 90) < trapped_proton_flux(500, 90)

    def test_negligible_at_geo(self):
        assert trapped_proton_flux(35786, 90) < 1.0

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            trapped_proton_flux(-100, 45)

    def test_invalid_inclination_raises(self):
        with pytest.raises(ValueError):
            trapped_proton_flux(500, 200)


# ── Trapped Electron Flux ───────────────────────────────────────────────────

class TestTrappedElectronFlux:
    def test_peak_at_20000km(self):
        flux = trapped_electron_flux(20000, 90)
        assert flux == pytest.approx(1e7, rel=0.01)

    def test_zero_below_inner_edge(self):
        assert trapped_electron_flux(500, 90) == 0.0

    def test_zero_at_equator(self):
        assert trapped_electron_flux(20000, 0) == 0.0

    def test_lower_at_geo_than_peak(self):
        assert trapped_electron_flux(35786, 90) < trapped_electron_flux(20000, 90)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            trapped_electron_flux(-1, 45)


# ── Solar Proton Event Flux ─────────────────────────────────────────────────

class TestSolarProtonEventFlux:
    def test_class_a(self):
        assert solar_proton_event_flux("A") == 1.0

    def test_class_c(self):
        assert solar_proton_event_flux("C") == 100.0

    def test_class_x(self):
        assert solar_proton_event_flux("X") == 10000.0

    def test_monotonic_increasing(self):
        classes = ["A", "B", "C", "M", "X"]
        fluxes = [solar_proton_event_flux(c) for c in classes]
        assert fluxes == sorted(fluxes)

    def test_invalid_class_raises(self):
        with pytest.raises(ValueError):
            solar_proton_event_flux("Z")


# ── Galactic Cosmic Ray Flux ────────────────────────────────────────────────

class TestGalacticCosmicRayFlux:
    def test_solar_minimum_higher(self):
        assert galactic_cosmic_ray_flux("minimum") > galactic_cosmic_ray_flux("maximum")

    def test_solar_minimum_value(self):
        assert galactic_cosmic_ray_flux("minimum") == pytest.approx(4.0)

    def test_solar_maximum_value(self):
        assert galactic_cosmic_ray_flux("maximum") == pytest.approx(1.5)

    def test_invalid_activity_raises(self):
        with pytest.raises(ValueError):
            galactic_cosmic_ray_flux("moderate")


# ── Dose Behind Shielding ───────────────────────────────────────────────────

class TestDoseBehindShielding:
    def test_zero_shielding_no_attenuation(self):
        assert dose_behind_shielding(1e4, 0.0) == pytest.approx(1e4 * 1e-8)

    def test_increasing_shielding_reduces_dose(self):
        d1 = dose_behind_shielding(1e4, 1.0)
        d2 = dose_behind_shielding(1e4, 5.0)
        assert d2 < d1

    def test_heavy_shielding_approaches_zero(self):
        assert dose_behind_shielding(1e4, 100.0) < 1e-6

    def test_negative_flux_raises(self):
        with pytest.raises(ValueError):
            dose_behind_shielding(-1, 1.0)

    def test_negative_shielding_raises(self):
        with pytest.raises(ValueError):
            dose_behind_shielding(1e4, -1.0)


# ── Mission Dose ────────────────────────────────────────────────────────────

class TestMissionDose:
    def test_zero_duration_zero_dose(self):
        from src.space.impact_assessment import mission_dose
        assert mission_dose(1e-6, 0) == 0.0

    def test_linear_with_duration(self):
        from src.space.impact_assessment import mission_dose
        d1 = mission_dose(1e-6, 10)
        d2 = mission_dose(1e-6, 20)
        assert d2 == pytest.approx(2 * d1)

    def test_known_value(self):
        from src.space.impact_assessment import mission_dose
        # 1e-6 rad/s * 86400 s/day * 1 day = 0.0864 rad
        assert mission_dose(1e-6, 1) == pytest.approx(0.0864)

    def test_negative_inputs_raise(self):
        from src.space.impact_assessment import mission_dose
        with pytest.raises(ValueError):
            mission_dose(-1, 10)
        with pytest.raises(ValueError):
            mission_dose(1e-6, -1)


# ── SEU Rate ────────────────────────────────────────────────────────────────

class TestSeuRate:
    def test_zero_flux_zero_rate(self):
        assert seu_rate(0, 1e-12) == 0.0

    def test_proportional_to_flux(self):
        r1 = seu_rate(1e3, 1e-12)
        r2 = seu_rate(2e3, 1e-12)
        assert r2 == pytest.approx(2 * r1)

    def test_proportional_to_cross_section(self):
        r1 = seu_rate(1e3, 1e-12)
        r2 = seu_rate(1e3, 1e-11)
        assert r2 == pytest.approx(10 * r1)

    def test_negative_flux_raises(self):
        with pytest.raises(ValueError):
            seu_rate(-1, 1e-12)

    def test_zero_cross_section_raises(self):
        with pytest.raises(ValueError):
            seu_rate(1e3, 0)


# ── Displacement Damage ─────────────────────────────────────────────────────

class TestDisplacementDamage:
    def test_zero_fluence_zero_damage(self):
        assert displacement_damage_dose(0) == 0.0

    def test_proportional_to_fluence(self):
        d1 = displacement_damage_dose(1e10)
        d2 = displacement_damage_dose(2e10)
        assert d2 == pytest.approx(2 * d1)

    def test_negative_fluence_raises(self):
        with pytest.raises(ValueError):
            displacement_damage_dose(-1)


# ── Thermal Impact ──────────────────────────────────────────────────────────

class TestAssessThermalImpact:
    def test_x_flare_high(self):
        assert assess_thermal_impact("X", 500) == "High"

    def test_m_flare_moderate(self):
        assert assess_thermal_impact("M", 500) == "Moderate"

    def test_c_flare_low(self):
        assert assess_thermal_impact("C", 500) == "Low"

    def test_a_flare_low(self):
        assert assess_thermal_impact("A", 500) == "Low"

    def test_invalid_flare_raises(self):
        with pytest.raises(ValueError):
            assess_thermal_impact("Z", 500)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            assess_thermal_impact("X", -100)


# ── Drag Impact ─────────────────────────────────────────────────────────────

class TestAssessDragImpact:
    def test_g5_low_altitude_high(self):
        assert assess_drag_impact("G5-Extreme", 400, 100) == "High"

    def test_g5_high_altitude_moderate(self):
        assert assess_drag_impact("G5-Extreme", 1000, 100) == "Moderate"

    def test_g1_low_altitude_low(self):
        assert assess_drag_impact("G1-Minor", 400, 100) == "Low"

    def test_none_storm_low(self):
        assert assess_drag_impact("None", 400, 100) == "Low"

    def test_g3_moderate(self):
        assert assess_drag_impact("G3-Strong", 500, 100) == "Moderate"

    def test_low_ballistic_coefficient_worse(self):
        assert assess_drag_impact("G5-Extreme", 400, 10) == "High"

    def test_invalid_storm_raises(self):
        with pytest.raises(ValueError):
            assess_drag_impact("G9", 500, 100)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            assess_drag_impact("G1-Minor", -100, 100)

    def test_zero_ballistic_coefficient_raises(self):
        with pytest.raises(ValueError):
            assess_drag_impact("G1-Minor", 500, 0)


# ── Comm Impact ─────────────────────────────────────────────────────────────

class TestAssessCommImpact:
    def test_ka_g5_high(self):
        assert assess_comm_impact("G5-Extreme", "Ka") == "High"

    def test_l_g5_moderate(self):
        assert assess_comm_impact("G5-Extreme", "L") == "Moderate"

    def test_l_g1_low(self):
        assert assess_comm_impact("G1-Minor", "L") == "Low"

    def test_vhf_g5_low(self):
        assert assess_comm_impact("G5-Extreme", "VHF") == "Low"

    def test_none_storm_low(self):
        assert assess_comm_impact("None", "Ka") == "Low"

    def test_c_g4_high(self):
        assert assess_comm_impact("G4-Severe", "C") == "High"

    def test_invalid_storm_raises(self):
        with pytest.raises(ValueError):
            assess_comm_impact("G10", "L")

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError):
            assess_comm_impact("G1-Minor", "X-band")


# ── Charging Impact ─────────────────────────────────────────────────────────

class TestAssessChargingImpact:
    def test_g5_geo_high(self):
        assert assess_charging_impact("G5-Extreme", 35786) == "High"

    def test_g5_leo_moderate(self):
        assert assess_charging_impact("G5-Extreme", 500) == "Moderate"

    def test_g3_geo_high(self):
        assert assess_charging_impact("G3-Strong", 35786) == "High"

    def test_g3_leo_low(self):
        assert assess_charging_impact("G3-Strong", 500) == "Low"

    def test_g1_geo_moderate(self):
        assert assess_charging_impact("G1-Minor", 35786) == "Moderate"

    def test_none_geo_low(self):
        assert assess_charging_impact("None", 35786) == "Low"

    def test_invalid_storm_raises(self):
        with pytest.raises(ValueError):
            assess_charging_impact("G0", 500)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            assess_charging_impact("G1-Minor", -100)


# ── Atmospheric Density ─────────────────────────────────────────────────────

class TestAtmosphericDensity:
    def test_decreases_with_altitude(self):
        d1 = atmospheric_density(300)
        d2 = atmospheric_density(600)
        assert d2 < d1

    def test_storm_increases_density(self):
        d_quiet = atmospheric_density(500, "None")
        d_storm = atmospheric_density(500, "G5-Extreme")
        assert d_storm > d_quiet

    def test_known_value_quiet(self):
        # 2.5e-10 * exp(-(400-200)/50) = 2.5e-10 * exp(-4)
        expected = 2.5e-10 * math.exp(-4)
        assert atmospheric_density(400, "None") == pytest.approx(expected)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            atmospheric_density(-100)

    def test_invalid_storm_raises(self):
        with pytest.raises(ValueError):
            atmospheric_density(500, "G6")


# ── Drag Acceleration ───────────────────────────────────────────────────────

class TestDragAcceleration:
    def test_positive_at_leo(self):
        a = drag_acceleration(400, "None", 100)
        assert a > 0

    def test_storm_increases_drag(self):
        a_quiet = drag_acceleration(500, "None", 100)
        a_storm = drag_acceleration(500, "G5-Extreme", 100)
        assert a_storm > a_quiet

    def test_higher_altitude_less_drag(self):
        a_low = drag_acceleration(300, "None", 100)
        a_high = drag_acceleration(800, "None", 100)
        assert a_high < a_low

    def test_zero_ballistic_coefficient_raises(self):
        with pytest.raises(ValueError):
            drag_acceleration(500, "None", 0)


# ── Orbit Decay Rate ────────────────────────────────────────────────────────

class TestOrbitDecayRate:
    def test_positive_at_leo(self):
        rate = orbit_decay_rate(400, "None", 100)
        assert rate > 0

    def test_storm_increases_decay(self):
        r_quiet = orbit_decay_rate(500, "None", 100)
        r_storm = orbit_decay_rate(500, "G5-Extreme", 100)
        assert r_storm > r_quiet

    def test_higher_altitude_slower_decay(self):
        r_low = orbit_decay_rate(300, "None", 100)
        r_high = orbit_decay_rate(800, "None", 100)
        assert r_high < r_low

    def test_zero_at_geo(self):
        assert orbit_decay_rate(35786, "None", 100) < 1e-6


# ── Orbit Regime ────────────────────────────────────────────────────────────

class TestOrbitRegime:
    def test_leo(self):
        assert orbit_regime(500) == "LEO"

    def test_meo(self):
        assert orbit_regime(20000) == "MEO"

    def test_geo(self):
        assert orbit_regime(35786) == "GEO"

    def test_heo(self):
        assert orbit_regime(50000) == "HEO"

    def test_leo_boundary(self):
        assert orbit_regime(1999) == "LEO"

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            orbit_regime(-1)


# ── Orbital Velocity ────────────────────────────────────────────────────────

class TestOrbitalVelocity:
    def test_leo_velocity(self):
        v = orbital_velocity(400)
        assert v == pytest.approx(7.67, abs=0.1)

    def test_geo_velocity(self):
        v = orbital_velocity(35786)
        assert v == pytest.approx(3.07, abs=0.05)

    def test_decreases_with_altitude(self):
        assert orbital_velocity(400) > orbital_velocity(1000)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError):
            orbital_velocity(-100)


# ── Total Radiation Environment ─────────────────────────────────────────────

class TestTotalRadiationEnvironment:
    def test_returns_all_components(self):
        env = total_radiation_environment(500, 90, "C", "minimum")
        assert "trapped_proton_flux" in env
        assert "trapped_electron_flux" in env
        assert "solar_proton_flux" in env
        assert "gcr_flux" in env

    def test_values_match_individual_functions(self):
        env = total_radiation_environment(500, 90, "M", "maximum")
        assert env["trapped_proton_flux"] == pytest.approx(trapped_proton_flux(500, 90))
        assert env["solar_proton_flux"] == pytest.approx(solar_proton_event_flux("M"))
        assert env["gcr_flux"] == pytest.approx(galactic_cosmic_ray_flux("maximum"))

    def test_all_non_negative(self):
        env = total_radiation_environment(500, 90, "X", "minimum")
        assert all(v >= 0 for v in env.values())


# ── Estimate Dose ───────────────────────────────────────────────────────────

class TestEstimateDose:
    def test_zero_environment_zero_dose(self):
        env = {"trapped_proton_flux": 0, "trapped_electron_flux": 0,
               "solar_proton_flux": 0, "gcr_flux": 0}
        dose = estimate_dose(env, 1.0, 365)
        assert dose.total_dose_rad == 0.0

    def test_dose_increases_with_duration(self):
        env = total_radiation_environment(500, 90, "A", "minimum")
        d1 = estimate_dose(env, 1.0, 100)
        d2 = estimate_dose(env, 1.0, 200)
        assert d2.total_dose_rad == pytest.approx(2 * d1.total_dose_rad)

    def test_more_shielding_less_dose(self):
        env = total_radiation_environment(500, 90, "A", "minimum")
        d1 = estimate_dose(env, 0.5, 365)
        d2 = estimate_dose(env, 5.0, 365)
        assert d2.total_dose_rad < d1.total_dose_rad

    def test_dominant_source_identified(self):
        env = total_radiation_environment(500, 90, "X", "minimum")
        dose = estimate_dose(env, 1.0, 365)
        assert dose.dominant_source != ""

    def test_dose_rate_positive(self):
        env = total_radiation_environment(500, 90, "A", "minimum")
        dose = estimate_dose(env, 1.0, 365)
        assert dose.dose_rate_rad_s > 0


# ── SEU Probability ─────────────────────────────────────────────────────────

class TestSeuProbability:
    def test_zero_flux_zero_probability(self):
        assert seu_probability(0, 1e-12, 1e6) == 0.0

    def test_increases_with_duration(self):
        p1 = seu_probability(1e3, 1e-12, 1e5)
        p2 = seu_probability(1e3, 1e-12, 2e5)
        assert p2 > p1

    def test_bounded_0_to_1(self):
        p = seu_probability(1e6, 1e-10, 1e7)
        assert 0 <= p <= 1

    def test_high_flux_certain(self):
        p = seu_probability(1e9, 1e-8, 1e6)
        assert p > 0.99


# ── Mission Lifetime Risk ───────────────────────────────────────────────────

class TestMissionLifetimeRisk:
    def test_zero_dose_rate_zero_risk(self):
        assert mission_lifetime_risk(0, 365, 1000) == 0.0

    def test_high_dose_rate_high_risk(self):
        risk = mission_lifetime_risk(1e-4, 365, 100)
        assert risk > 0.9

    def test_low_dose_rate_low_risk(self):
        risk = mission_lifetime_risk(1e-8, 30, 10000)
        assert risk < 0.1

    def test_bounded_0_to_1(self):
        risk = mission_lifetime_risk(1e-3, 365, 10)
        assert 0 <= risk <= 1

    def test_negative_dose_rate_raises(self):
        with pytest.raises(ValueError):
            mission_lifetime_risk(-1, 365, 100)

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError):
            mission_lifetime_risk(1e-6, 365, 0)


# ── Mission Risk Score ──────────────────────────────────────────────────────

class TestMissionRiskScore:
    def test_all_low_standard(self):
        score = mission_risk_score(_impact(), "standard")
        assert score == pytest.approx(25.0)

    def test_all_high_standard(self):
        score = mission_risk_score(
            _impact("High", "High", "High", "High", "High"), "standard"
        )
        assert score == pytest.approx(100.0)

    def test_crewed_higher_than_standard(self):
        s_std = mission_risk_score(_impact(), "standard")
        s_crew = mission_risk_score(_impact(), "crewed")
        assert s_crew > s_std

    def test_capped_at_100(self):
        score = mission_risk_score(
            _impact("Critical", "Critical", "Critical", "Critical", "Critical"),
            "crewed",
        )
        assert score <= 100.0

    def test_experimental_lower(self):
        s_std = mission_risk_score(_impact(), "standard")
        s_exp = mission_risk_score(_impact(), "experimental")
        assert s_exp < s_std

    def test_invalid_criticality_raises(self):
        with pytest.raises(ValueError):
            mission_risk_score(_impact(), "military")


# ── Risk Category ───────────────────────────────────────────────────────────

class TestRiskCategory:
    def test_low(self):
        assert risk_category(10) == "Low"

    def test_moderate(self):
        assert risk_category(30) == "Moderate"

    def test_high(self):
        assert risk_category(60) == "High"

    def test_critical(self):
        assert risk_category(80) == "Critical"

    def test_boundary_low_moderate(self):
        assert risk_category(25) == "Moderate"

    def test_boundary_moderate_high(self):
        assert risk_category(50) == "High"

    def test_boundary_high_critical(self):
        assert risk_category(75) == "Critical"

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            risk_category(-1)


# ── Recommend Mitigations ───────────────────────────────────────────────────

class TestRecommendMitigations:
    def test_critical_risk_has_abort(self):
        impact = _impact("Critical", "Critical", "Critical", "Critical", "Critical")
        mitigations = recommend_mitigations("Critical", impact)
        assert any("abort" in m.lower() or "safe mode" in m.lower() for m in mitigations)

    def test_high_risk_has_monitoring(self):
        impact = _impact("High", "Low", "Low", "Low", "Low")
        mitigations = recommend_mitigations("High", impact)
        assert any("monitoring" in m.lower() for m in mitigations)

    def test_radiation_mitigation(self):
        impact = _impact("High", "Low", "Low", "Low", "Low")
        mitigations = recommend_mitigations("Moderate", impact)
        assert any("shielding" in m.lower() or "rad-hard" in m.lower() for m in mitigations)

    def test_drag_mitigation(self):
        impact = _impact("Low", "Low", "High", "Low", "Low")
        mitigations = recommend_mitigations("Moderate", impact)
        assert any("reboost" in m.lower() or "drag" in m.lower() for m in mitigations)

    def test_comm_mitigation(self):
        impact = _impact("Low", "Low", "Low", "High", "Low")
        mitigations = recommend_mitigations("Moderate", impact)
        assert any("frequency" in m.lower() or "coding" in m.lower() for m in mitigations)

    def test_charging_mitigation(self):
        impact = _impact("Low", "Low", "Low", "Low", "High")
        mitigations = recommend_mitigations("Moderate", impact)
        assert any("charge" in m.lower() for m in mitigations)

    def test_low_risk_no_mitigations(self):
        impact = _impact("Low", "Low", "Low", "Low", "Low")
        mitigations = recommend_mitigations("Low", impact)
        assert len(mitigations) == 0


# ── Cumulative Mission Risk ─────────────────────────────────────────────────

class TestCumulativeMissionRisk:
    def test_single_phase(self):
        phases = [{"name": "ops", "duration_days": 365, "risk_score": 50}]
        assert cumulative_mission_risk(phases) == pytest.approx(50.0)

    def test_time_weighted_average(self):
        phases = [
            {"name": "early", "duration_days": 100, "risk_score": 80},
            {"name": "late", "duration_days": 300, "risk_score": 20},
        ]
        expected = (80 * 100 + 20 * 300) / 400
        assert cumulative_mission_risk(phases) == pytest.approx(expected)

    def test_empty_phases_raises(self):
        with pytest.raises(ValueError):
            cumulative_mission_risk([])

    def test_zero_total_duration_raises(self):
        phases = [{"name": "a", "duration_days": 0, "risk_score": 50}]
        with pytest.raises(ValueError):
            cumulative_mission_risk(phases)


# ── Assess Radiation Impact ─────────────────────────────────────────────────

class TestAssessRadiationImpact:
    def test_leo_polar_high_trapped_proton(self):
        sat = _satellite(alt=500, inc=90)
        env = _env(flare="A", storm="None", solar="minimum")
        result = assess_radiation_impact(sat, env)
        assert result.radiation in ("Low", "Moderate", "High", "Critical")

    def test_geo_low_trapped_proton(self):
        sat = _satellite(alt=35786, inc=0)
        env = _env(flare="A", storm="None", solar="minimum")
        result = assess_radiation_impact(sat, env)
        assert result.radiation in ("Low", "Moderate", "High", "Critical")

    def test_x_flare_increases_radiation(self):
        sat = _satellite(alt=500, inc=90)
        env_quiet = _env(flare="A", storm="None", solar="minimum")
        env_flare = _env(flare="X", storm="None", solar="minimum")
        r_quiet = assess_radiation_impact(sat, env_quiet)
        r_flare = assess_radiation_impact(sat, env_flare)
        assert r_flare.total_dose_rad > r_quiet.total_dose_rad

    def test_more_shielding_reduces_dose(self):
        sat_thin = _satellite(alt=500, inc=90, shielding=0.5)
        sat_thick = _satellite(alt=500, inc=90, shielding=5.0)
        env = _env(flare="A", storm="None", solar="minimum")
        d_thin = assess_radiation_impact(sat_thin, env)
        d_thick = assess_radiation_impact(sat_thick, env)
        assert d_thick.total_dose_rad < d_thin.total_dose_rad

    def test_returns_impact_assessment(self):
        sat = _satellite()
        env = _env()
        result = assess_radiation_impact(sat, env)
        assert isinstance(result, ImpactAssessment)


# ── Assess Satellite Mission Risk ───────────────────────────────────────────

class TestAssessSatelliteMissionRisk:
    def test_returns_mission_risk(self):
        from src.space.impact_assessment import MissionRisk
        sat = _satellite()
        env = _env()
        result = assess_satellite_mission_risk(sat, env)
        assert isinstance(result, MissionRisk)

    def test_risk_score_bounded(self):
        sat = _satellite()
        env = _env()
        result = assess_satellite_mission_risk(sat, env)
        assert 0 <= result.risk_score <= 100

    def test_risk_category_valid(self):
        sat = _satellite()
        env = _env()
        result = assess_satellite_mission_risk(sat, env)
        assert result.risk_category in ("Low", "Moderate", "High", "Critical")

    def test_extreme_conditions_high_risk(self):
        sat = _satellite(alt=400, inc=90, shielding=0.1, duration=365)
        env = _env(flare="X", storm="G5-Extreme", solar="minimum")
        result = assess_satellite_mission_risk(sat, env, frequency_band="Ka", mission_criticality="crewed")
        assert result.risk_category in ("High", "Critical")

    def test_benign_conditions_low_risk(self):
        sat = _satellite(alt=35786, inc=0, shielding=5.0, duration=30)
        env = _env(flare="A", storm="None", solar="maximum")
        result = assess_satellite_mission_risk(sat, env, frequency_band="VHF", mission_criticality="experimental")
        assert result.risk_category in ("Low", "Moderate")

    def test_has_mitigations_for_high_risk(self):
        sat = _satellite(alt=400, inc=90, shielding=0.1)
        env = _env(flare="X", storm="G5-Extreme")
        result = assess_satellite_mission_risk(sat, env)
        if result.risk_category in ("High", "Critical"):
            assert len(result.mitigations) > 0

    def test_satellite_name_preserved(self):
        sat = _satellite(name="ISS-LIKE")
        env = _env()
        result = assess_satellite_mission_risk(sat, env)
        assert result.satellite_name == "ISS-LIKE"


# ── Compute Impact Score ────────────────────────────────────────────────────

class TestComputeImpactScore:
    def test_all_low(self):
        assert compute_impact_score(_impact()) == pytest.approx(25.0)

    def test_all_high(self):
        score = compute_impact_score(_impact("High", "High", "High", "High", "High"))
        assert score == pytest.approx(100.0)

    def test_moderate(self):
        score = compute_impact_score(_impact("Moderate", "Moderate", "Moderate", "Moderate", "Moderate"))
        assert score == pytest.approx(50.0)

    def test_bounded(self):
        score = compute_impact_score(_impact("Critical", "Critical", "Critical", "Critical", "Critical"))
        assert score <= 100.0
