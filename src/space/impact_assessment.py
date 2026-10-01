"""Satellite impact assessment, radiation modeling, and mission risk analysis.

Extends the space weather module with quantitative models for:
- Trapped proton/electron radiation belts
- Solar proton events and galactic cosmic rays
- Dose behind shielding, SEU rates, displacement damage
- Atmospheric drag and orbit decay
- Thermal, communication, and charging impacts
- Mission-level risk scoring and mitigation recommendations
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
EARTH_MU_KM3_S2 = 398600.4418  # km³/s²
EARTH_RADIUS_KM = 6371.0
SECONDS_PER_DAY = 86400.0

# Radiation belt model parameters
TRAPPED_PROTON_PEAK_ALT_KM = 500.0
TRAPPED_PROTON_PEAK_FLUX = 1e4  # particles/cm²/s
TRAPPED_PROTON_SCALE_HEIGHT_KM = 50.0

TRAPPED_ELECTRON_INNER_EDGE_KM = 12000.0
TRAPPED_ELECTRON_PEAK_ALT_KM = 20000.0
TRAPPED_ELECTRON_PEAK_FLUX = 1e7  # particles/cm²/s
TRAPPED_ELECTRON_SCALE_HEIGHT_KM = 5000.0

# Solar proton event flux by flare class (particles/cm²/s)
SPE_FLUX_BY_CLASS = {"A": 1.0, "B": 10.0, "C": 100.0, "M": 1000.0, "X": 10000.0}

# Galactic cosmic ray flux by solar activity (particles/cm²/s)
GCR_FLUX = {"minimum": 4.0, "maximum": 1.5}

# Dose conversion factor (rad per particle/cm²)
DOSE_CONVERSION_RAD = 1e-8

# SEU cross-section (cm²) — typical for modern electronics
SEU_CROSS_SECTION_CM2 = 1e-12

# Displacement damage coefficient (rad per 1e10 particles/cm²)
DISPLACEMENT_DAMAGE_COEFF = 1e-3

# Atmospheric density model
RHO0_KG_M3 = 2.5e-10  # kg/m³ at 200 km
DENSITY_SCALE_HEIGHT_KM = 50.0
STORM_DENSITY_MULTIPLIER = 10.0

# Drag model
DRAG_COEFFICIENT = 2.2
SATELLITE_AREA_M2 = 1.0  # m² (reference area)

# Risk scoring weights
RISK_WEIGHTS = {"radiation": 0.30, "thermal": 0.15, "drag": 0.20, "comm": 0.20, "charging": 0.15}

# Criticality multipliers
CRITICALITY_MULTIPLIERS = {"experimental": 0.8, "standard": 1.0, "crewed": 1.5}

# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class SatelliteSpec:
    """Satellite specification for impact assessment."""
    name: str
    altitude_km: float
    inclination_deg: float
    shielding_g_cm2: float = 1.0
    mission_duration_days: float = 365.0
    mass_kg: float = 100.0
    area_m2: float = 1.0


@dataclass
class RadiationEnvironment:
    """Space weather radiation environment."""
    flare_class: str = "A"
    storm_severity: str = "None"
    solar_activity: str = "minimum"


@dataclass
class ImpactAssessment:
    """Multi-domain impact assessment result."""
    radiation: str = "Low"
    thermal: str = "Low"
    drag: str = "Low"
    comm: str = "Low"
    charging: str = "Low"
    overall: str = "Low"
    dose_rate_rad_s: float = 0.0
    total_dose_rad: float = 0.0
    seu_rate_per_s: float = 0.0
    dominant_source: str = "none"


@dataclass
class MissionRisk:
    """Mission-level risk assessment result."""
    satellite_name: str
    risk_score: float
    risk_category: str
    impact: ImpactAssessment
    mitigations: List[str] = field(default_factory=list)


# ── Trapped Proton Flux ─────────────────────────────────────────────────────

def trapped_proton_flux(altitude_km: float, inclination_deg: float) -> float:
    """Trapped proton flux (particles/cm²/s) at given altitude and inclination.

    Simplified AP-8-like model: Gaussian in altitude centered at 500 km,
    scaled by sin(inclination) to approximate magnetic latitude effect.
    """
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")
    if not 0 <= inclination_deg <= 180:
        raise ValueError("Inclination must be in [0, 180]")

    alt_factor = math.exp(-0.5 * ((altitude_km - TRAPPED_PROTON_PEAK_ALT_KM) / TRAPPED_PROTON_SCALE_HEIGHT_KM) ** 2)
    inc_factor = math.sin(math.radians(inclination_deg))
    return TRAPPED_PROTON_PEAK_FLUX * alt_factor * inc_factor


# ── Trapped Electron Flux ───────────────────────────────────────────────────

def trapped_electron_flux(altitude_km: float, inclination_deg: float) -> float:
    """Trapped electron flux (particles/cm²/s) at given altitude and inclination.

    Simplified AE-8-like model: zero below inner edge (12000 km),
    Gaussian peak at 20000 km, scaled by sin(inclination).
    """
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")
    if not 0 <= inclination_deg <= 180:
        raise ValueError("Inclination must be in [0, 180]")

    if altitude_km < TRAPPED_ELECTRON_INNER_EDGE_KM:
        return 0.0

    alt_factor = math.exp(-0.5 * ((altitude_km - TRAPPED_ELECTRON_PEAK_ALT_KM) / TRAPPED_ELECTRON_SCALE_HEIGHT_KM) ** 2)
    inc_factor = math.sin(math.radians(inclination_deg))
    return TRAPPED_ELECTRON_PEAK_FLUX * alt_factor * inc_factor


# ── Solar Proton Event Flux ─────────────────────────────────────────────────

def solar_proton_event_flux(flare_class: str) -> float:
    """Solar proton event flux (particles/cm²/s) by flare class."""
    if flare_class not in SPE_FLUX_BY_CLASS:
        raise ValueError(f"Invalid flare class: {flare_class}")
    return SPE_FLUX_BY_CLASS[flare_class]


# ── Galactic Cosmic Ray Flux ────────────────────────────────────────────────

def galactic_cosmic_ray_flux(solar_activity: str) -> float:
    """Galactic cosmic ray flux (particles/cm²/s) by solar activity level."""
    if solar_activity not in GCR_FLUX:
        raise ValueError(f"Invalid solar activity: {solar_activity}")
    return GCR_FLUX[solar_activity]


# ── Dose Behind Shielding ───────────────────────────────────────────────────

def dose_behind_shielding(particle_flux: float, shielding_g_cm2: float) -> float:
    """Dose rate (rad/s) behind aluminum shielding.

    Uses exponential attenuation: dose = flux * conversion * exp(-shielding/10)
    """
    if particle_flux < 0:
        raise ValueError("Particle flux cannot be negative")
    if shielding_g_cm2 < 0:
        raise ValueError("Shielding cannot be negative")

    attenuation = math.exp(-shielding_g_cm2 / 10.0)
    return particle_flux * DOSE_CONVERSION_RAD * attenuation


# ── Mission Dose ────────────────────────────────────────────────────────────

def mission_dose(dose_rate_rad_s: float, duration_days: float) -> float:
    """Total mission dose (rad) from a constant dose rate."""
    if dose_rate_rad_s < 0:
        raise ValueError("Dose rate cannot be negative")
    if duration_days < 0:
        raise ValueError("Duration cannot be negative")
    return dose_rate_rad_s * duration_days * SECONDS_PER_DAY


# ── SEU Rate ────────────────────────────────────────────────────────────────

def seu_rate(particle_flux: float, cross_section_cm2: float) -> float:
    """Single Event Upset rate (events/s) from particle flux and cross-section."""
    if particle_flux < 0:
        raise ValueError("Particle flux cannot be negative")
    if cross_section_cm2 <= 0:
        raise ValueError("Cross-section must be positive")
    return particle_flux * cross_section_cm2


# ── Displacement Damage ─────────────────────────────────────────────────────

def displacement_damage_dose(particle_fluence: float) -> float:
    """Displacement damage dose (rad) from particle fluence (particles/cm²)."""
    if particle_fluence < 0:
        raise ValueError("Particle fluence cannot be negative")
    return particle_fluence * DISPLACEMENT_DAMAGE_COEFF / 1e10


# ── Thermal Impact ──────────────────────────────────────────────────────────

def assess_thermal_impact(flare_class: str, altitude_km: float) -> str:
    """Assess thermal impact from solar flare heating.

    X-class flares cause significant heating at all altitudes.
    M-class flares cause moderate heating at LEO.
    """
    if flare_class not in ("A", "B", "C", "M", "X"):
        raise ValueError(f"Invalid flare class: {flare_class}")
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")

    if flare_class == "X":
        return "High"
    elif flare_class == "M":
        return "Moderate"
    else:
        return "Low"


# ── Drag Impact ─────────────────────────────────────────────────────────────

def assess_drag_impact(storm_severity: str, altitude_km: float, ballistic_coefficient: float) -> str:
    """Assess atmospheric drag impact from geomagnetic storm and altitude.

    Drag is most severe at low altitudes during strong storms.
    """
    valid_storms = ("None", "Quiet", "G1-Minor", "G2-Moderate", "G3-Strong", "G4-Severe", "G5-Extreme")
    if storm_severity not in valid_storms:
        raise ValueError(f"Invalid storm severity: {storm_severity}")
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")
    if ballistic_coefficient <= 0:
        raise ValueError("Ballistic coefficient must be positive")

    if storm_severity in ("G4-Severe", "G5-Extreme"):
        return "High" if altitude_km < 1000 else "Moderate"
    elif storm_severity == "G3-Strong":
        return "Moderate" if altitude_km < 2000 else "Low"
    elif storm_severity in ("G1-Minor", "G2-Moderate"):
        return "Low"
    else:
        return "Low"


# ── Comm Impact ─────────────────────────────────────────────────────────────

def assess_comm_impact(storm_severity: str, frequency_band: str) -> str:
    """Assess communication disruption from geomagnetic storm.

    Higher frequency bands (Ka, X) are more susceptible to ionospheric scintillation.
    """
    valid_storms = ("None", "Quiet", "G1-Minor", "G2-Moderate", "G3-Strong", "G4-Severe", "G5-Extreme")
    valid_bands = ("VHF", "UHF", "L", "S", "C", "X", "Ku", "Ka")
    if storm_severity not in valid_storms:
        raise ValueError(f"Invalid storm severity: {storm_severity}")
    if frequency_band not in valid_bands:
        raise ValueError(f"Invalid frequency band: {frequency_band}")

    if storm_severity == "G5-Extreme":
        if frequency_band in ("X", "Ku", "Ka"):
            return "High"
        elif frequency_band in ("C", "S", "L"):
            return "Moderate"
        else:
            return "Low"
    elif storm_severity == "G4-Severe":
        if frequency_band in ("X", "Ku", "Ka", "C"):
            return "High"
        elif frequency_band in ("S", "L"):
            return "Moderate"
        else:
            return "Low"
    elif storm_severity == "G3-Strong":
        if frequency_band in ("X", "Ku", "Ka"):
            return "Moderate"
        else:
            return "Low"
    elif storm_severity in ("G1-Minor", "G2-Moderate"):
        return "Low"
    else:
        return "Low"


# ── Charging Impact ─────────────────────────────────────────────────────────

def assess_charging_impact(storm_severity: str, altitude_km: float) -> str:
    """Assess spacecraft charging impact from geomagnetic storm.

    GEO is most susceptible to deep dielectric charging during storms.
    """
    valid_storms = ("None", "Quiet", "G1-Minor", "G2-Moderate", "G3-Strong", "G4-Severe", "G5-Extreme")
    if storm_severity not in valid_storms:
        raise ValueError(f"Invalid storm severity: {storm_severity}")
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")

    if storm_severity in ("G4-Severe", "G5-Extreme"):
        return "High" if altitude_km > 10000 else "Moderate"
    elif storm_severity == "G3-Strong":
        return "High" if altitude_km > 10000 else "Low"
    elif storm_severity in ("G1-Minor", "G2-Moderate"):
        return "Moderate" if altitude_km > 10000 else "Low"
    else:
        return "Low"


# ── Atmospheric Density ─────────────────────────────────────────────────────

def atmospheric_density(altitude_km: float, storm_severity: str = "None") -> float:
    """Atmospheric density (kg/m³) at given altitude and storm condition.

    Exponential model with storm-driven enhancement.
    """
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")
    valid_storms = ("None", "Quiet", "G1-Minor", "G2-Moderate", "G3-Strong", "G4-Severe", "G5-Extreme")
    if storm_severity not in valid_storms:
        raise ValueError(f"Invalid storm severity: {storm_severity}")

    base_density = RHO0_KG_M3 * math.exp(-(altitude_km - 200) / DENSITY_SCALE_HEIGHT_KM)

    if storm_severity in ("G4-Severe", "G5-Extreme"):
        return base_density * STORM_DENSITY_MULTIPLIER
    elif storm_severity == "G3-Strong":
        return base_density * 5.0
    elif storm_severity in ("G1-Minor", "G2-Moderate"):
        return base_density * 2.0
    else:
        return base_density


# ── Drag Acceleration ───────────────────────────────────────────────────────

def drag_acceleration(altitude_km: float, storm_severity: str, ballistic_coefficient: float) -> float:
    """Drag acceleration (m/s²) at given altitude and storm condition.

    a_drag = 0.5 * rho * v² * Cd * A / m = 0.5 * rho * v² / ballistic_coefficient
    where ballistic_coefficient = m / (Cd * A) in kg/m².
    """
    if ballistic_coefficient <= 0:
        raise ValueError("Ballistic coefficient must be positive")

    rho = atmospheric_density(altitude_km, storm_severity)
    v = orbital_velocity(altitude_km) * 1000.0  # convert km/s to m/s
    return 0.5 * rho * v * v / ballistic_coefficient


# ── Orbit Decay Rate ────────────────────────────────────────────────────────

def orbit_decay_rate(altitude_km: float, storm_severity: str, ballistic_coefficient: float) -> float:
    """Orbit decay rate (km/day) from atmospheric drag.

    da/dt = -2π * ρ * a² / (n * ballistic_coefficient)
    where n is mean motion in rad/s.
    """
    if ballistic_coefficient <= 0:
        raise ValueError("Ballistic coefficient must be positive")

    rho = atmospheric_density(altitude_km, storm_severity)
    a_km = EARTH_RADIUS_KM + altitude_km
    n = math.sqrt(EARTH_MU_KM3_S2 / (a_km ** 3))  # rad/s

    # da/dt in km/s, then convert to km/day
    da_dt_km_s = -2 * math.pi * rho * (a_km * 1000) ** 2 / (ballistic_coefficient * n * 1000)
    return abs(da_dt_km_s) * SECONDS_PER_DAY


# ── Orbit Regime ────────────────────────────────────────────────────────────

def orbit_regime(altitude_km: float) -> str:
    """Classify orbit regime by altitude."""
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")
    if altitude_km < 2000:
        return "LEO"
    elif altitude_km < 35786:
        return "MEO"
    elif altitude_km < 40000:
        return "GEO"
    else:
        return "HEO"


# ── Orbital Velocity ────────────────────────────────────────────────────────

def orbital_velocity(altitude_km: float) -> float:
    """Circular orbital velocity (km/s) at given altitude."""
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")
    r_km = EARTH_RADIUS_KM + altitude_km
    return math.sqrt(EARTH_MU_KM3_S2 / r_km)


# ── Total Radiation Environment ─────────────────────────────────────────────

def total_radiation_environment(
    altitude_km: float,
    inclination_deg: float,
    flare_class: str,
    solar_activity: str,
) -> Dict[str, float]:
    """Compute total radiation environment components.

    Returns dict with trapped_proton_flux, trapped_electron_flux,
    solar_proton_flux, and gcr_flux.
    """
    return {
        "trapped_proton_flux": trapped_proton_flux(altitude_km, inclination_deg),
        "trapped_electron_flux": trapped_electron_flux(altitude_km, inclination_deg),
        "solar_proton_flux": solar_proton_event_flux(flare_class),
        "gcr_flux": galactic_cosmic_ray_flux(solar_activity),
    }


# ── Estimate Dose ───────────────────────────────────────────────────────────

def estimate_dose(
    radiation_env: Dict[str, float],
    shielding_g_cm2: float,
    duration_days: float,
) -> ImpactAssessment:
    """Estimate total radiation dose from environment components.

    Computes dose rate and total dose for each radiation source,
    applies shielding attenuation, and identifies dominant source.
    """
    sources = {
        "trapped_proton": radiation_env.get("trapped_proton_flux", 0),
        "trapped_electron": radiation_env.get("trapped_electron_flux", 0),
        "solar_proton": radiation_env.get("solar_proton_flux", 0),
        "gcr": radiation_env.get("gcr_flux", 0),
    }

    dose_rates = {}
    for name, flux in sources.items():
        dose_rates[name] = dose_behind_shielding(flux, shielding_g_cm2)

    total_rate = sum(dose_rates.values())
    total_dose = mission_dose(total_rate, duration_days)

    dominant = max(dose_rates, key=dose_rates.get) if total_rate > 0 else "none"

    # Classify radiation risk level
    if total_dose > 10000:
        radiation_level = "Critical"
    elif total_dose > 1000:
        radiation_level = "High"
    elif total_dose > 100:
        radiation_level = "Moderate"
    else:
        radiation_level = "Low"

    return ImpactAssessment(
        radiation=radiation_level,
        total_dose_rad=total_dose,
        dose_rate_rad_s=total_rate,
        dominant_source=dominant,
    )


# ── SEU Probability ─────────────────────────────────────────────────────────

def seu_probability(
    particle_flux: float,
    cross_section_cm2: float,
    duration_s: float,
) -> float:
    """Probability of at least one SEU over a duration.

    P(SEU) = 1 - exp(-flux * cross_section * duration)
    """
    if particle_flux < 0:
        raise ValueError("Particle flux cannot be negative")
    if cross_section_cm2 <= 0:
        raise ValueError("Cross-section must be positive")
    if duration_s < 0:
        raise ValueError("Duration cannot be negative")

    expected_events = particle_flux * cross_section_cm2 * duration_s
    return 1.0 - math.exp(-expected_events)


# ── Mission Lifetime Risk ───────────────────────────────────────────────────

def mission_lifetime_risk(
    dose_rate_rad_s: float,
    duration_days: float,
    threshold_rad: float,
) -> float:
    """Mission lifetime risk from cumulative radiation dose.

    Risk = 1 - exp(-total_dose / threshold)
    """
    if dose_rate_rad_s < 0:
        raise ValueError("Dose rate cannot be negative")
    if duration_days < 0:
        raise ValueError("Duration cannot be negative")
    if threshold_rad <= 0:
        raise ValueError("Threshold must be positive")

    total_dose = mission_dose(dose_rate_rad_s, duration_days)
    return 1.0 - math.exp(-total_dose / threshold_rad)


# ── Mission Risk Score ──────────────────────────────────────────────────────

def mission_risk_score(impact: ImpactAssessment, mission_criticality: str = "standard") -> float:
    """Compute mission risk score (0-100) from impact assessment.

    Weighted sum of domain risk levels, scaled by mission criticality.
    """
    if mission_criticality not in CRITICALITY_MULTIPLIERS:
        raise ValueError(f"Invalid mission criticality: {mission_criticality}")

    level_values = {"Low": 25, "Moderate": 50, "High": 100, "Critical": 100}

    score = sum(
        RISK_WEIGHTS[domain] * level_values.get(getattr(impact, domain), 25)
        for domain in RISK_WEIGHTS
    )

    score *= CRITICALITY_MULTIPLIERS[mission_criticality]
    return min(score, 100.0)


# ── Risk Category ───────────────────────────────────────────────────────────

def risk_category(score: float) -> str:
    """Classify risk score into category."""
    if score < 0:
        raise ValueError("Risk score cannot be negative")
    if score < 25:
        return "Low"
    elif score < 50:
        return "Moderate"
    elif score < 75:
        return "High"
    else:
        return "Critical"


# ── Recommend Mitigations ───────────────────────────────────────────────────

def recommend_mitigations(risk_category_str: str, impact: ImpactAssessment) -> List[str]:
    """Recommend mitigation strategies based on risk category and impact domains."""
    mitigations = []

    if risk_category_str == "Critical":
        mitigations.append("Enter safe mode and abort non-critical operations")
        mitigations.append("Orient spacecraft to minimize radiation exposure")
        return mitigations

    if risk_category_str == "High":
        mitigations.append("Increase monitoring frequency to 1-minute cadence")

    if impact.radiation in ("High", "Critical"):
        mitigations.append("Enable radiation-hardened mode and increase shielding coverage")

    if impact.drag in ("High", "Critical"):
        mitigations.append("Plan reboost maneuver to compensate for increased drag")

    if impact.comm in ("High", "Critical"):
        mitigations.append("Switch to lower frequency band and enable forward error correction")

    if impact.charging in ("High", "Critical"):
        mitigations.append("Enable charge dissipation system and reduce high-voltage operations")

    if impact.thermal in ("High", "Critical"):
        mitigations.append("Reduce power dissipation and enable thermal protection mode")

    return mitigations


# ── Cumulative Mission Risk ─────────────────────────────────────────────────

def cumulative_mission_risk(phases: List[Dict]) -> float:
    """Compute cumulative mission risk as time-weighted average of phase risks.

    Each phase dict must have 'duration_days' and 'risk_score' keys.
    """
    if not phases:
        raise ValueError("At least one phase is required")

    total_duration = sum(p["duration_days"] for p in phases)
    if total_duration <= 0:
        raise ValueError("Total duration must be positive")

    weighted_sum = sum(p["duration_days"] * p["risk_score"] for p in phases)
    return weighted_sum / total_duration


# ── Assess Radiation Impact ─────────────────────────────────────────────────

def assess_radiation_impact(
    satellite: SatelliteSpec,
    env: RadiationEnvironment,
) -> ImpactAssessment:
    """Assess radiation impact on a satellite from space weather environment.

    Computes trapped proton/electron flux, solar proton event flux,
    and GCR flux at the satellite's orbit, then estimates total dose.
    """
    radiation_env = total_radiation_environment(
        satellite.altitude_km,
        satellite.inclination_deg,
        env.flare_class,
        env.solar_activity,
    )

    dose = estimate_dose(radiation_env, satellite.shielding_g_cm2, satellite.mission_duration_days)

    # Compute SEU rate from total particle flux
    total_flux = sum(radiation_env.values())
    dose.seu_rate_per_s = seu_rate(total_flux, SEU_CROSS_SECTION_CM2)

    return dose


# ── Assess Satellite Mission Risk ───────────────────────────────────────────

def assess_satellite_mission_risk(
    satellite: SatelliteSpec,
    env: RadiationEnvironment,
    frequency_band: str = "S",
    mission_criticality: str = "standard",
) -> MissionRisk:
    """Comprehensive mission risk assessment for a satellite.

    Combines radiation, thermal, drag, comm, and charging impacts
    into an overall mission risk score with mitigation recommendations.
    """
    # Radiation impact
    radiation = assess_radiation_impact(satellite, env)

    # Thermal impact
    thermal = assess_thermal_impact(env.flare_class, satellite.altitude_km)

    # Drag impact
    ballistic_coeff = satellite.mass_kg / (DRAG_COEFFICIENT * satellite.area_m2)
    drag = assess_drag_impact(env.storm_severity, satellite.altitude_km, ballistic_coeff)

    # Comm impact
    comm = assess_comm_impact(env.storm_severity, frequency_band)

    # Charging impact
    charging = assess_charging_impact(env.storm_severity, satellite.altitude_km)

    # Overall impact
    impact = ImpactAssessment(
        radiation=radiation.radiation,
        thermal=thermal,
        drag=drag,
        comm=comm,
        charging=charging,
        overall="Low",
        dose_rate_rad_s=radiation.dose_rate_rad_s,
        total_dose_rad=radiation.total_dose_rad,
        seu_rate_per_s=radiation.seu_rate_per_s,
    )

    # Compute overall risk level
    levels = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    max_level = max(levels[impact.radiation], levels[impact.thermal], levels[impact.drag],
                    levels[impact.comm], levels[impact.charging])
    impact.overall = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}[max_level]

    # Mission risk score
    score = mission_risk_score(impact, mission_criticality)
    category = risk_category(score)

    # Mitigations
    mitigations = recommend_mitigations(category, impact)

    return MissionRisk(
        satellite_name=satellite.name,
        risk_score=score,
        risk_category=category,
        impact=impact,
        mitigations=mitigations,
    )


# ── Compute Impact Score ────────────────────────────────────────────────────

def compute_impact_score(impact: ImpactAssessment) -> float:
    """Compute overall impact score (0-100) from multi-domain impact assessment."""
    level_values = {"Low": 25, "Moderate": 50, "High": 100, "Critical": 100}
    return sum(
        RISK_WEIGHTS[domain] * level_values.get(getattr(impact, domain), 25)
        for domain in RISK_WEIGHTS
    )
