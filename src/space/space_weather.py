"""Space weather: solar flares, geomagnetic storms, satellite impact."""
from dataclasses import dataclass
from typing import List


# ── Solar Flares ─────────────────────────────────────────────────────────

@dataclass
class SolarFlare:
    flare_class: str
    peak_flux: float
    start_index: int


def classify_flare(flux: float) -> str:
    """Classify solar flare by X-ray flux (W/m²)."""
    if flux < 0:
        raise ValueError("Flux cannot be negative")
    if flux < 1e-7:
        return "A"
    if flux < 1e-6:
        return "B"
    if flux < 1e-5:
        return "C"
    if flux < 1e-4:
        return "M"
    return "X"


def detect_flares(flux_series: List[float]) -> List[SolarFlare]:
    """Detect flare events from X-ray flux time series.

    A flare is a contiguous region where flux >= 1e-6 (C-class threshold).
    """
    flares = []
    in_flare = False
    start = 0
    peak = 0.0

    for i, flux in enumerate(flux_series):
        if flux >= 1e-6:
            if not in_flare:
                in_flare = True
                start = i
                peak = flux
            else:
                peak = max(peak, flux)
        else:
            if in_flare:
                flares.append(SolarFlare(
                    flare_class=classify_flare(peak),
                    peak_flux=peak,
                    start_index=start,
                ))
                in_flare = False

    if in_flare:
        flares.append(SolarFlare(
            flare_class=classify_flare(peak),
            peak_flux=peak,
            start_index=start,
        ))

    return flares


# ── Geomagnetic Storms ──────────────────────────────────────────────────

@dataclass
class GeomagneticStorm:
    severity: str
    kp_index: int
    dst_index: float


def storm_severity(kp: int) -> str:
    """Classify geomagnetic storm severity by Kp index."""
    if kp < 0:
        raise ValueError("Kp index cannot be negative")
    if kp <= 2:
        return "None"
    if kp <= 4:
        return "Quiet"
    if kp == 5:
        return "G1-Minor"
    if kp == 6:
        return "G2-Moderate"
    if kp == 7:
        return "G3-Strong"
    if kp == 8:
        return "G4-Severe"
    return "G5-Extreme"


def predict_geomagnetic_storm(kp_index: int, dst_index: float) -> GeomagneticStorm:
    """Predict geomagnetic storm from Kp index and Dst index.

    Severe Dst (< -100 nT) can escalate severity by one level.
    """
    severity = storm_severity(kp_index)

    # Escalate if Dst is severely negative
    if dst_index < -100 and severity not in ("G5-Extreme", "Quiet", "None"):
        escalation = {
            "G1-Minor": "G2-Moderate",
            "G2-Moderate": "G3-Strong",
            "G3-Strong": "G4-Severe",
            "G4-Severe": "G5-Extreme",
        }
        severity = escalation.get(severity, severity)

    return GeomagneticStorm(
        severity=severity,
        kp_index=kp_index,
        dst_index=dst_index,
    )


# ── Satellite Impact ────────────────────────────────────────────────────

@dataclass
class SatelliteImpact:
    radiation_risk: str
    drag_increase: str
    comm_disruption: str
    overall_risk: str


def assess_satellite_impact(
    flare_class: str,
    storm_severity_str: str,
    altitude_km: float,
) -> SatelliteImpact:
    """Assess satellite impact from space weather conditions."""
    valid_classes = ("A", "B", "C", "M", "X")
    if flare_class not in valid_classes:
        raise ValueError(f"Invalid flare class: {flare_class}")
    if altitude_km < 0:
        raise ValueError("Altitude cannot be negative")

    # Radiation risk: driven by flare class and altitude
    if flare_class == "X":
        radiation_risk = "High"
    elif flare_class == "M":
        radiation_risk = "Moderate" if altitude_km > 1000 else "Low"
    elif flare_class == "C":
        radiation_risk = "Low"
    else:
        radiation_risk = "Low"

    # Drag increase: driven by storm severity, flare class, and altitude
    if storm_severity_str in ("G4-Severe", "G5-Extreme"):
        drag_increase = "High" if altitude_km < 1000 else "Moderate"
    elif storm_severity_str == "G3-Strong":
        drag_increase = "Moderate" if altitude_km < 2000 else "Low"
    elif storm_severity_str in ("G1-Minor", "G2-Moderate"):
        if flare_class in ("M", "X") and altitude_km < 1000:
            drag_increase = "Moderate"
        else:
            drag_increase = "Low"
    else:
        if flare_class == "X" and altitude_km < 1000:
            drag_increase = "Moderate"
        else:
            drag_increase = "Low"

    # Communication disruption: driven by storm severity
    if storm_severity_str in ("G4-Severe", "G5-Extreme"):
        comm_disruption = "High"
    elif storm_severity_str == "G3-Strong":
        comm_disruption = "Moderate"
    elif storm_severity_str in ("G1-Minor", "G2-Moderate"):
        comm_disruption = "Low"
    else:
        comm_disruption = "Low"

    # Overall risk: worst of the three
    risk_levels = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    overall = max(radiation_risk, drag_increase, comm_disruption, key=lambda r: risk_levels[r])
    if radiation_risk == "High" and comm_disruption == "High":
        overall = "Critical"

    return SatelliteImpact(
        radiation_risk=radiation_risk,
        drag_increase=drag_increase,
        comm_disruption=comm_disruption,
        overall_risk=overall,
    )
