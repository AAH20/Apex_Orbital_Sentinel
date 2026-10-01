"""TLE (Two-Line Element) file parser.

Supports 2-line and 3-line TLE formats with checksum validation
and field extraction. Does NOT implement SGP4 propagation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List


class TLEParseError(Exception):
    """Raised when TLE parsing fails."""
    pass


class TLEChecksumError(Exception):
    """Raised when TLE checksum validation fails."""
    pass


@dataclass
class TLE:
    """Two-Line Element set representation.

    Parses and validates TLE data, providing access to all orbital elements.
    """
    satellite_number: int
    classification: str
    international_designator: str
    epoch_year: int
    epoch_day: float
    mean_motion_derivative: float
    mean_motion_sec_derivative: float
    bstar_drag: float
    ephemeris_type: int
    element_set_number: int
    line1_checksum: int
    line2_checksum: int
    inclination: float
    raan: float
    eccentricity: float
    arg_of_perigee: float
    mean_anomaly: float
    mean_motion: float
    revolution_number: int
    name: str = ""
    raw_line1: str = ""
    raw_line2: str = ""

    @property
    def epoch(self) -> datetime:
        """Convert epoch year/day to datetime."""
        year = 2000 + self.epoch_year if self.epoch_year < 57 else 1900 + self.epoch_year
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        return year_start + timedelta(days=self.epoch_day - 1)

    @property
    def period_seconds(self) -> float:
        """Orbital period in seconds."""
        return 86400.0 / self.mean_motion

    def is_valid(self) -> bool:
        """Check if both line checksums are valid."""
        if self.raw_line1 and self.raw_line2:
            return (
                self.line1_checksum == validate_checksum(self.raw_line1)
                and self.line2_checksum == validate_checksum(self.raw_line2)
            )
        return False


def validate_checksum(line: str) -> int:
    """Calculate TLE checksum for a line (excluding the checksum character).

    The checksum is the sum of all digits, with minus signs counting as 1,
    modulo 10. The checksum character is the last character of the line,
    so only the first 68 characters are used in the calculation.
    """
    total = 0
    for char in line[:68]:
        if char.isdigit():
            total += int(char)
        elif char == "-":
            total += 1
    return total % 10


def parse_scientific(field: str) -> float:
    """Parse TLE scientific notation field (e.g., '-11606-4' -> -0.11606e-4).

    TLE format: sign, 5 digits, sign, 1 exponent digit.
    The decimal point is implied before the mantissa.
    Field is fixed-width (8 chars); leading spaces indicate positive sign.
    """
    if not field or not field.strip():
        return 0.0
    try:
        sign = -1 if field[0] == "-" else 1
        mantissa = field[1:6]
        exp_sign = -1 if field[6] == "-" else 1
        exponent = int(field[7])
        value = sign * float(f"0.{mantissa}") * (10 ** (exp_sign * exponent))
        return value
    except (ValueError, IndexError):
        return 0.0


def parse_tle(text: str) -> List[TLE]:
    """Parse TLE text and return list of TLE objects.

    Supports both 2-line and 3-line TLE formats. Blank lines are ignored.
    """
    if not text or not text.strip():
        raise TLEParseError("Empty TLE text")

    lines = [line.rstrip("\n\r") for line in text.strip().split("\n")]
    records: List[TLE] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if this is a name line (3-line TLE) or line 1 (2-line TLE)
        if line.startswith("1 ") and len(line) >= 69:
            # 2-line TLE: line 1 at i, line 2 at i+1
            if i + 1 >= len(lines):
                raise TLEParseError(f"Incomplete TLE at line {i + 1}: missing line 2")
            line1 = line
            line2 = lines[i + 1].strip()
            name = ""
            i += 2
        elif (
            i + 2 < len(lines)
            and lines[i + 1].strip().startswith("1 ")
            and lines[i + 2].strip().startswith("2 ")
        ):
            # 3-line TLE: name at i, line 1 at i+1, line 2 at i+2
            name = line
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            i += 3
        else:
            raise TLEParseError(
                f"Invalid TLE format at line {i + 1}: expected line starting with '1'"
            )

        record = _parse_record(name, line1, line2)
        records.append(record)

    if not records:
        raise TLEParseError("No valid TLE records found")

    return records


def parse_tle_file(filepath: str) -> List[TLE]:
    """Parse TLE from a file path."""
    path = Path(filepath)
    if not path.exists():
        raise TLEParseError(f"File not found: {filepath}")
    text = path.read_text()
    return parse_tle(text)


def _parse_record(name: str, line1: str, line2: str) -> TLE:
    """Parse a single TLE record from two lines."""
    if len(line1) < 69:
        raise TLEParseError(f"Line 1 too short: {len(line1)} chars, need 69")
    if len(line2) < 69:
        raise TLEParseError(f"Line 2 too short: {len(line2)} chars, need 69")

    if not line1.startswith("1 "):
        raise TLEParseError(f"Line 1 must start with '1 ': {line1[:10]}")
    if not line2.startswith("2 "):
        raise TLEParseError(f"Line 2 must start with '2 ': {line2[:10]}")

    try:
        satellite_number = int(line1[2:7])
    except ValueError as e:
        raise TLEParseError(f"Invalid satellite number: {line1[2:7]}") from e

    classification = line1[7]
    international_designator = line1[9:17].strip()

    try:
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
    except ValueError as e:
        raise TLEParseError(f"Invalid epoch: {line1[18:32]}") from e

    try:
        mean_motion_derivative = float(line1[33:43])
    except ValueError as e:
        raise TLEParseError(f"Invalid mean motion derivative: {line1[33:43]}") from e

    mean_motion_sec_derivative = parse_scientific(line1[44:52])
    bstar_drag = parse_scientific(line1[53:61])

    try:
        ephemeris_type = int(line1[62])
        element_set_number = int(line1[64:68])
        line1_checksum = int(line1[68])
    except ValueError as e:
        raise TLEParseError(f"Invalid line 1 fields: {line1[62:]}") from e

    try:
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33].strip())
        arg_of_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        revolution_number = int(line2[63:68])
        line2_checksum = int(line2[68])
    except ValueError as e:
        raise TLEParseError(f"Invalid line 2 fields: {e}") from e

    return TLE(
        satellite_number=satellite_number,
        classification=classification,
        international_designator=international_designator,
        epoch_year=epoch_year,
        epoch_day=epoch_day,
        mean_motion_derivative=mean_motion_derivative,
        mean_motion_sec_derivative=mean_motion_sec_derivative,
        bstar_drag=bstar_drag,
        ephemeris_type=ephemeris_type,
        element_set_number=element_set_number,
        line1_checksum=line1_checksum,
        line2_checksum=line2_checksum,
        inclination=inclination,
        raan=raan,
        eccentricity=eccentricity,
        arg_of_perigee=arg_of_perigee,
        mean_anomaly=mean_anomaly,
        mean_motion=mean_motion,
        revolution_number=revolution_number,
        name=name,
        raw_line1=line1,
        raw_line2=line2,
    )
