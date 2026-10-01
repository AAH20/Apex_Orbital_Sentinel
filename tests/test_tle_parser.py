"""Tests for TLE file parser — 2-line and 3-line TLE parsing, checksum validation, field extraction."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from space.tle_parser import (
    TLE,
    TLEParseError,
    TLEChecksumError,
    parse_tle,
    parse_tle_file,
    validate_checksum,
    parse_scientific,
)


# Well-known TLEs for testing
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

NOAA_LINE1 = "1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
NOAA_LINE2 = "2 28654  99.0000 247.4627 0006703 130.5360 325.0288 14.20000000563537"


class TestTLEParsing:
    """TLE file parsing tests."""

    def test_parse_single_2line_tle(self):
        """Parse a single 2-line TLE."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert len(records) == 1
        assert records[0].satellite_number == 25544
        assert records[0].classification == "U"

    def test_parse_single_3line_tle(self):
        """Parse a single 3-line TLE with satellite name."""
        tle_text = f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert len(records) == 1
        assert records[0].name == "ISS (ZARYA)"
        assert records[0].satellite_number == 25544

    def test_parse_multiple_tles(self):
        """Parse multiple TLEs from one file."""
        tle_text = (
            f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n"
            f"NOAA 18\n{NOAA_LINE1}\n{NOAA_LINE2}\n"
        )
        records = parse_tle(tle_text)
        assert len(records) == 2
        assert records[0].name == "ISS (ZARYA)"
        assert records[1].name == "NOAA 18"

    def test_parse_tle_fields(self):
        """Verify all TLE fields are correctly extracted."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        r = records[0]
        assert r.international_designator == "98067A"
        assert r.epoch_year == 8
        assert r.epoch_day == pytest.approx(264.51782528)
        assert r.mean_motion_derivative == pytest.approx(-0.00002182)
        assert r.mean_motion_sec_derivative == pytest.approx(0.0)
        assert r.bstar_drag == pytest.approx(-0.11606e-4)
        assert r.ephemeris_type == 0
        assert r.element_set_number == 292
        assert r.line1_checksum == 7
        assert r.line2_checksum == 7
        assert r.inclination == pytest.approx(51.6416)
        assert r.raan == pytest.approx(247.4627)
        assert r.eccentricity == pytest.approx(0.0006703)
        assert r.arg_of_perigee == pytest.approx(130.5360)
        assert r.mean_anomaly == pytest.approx(325.0288)
        assert r.mean_motion == pytest.approx(15.72125391)
        assert r.revolution_number == 56353

    def test_parse_tle_with_blank_lines(self):
        """Parse TLE with blank lines between entries."""
        tle_text = (
            f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n"
            f"\n"
            f"NOAA 18\n{NOAA_LINE1}\n{NOAA_LINE2}\n"
        )
        records = parse_tle(tle_text)
        assert len(records) == 2

    def test_parse_empty_string_raises(self):
        """Empty string raises TLEParseError."""
        with pytest.raises(TLEParseError):
            parse_tle("")

    def test_parse_whitespace_only_raises(self):
        """Whitespace-only string raises TLEParseError."""
        with pytest.raises(TLEParseError):
            parse_tle("   \n  \n")

    def test_parse_incomplete_tle_raises(self):
        """Incomplete TLE (only line 1) raises TLEParseError."""
        tle_text = f"{ISS_LINE1}\n"
        with pytest.raises(TLEParseError):
            parse_tle(tle_text)

    def test_parse_invalid_line_prefix_raises(self):
        """Line not starting with '1' or '2' raises TLEParseError."""
        tle_text = f"X 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n{ISS_LINE2}\n"
        with pytest.raises(TLEParseError):
            parse_tle(tle_text)

    def test_parse_tle_from_file(self, tmp_path):
        """Parse TLE from a file path."""
        tle_file = tmp_path / "test.tle"
        tle_file.write_text(f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n")
        records = parse_tle_file(str(tle_file))
        assert len(records) == 1
        assert records[0].satellite_number == 25544

    def test_parse_tle_file_not_found_raises(self):
        """Non-existent file raises TLEParseError."""
        with pytest.raises(TLEParseError):
            parse_tle_file("/nonexistent/path/file.tle")

    def test_parse_short_line_raises(self):
        """Line shorter than 69 chars raises TLEParseError."""
        tle_text = f"1 25544U 98067A\n{ISS_LINE2}\n"
        with pytest.raises(TLEParseError):
            parse_tle(tle_text)


class TestTLEChecksum:
    """TLE checksum validation tests."""

    def test_validate_checksum_line1(self):
        """Validate line 1 checksum calculation."""
        assert validate_checksum(ISS_LINE1) == 7

    def test_validate_checksum_line2(self):
        """Validate line 2 checksum calculation."""
        assert validate_checksum(ISS_LINE2) == 7

    def test_checksum_valid_for_parsed_tle(self):
        """Parsed TLE has valid checksums."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].line1_checksum == validate_checksum(records[0].raw_line1)
        assert records[0].line2_checksum == validate_checksum(records[0].raw_line2)

    def test_checksum_invalid_line1(self):
        """Invalid line 1 checksum detected."""
        bad_line1 = ISS_LINE1[:-1] + "8"  # Change checksum from 7 to 8
        tle_text = f"{bad_line1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].line1_checksum != validate_checksum(records[0].raw_line1)

    def test_checksum_invalid_line2(self):
        """Invalid line 2 checksum detected."""
        bad_line2 = ISS_LINE2[:-1] + "8"  # Change checksum from 7 to 8
        tle_text = f"{ISS_LINE1}\n{bad_line2}\n"
        records = parse_tle(tle_text)
        assert records[0].line2_checksum != validate_checksum(records[0].raw_line2)

    def test_is_valid_method(self):
        """TLE.is_valid() returns True for valid checksums."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].is_valid() is True

    def test_is_valid_method_invalid(self):
        """TLE.is_valid() returns False for invalid checksums."""
        bad_line1 = ISS_LINE1[:-1] + "8"
        tle_text = f"{bad_line1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].is_valid() is False


class TestTLEFieldExtraction:
    """TLE field extraction and derived property tests."""

    def test_epoch_conversion(self):
        """Epoch year and day convert to correct datetime."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        epoch = records[0].epoch
        assert epoch.year == 2008
        assert epoch.month == 9
        assert epoch.day == 20

    def test_period_seconds(self):
        """Orbital period is correctly derived from mean motion."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        period = records[0].period_seconds
        assert 5000 < period < 6000  # ~92 minutes for ISS

    def test_raw_lines_preserved(self):
        """Raw TLE lines are preserved in the record."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].raw_line1 == ISS_LINE1
        assert records[0].raw_line2 == ISS_LINE2

    def test_name_defaults_to_empty(self):
        """Name defaults to empty string for 2-line TLE."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].name == ""

    def test_name_extracted_from_3line(self):
        """Name is extracted from 3-line TLE format."""
        tle_text = f"ISS (ZARYA)\n{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].name == "ISS (ZARYA)"

    def test_international_designator(self):
        """International designator is correctly extracted."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].international_designator == "98067A"

    def test_bstar_drag(self):
        """BSTAR drag term is correctly parsed."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].bstar_drag == pytest.approx(-0.11606e-4)

    def test_mean_motion_sec_derivative(self):
        """Mean motion second derivative is correctly parsed."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records = parse_tle(tle_text)
        assert records[0].mean_motion_sec_derivative == pytest.approx(0.0)


class TestParseScientific:
    """Scientific notation field parsing tests."""

    def test_parse_scientific_negative(self):
        """Parse negative scientific notation."""
        assert parse_scientific("-11606-4") == pytest.approx(-0.11606e-4)

    def test_parse_scientific_zero(self):
        """Parse zero scientific notation."""
        assert parse_scientific(" 00000-0") == pytest.approx(0.0)

    def test_parse_scientific_positive(self):
        """Parse positive scientific notation."""
        assert parse_scientific(" 12345+2") == pytest.approx(0.12345e+2)

    def test_parse_scientific_empty(self):
        """Parse empty field returns 0.0."""
        assert parse_scientific("") == 0.0
        assert parse_scientific("   ") == 0.0


class TestTLEDataclass:
    """TLE dataclass behavior tests."""

    def test_tle_creation(self):
        """TLE dataclass can be created with all fields."""
        tle = TLE(
            name="TEST",
            satellite_number=12345,
            classification="U",
            international_designator="2020-001A",
            epoch_year=20,
            epoch_day=100.5,
            mean_motion_derivative=0.0,
            mean_motion_sec_derivative=0.0,
            bstar_drag=0.0,
            ephemeris_type=0,
            element_set_number=1,
            line1_checksum=0,
            line2_checksum=0,
            inclination=51.0,
            raan=100.0,
            eccentricity=0.001,
            arg_of_perigee=200.0,
            mean_anomaly=300.0,
            mean_motion=15.0,
            revolution_number=100,
        )
        assert tle.satellite_number == 12345
        assert tle.name == "TEST"

    def test_tle_equality(self):
        """Two TLE objects with same data are equal."""
        tle_text = f"{ISS_LINE1}\n{ISS_LINE2}\n"
        records1 = parse_tle(tle_text)
        records2 = parse_tle(tle_text)
        assert records1[0] == records2[0]
