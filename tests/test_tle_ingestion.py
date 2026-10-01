"""Tests for TLE/OMM ingestion — TDD RED phase."""
import os
import sys
import pytest
import tempfile
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from space.tle_ingestion import (
    TLEParser,
    OMMParser,
    TLEValidator,
    TLEStorage,
    TLERecord,
    OMMRecord,
    TLEParseError,
    OMMParseError,
    ValidationError,
)


# ── TLE Parsing Tests ──

class TestTLEParsing:
    """TLE file parsing tests."""

    def test_parse_single_2line_tle(self):
        """Parse a single 2-line TLE."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        assert len(records) == 1
        assert records[0].satellite_number == 25544
        assert records[0].classification == "U"

    def test_parse_single_3line_tle(self):
        """Parse a single 3-line TLE with satellite name."""
        tle_text = (
            "ISS (ZARYA)\n"
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        assert len(records) == 1
        assert records[0].name == "ISS (ZARYA)"
        assert records[0].satellite_number == 25544

    def test_parse_multiple_tles(self):
        """Parse multiple TLEs from one file."""
        tle_text = (
            "ISS (ZARYA)\n"
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
            "NOAA 18\n"
            "1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 28654  99.0000 247.4627 0006703 130.5360 325.0288 14.20000000563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        assert len(records) == 2
        assert records[0].name == "ISS (ZARYA)"
        assert records[1].name == "NOAA 18"

    def test_parse_tle_fields(self):
        """Verify all TLE fields are correctly extracted."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        r = records[0]
        assert r.international_designator == "98067A"
        assert r.epoch_year == 8
        assert r.epoch_day == 264.51782528
        assert r.mean_motion_derivative == -0.00002182
        assert r.mean_motion_sec_derivative == 0.0
        assert r.bstar_drag == -0.11606e-4
        assert r.ephemeris_type == 0
        assert r.element_set_number == 292
        assert r.line1_checksum == 7
        assert r.line2_checksum == 7
        assert r.inclination == 51.6416
        assert r.raan == 247.4627
        assert r.eccentricity == 0.0006703
        assert r.arg_of_perigee == 130.5360
        assert r.mean_anomaly == 325.0288
        assert r.mean_motion == 15.72125391
        assert r.revolution_number == 56353

    def test_parse_tle_with_blank_lines(self):
        """Parse TLE with blank lines between entries."""
        tle_text = (
            "ISS (ZARYA)\n"
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
            "\n"
            "NOAA 18\n"
            "1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 28654  99.0000 247.4627 0006703 130.5360 325.0288 14.20000000563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        assert len(records) == 2

    def test_parse_empty_string_raises(self):
        """Empty string raises TLEParseError."""
        parser = TLEParser()
        with pytest.raises(TLEParseError):
            parser.parse("")

    def test_parse_incomplete_tle_raises(self):
        """Incomplete TLE (only line 1) raises TLEParseError."""
        tle_text = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
        parser = TLEParser()
        with pytest.raises(TLEParseError):
            parser.parse(tle_text)

    def test_parse_invalid_line_prefix_raises(self):
        """Line not starting with '1' or '2' raises TLEParseError."""
        tle_text = (
            "X 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        with pytest.raises(TLEParseError):
            parser.parse(tle_text)

    def test_parse_tle_from_file(self, tmp_path):
        """Parse TLE from a file path."""
        tle_file = tmp_path / "test.tle"
        tle_file.write_text(
            "ISS (ZARYA)\n"
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse_file(str(tle_file))
        assert len(records) == 1
        assert records[0].satellite_number == 25544


# ── TLE Validation Tests ──

class TestTLEValidation:
    """TLE validation tests."""

    def test_valid_tle_passes(self):
        """Valid TLE passes validation."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        assert validator.validate(records[0]) is True

    def test_invalid_checksum_line1_fails(self):
        """Invalid line 1 checksum fails validation."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(records[0])

    def test_invalid_checksum_line2_fails(self):
        """Invalid line 2 checksum fails validation."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563538\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(records[0])

    def test_inclination_out_of_range_fails(self):
        """Inclination > 180 degrees fails validation."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544 181.0000 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(records[0])

    def test_eccentricity_out_of_range_fails(self):
        """Eccentricity >= 1.0 fails validation."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 1006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(records[0])

    def test_mean_motion_out_of_range_fails(self):
        """Mean motion <= 0 fails validation."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288  0.00000000563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(records[0])

    def test_validate_multiple_records(self):
        """Validate multiple records, return list of valid ones."""
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
            "1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928\n"
            "2 28654  99.0000 247.4627 0006703 130.5360 325.0288 14.20000000563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        validator = TLEValidator()
        valid = validator.validate_all(records)
        assert len(valid) == 1
        assert valid[0].satellite_number == 25544


# ── OMM XML Parsing Tests ──

class TestOMMXMLParsing:
    """OMM XML parsing tests."""

    def test_parse_omm_xml(self):
        """Parse a valid OMM XML."""
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""
        parser = OMMParser()
        records = parser.parse(xml_text)
        assert len(records) == 1
        assert records[0].object_name == "ISS (ZARYA)"
        assert records[0].object_id == "1998-067A"
        assert records[0].norad_cat_id == 25544
        assert records[0].mean_motion == 15.72125391
        assert records[0].eccentricity == 0.0006703
        assert records[0].inclination == 51.6416
        assert records[0].ra_of_asc_node == 247.4627
        assert records[0].arg_of_pericenter == 130.5360
        assert records[0].mean_anomaly == 325.0288

    def test_parse_omm_xml_multiple_segments(self):
        """Parse OMM XML with multiple segments."""
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
    <segment>
      <metadata>
        <OBJECT_NAME>NOAA 18</OBJECT_NAME>
        <OBJECT_ID>2005-018A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>14.20000000</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>99.0000</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>28654</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""
        parser = OMMParser()
        records = parser.parse(xml_text)
        assert len(records) == 2
        assert records[0].object_name == "ISS (ZARYA)"
        assert records[1].object_name == "NOAA 18"

    def test_parse_omm_xml_missing_required_field_raises(self):
        """OMM XML missing required field raises OMMParseError."""
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""
        parser = OMMParser()
        with pytest.raises(OMMParseError):
            parser.parse(xml_text)

    def test_parse_omm_xml_invalid_xml_raises(self):
        """Invalid XML raises OMMParseError."""
        xml_text = "not valid xml <<<"
        parser = OMMParser()
        with pytest.raises(OMMParseError):
            parser.parse(xml_text)

    def test_parse_omm_xml_from_file(self, tmp_path):
        """Parse OMM XML from a file path."""
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>""")
        parser = OMMParser()
        records = parser.parse_file(str(xml_file))
        assert len(records) == 1
        assert records[0].norad_cat_id == 25544


# ── Database Storage Tests ──

class TestTLEStorage:
    """Database storage tests."""

    def test_store_tle_record(self, tmp_path):
        """Store a TLE record in SQLite."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        storage.store(records[0])

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT satellite_number, inclination FROM tles WHERE satellite_number = 25544")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 25544
        assert abs(row[1] - 51.6416) < 1e-6

    def test_store_multiple_tle_records(self, tmp_path):
        """Store multiple TLE records."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
            "1 28654U 05018A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 28654  99.0000 247.4627 0006703 130.5360 325.0288 14.20000000563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        for r in records:
            storage.store(r)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM tles")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2

    def test_query_tle_by_satellite_number(self, tmp_path):
        """Query TLE by satellite number."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        storage.store(records[0])

        result = storage.query_by_satellite_number(25544)
        assert result is not None
        assert result.satellite_number == 25544

    def test_query_tle_not_found_returns_none(self, tmp_path):
        """Query for non-existent satellite returns None."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        result = storage.query_by_satellite_number(99999)
        assert result is None

    def test_store_omm_record(self, tmp_path):
        """Store an OMM record in SQLite."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""
        parser = OMMParser()
        records = parser.parse(xml_text)
        storage.store(records[0])

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT norad_cat_id, object_name FROM omm WHERE norad_cat_id = 25544")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 25544
        assert row[1] == "ISS (ZARYA)"

    def test_delete_tle_record(self, tmp_path):
        """Delete a TLE record."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        storage.store(records[0])

        storage.delete(25544)
        result = storage.query_by_satellite_number(25544)
        assert result is None

    def test_update_tle_record(self, tmp_path):
        """Update a TLE record."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        tle_text = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records = parser.parse(tle_text)
        storage.store(records[0])

        # Update with new inclination
        records[0].inclination = 52.0
        storage.update(records[0])

        result = storage.query_by_satellite_number(25544)
        assert result is not None
        assert abs(result.inclination - 52.0) < 1e-6

    def test_store_duplicate_satellite_number_replaces(self, tmp_path):
        """Storing duplicate satellite number replaces existing record."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        tle_text1 = (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        tle_text2 = (
            "1 25544U 98067A   08265.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  52.0000 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        records1 = parser.parse(tle_text1)
        records2 = parser.parse(tle_text2)
        storage.store(records1[0])
        storage.store(records2[0])

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM tles WHERE satellite_number = 25544")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

        result = storage.query_by_satellite_number(25544)
        assert abs(result.inclination - 52.0) < 1e-6


# ── Integration Tests ──

class TestIntegration:
    """End-to-end integration tests."""

    def test_tle_parse_validate_store(self, tmp_path):
        """Full pipeline: parse, validate, store TLE."""
        db_path = str(tmp_path / "test.db")
        tle_text = (
            "ISS (ZARYA)\n"
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927\n"
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537\n"
        )
        parser = TLEParser()
        validator = TLEValidator()
        storage = TLEStorage(db_path)

        records = parser.parse(tle_text)
        valid_records = validator.validate_all(records)
        for r in valid_records:
            storage.store(r)

        result = storage.query_by_satellite_number(25544)
        assert result is not None
        assert result.satellite_number == 25544
        assert result.name == "ISS (ZARYA)"

    def test_omm_parse_store(self, tmp_path):
        """Full pipeline: parse and store OMM."""
        db_path = str(tmp_path / "test.db")
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""
        parser = OMMParser()
        storage = TLEStorage(db_path)

        records = parser.parse(xml_text)
        for r in records:
            storage.store(r)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT object_name FROM omm WHERE norad_cat_id = 25544")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "ISS (ZARYA)"

    def test_tle_checksum_calculation(self):
        """Verify TLE checksum calculation is correct."""
        # Line 1 checksum: sum of digits, minus signs count as 1, mod 10
        line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
        parser = TLEParser()
        assert parser._calculate_checksum(line1[:68]) == 7

    def test_omm_to_tle_conversion(self):
        """Convert OMM record to TLE-like format."""
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<omm id="v1.0" version="2.0">
  <header>
    <CREATION_DATE>2008-09-20T17:54:32</CREATION_DATE>
    <ORIGINATOR>18 SPCS</ORIGINATOR>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>ISS (ZARYA)</OBJECT_NAME>
        <OBJECT_ID>1998-067A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
      <data>
        <meanElements>
          <EPOCH>2008-09-20T00:00:00.00000000</EPOCH>
          <MEAN_MOTION>15.72125391</MEAN_MOTION>
          <ECCENTRICITY>0.0006703</ECCENTRICITY>
          <INCLINATION>51.6416</INCLINATION>
          <RA_OF_ASC_NODE>247.4627</RA_OF_ASC_NODE>
          <ARG_OF_PERICENTER>130.5360</ARG_OF_PERICENTER>
          <MEAN_ANOMALY>325.0288</MEAN_ANOMALY>
        </meanElements>
        <tleParameters>
          <EPHEMERIS_TYPE>0</EPHEMERIS_TYPE>
          <CLASSIFICATION>U</CLASSIFICATION>
          <NORAD_CAT_ID>25544</NORAD_CAT_ID>
          <ELEMENT_SET_NO>292</ELEMENT_SET_NO>
          <REV_AT_EPOCH>56353</REV_AT_EPOCH>
          <BSTAR>0.0000000000000000</BSTAR>
          <MEAN_MOTION_DOT>-0.00002182</MEAN_MOTION_DOT>
          <MEAN_MOTION_DDOT>0.0000000000000000</MEAN_MOTION_DDOT>
        </tleParameters>
      </data>
    </segment>
  </body>
</omm>"""
        parser = OMMParser()
        records = parser.parse(xml_text)
        assert len(records) == 1
        # Verify all fields are accessible
        r = records[0]
        assert r.object_name == "ISS (ZARYA)"
        assert r.norad_cat_id == 25544
        assert r.mean_motion == 15.72125391
        assert r.eccentricity == 0.0006703
        assert r.inclination == 51.6416
        assert r.ra_of_asc_node == 247.4627
        assert r.arg_of_pericenter == 130.5360
        assert r.mean_anomaly == 325.0288
        assert r.epoch == "2008-09-20T00:00:00.00000000"
        assert r.bstar == 0.0
        assert r.mean_motion_dot == -0.00002182
        assert r.mean_motion_ddot == 0.0
        assert r.classification == "U"
        assert r.element_set_no == 292
        assert r.rev_at_epoch == 56353
        assert r.ephemeris_type == 0
