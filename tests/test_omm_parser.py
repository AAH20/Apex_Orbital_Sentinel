"""Tests for omm_parser module — OMM XML parsing, TLE validation, database storage."""
import os
import sys
import pytest
import tempfile
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from space.omm_parser import (
    OMMParser,
    OMMRecord,
    TLEValidator,
    TLEStorage,
    TLERecord,
    OMMParseError,
    ValidationError,
)


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

    def test_parse_omm_xml_header_fields(self):
        """OMM header fields are correctly parsed."""
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
        assert records[0].creation_date == "2008-09-20T17:54:32"
        assert records[0].originator == "18 SPCS"

    def test_parse_omm_xml_wrong_root_raises(self):
        """OMM XML with wrong root element raises OMMParseError."""
        xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<not_omm id="v1.0" version="2.0">
  <body>
  </body>
</not_omm>"""
        parser = OMMParser()
        with pytest.raises(OMMParseError):
            parser.parse(xml_text)

    def test_parse_omm_xml_empty_raises(self):
        """Empty OMM XML raises OMMParseError."""
        parser = OMMParser()
        with pytest.raises(OMMParseError):
            parser.parse("")


# ── TLE Validation Tests ──

class TestTLEValidation:
    """TLE validation tests."""

    def _make_tle_record(self, **kwargs):
        """Helper to create a TLERecord with default valid values."""
        defaults = dict(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=264.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=51.6416,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        defaults.update(kwargs)
        return TLERecord(**defaults)

    def test_valid_tle_passes(self):
        """Valid TLE passes validation."""
        record = self._make_tle_record()
        validator = TLEValidator()
        assert validator.validate(record) is True

    def test_invalid_checksum_line1_fails(self):
        """Invalid line 1 checksum fails validation."""
        record = self._make_tle_record(line1_checksum=8)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(record)

    def test_invalid_checksum_line2_fails(self):
        """Invalid line 2 checksum fails validation."""
        record = self._make_tle_record(line2_checksum=8)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(record)

    def test_inclination_out_of_range_fails(self):
        """Inclination > 180 degrees fails validation."""
        record = self._make_tle_record(inclination=181.0)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(record)

    def test_eccentricity_out_of_range_fails(self):
        """Eccentricity >= 1.0 fails validation."""
        record = self._make_tle_record(eccentricity=1.0)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(record)

    def test_mean_motion_out_of_range_fails(self):
        """Mean motion <= 0 fails validation."""
        record = self._make_tle_record(mean_motion=0.0)
        validator = TLEValidator()
        with pytest.raises(ValidationError):
            validator.validate(record)

    def test_validate_multiple_records(self):
        """Validate multiple records, return list of valid ones."""
        valid_record = self._make_tle_record()
        invalid_record = self._make_tle_record(satellite_number=99999, line1_checksum=99)
        validator = TLEValidator()
        valid = validator.validate_all([valid_record, invalid_record])
        assert len(valid) == 1
        assert valid[0].satellite_number == 25544


# ── Database Storage Tests ──

class TestTLEStorage:
    """Database storage tests."""

    def test_store_tle_record(self, tmp_path):
        """Store a TLE record in SQLite."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        record = TLERecord(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=264.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=51.6416,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        storage.store(record)

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
        for sat_num in [25544, 28654]:
            record = TLERecord(
                satellite_number=sat_num,
                classification="U",
                international_designator="98067A",
                epoch_year=8,
                epoch_day=264.51782528,
                mean_motion_derivative=-0.00002182,
                mean_motion_sec_derivative=0.0,
                bstar_drag=-0.11606e-4,
                ephemeris_type=0,
                element_set_number=292,
                line1_checksum=7,
                line2_checksum=7,
                inclination=51.6416,
                raan=247.4627,
                eccentricity=0.0006703,
                arg_of_perigee=130.5360,
                mean_anomaly=325.0288,
                mean_motion=15.72125391,
                revolution_number=56353,
                name="TEST",
                raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
                raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
            )
            storage.store(record)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM tles")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2

    def test_query_tle_by_satellite_number(self, tmp_path):
        """Query TLE by satellite number."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        record = TLERecord(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=264.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=51.6416,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        storage.store(record)

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
        record = OMMRecord(
            object_name="ISS (ZARYA)",
            object_id="1998-067A",
            center_name="EARTH",
            ref_frame="TEME",
            time_system="UTC",
            mean_element_theory="SGP4",
            epoch="2008-09-20T00:00:00.00000000",
            mean_motion=15.72125391,
            eccentricity=0.0006703,
            inclination=51.6416,
            ra_of_asc_node=247.4627,
            arg_of_pericenter=130.5360,
            mean_anomaly=325.0288,
            ephemeris_type=0,
            classification="U",
            norad_cat_id=25544,
            element_set_no=292,
            rev_at_epoch=56353,
            bstar=0.0,
            mean_motion_dot=-0.00002182,
            mean_motion_ddot=0.0,
            creation_date="2008-09-20T17:54:32",
            originator="18 SPCS",
        )
        storage.store(record)

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
        record = TLERecord(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=264.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=51.6416,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        storage.store(record)

        storage.delete(25544)
        result = storage.query_by_satellite_number(25544)
        assert result is None

    def test_update_tle_record(self, tmp_path):
        """Update a TLE record."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        record = TLERecord(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=264.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=51.6416,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        storage.store(record)

        # Update with new inclination
        record.inclination = 52.0
        storage.update(record)

        result = storage.query_by_satellite_number(25544)
        assert result is not None
        assert abs(result.inclination - 52.0) < 1e-6

    def test_store_duplicate_satellite_number_replaces(self, tmp_path):
        """Storing duplicate satellite number replaces existing record."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        record1 = TLERecord(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=264.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=51.6416,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        record2 = TLERecord(
            satellite_number=25544,
            classification="U",
            international_designator="98067A",
            epoch_year=8,
            epoch_day=265.51782528,
            mean_motion_derivative=-0.00002182,
            mean_motion_sec_derivative=0.0,
            bstar_drag=-0.11606e-4,
            ephemeris_type=0,
            element_set_number=292,
            line1_checksum=7,
            line2_checksum=7,
            inclination=52.0,
            raan=247.4627,
            eccentricity=0.0006703,
            arg_of_perigee=130.5360,
            mean_anomaly=325.0288,
            mean_motion=15.72125391,
            revolution_number=56353,
            name="ISS (ZARYA)",
            raw_line1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            raw_line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
        )
        storage.store(record1)
        storage.store(record2)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM tles WHERE satellite_number = 25544")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

        result = storage.query_by_satellite_number(25544)
        assert abs(result.inclination - 52.0) < 1e-6

    def test_query_omm_by_norad_id(self, tmp_path):
        """Query OMM by NORAD catalog ID."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        record = OMMRecord(
            object_name="ISS (ZARYA)",
            object_id="1998-067A",
            center_name="EARTH",
            ref_frame="TEME",
            time_system="UTC",
            mean_element_theory="SGP4",
            epoch="2008-09-20T00:00:00.00000000",
            mean_motion=15.72125391,
            eccentricity=0.0006703,
            inclination=51.6416,
            ra_of_asc_node=247.4627,
            arg_of_pericenter=130.5360,
            mean_anomaly=325.0288,
            ephemeris_type=0,
            classification="U",
            norad_cat_id=25544,
            element_set_no=292,
            rev_at_epoch=56353,
            bstar=0.0,
            mean_motion_dot=-0.00002182,
            mean_motion_ddot=0.0,
        )
        storage.store(record)

        result = storage.query_omm_by_norad_id(25544)
        assert result is not None
        assert result.object_name == "ISS (ZARYA)"

    def test_query_omm_not_found_returns_none(self, tmp_path):
        """Query for non-existent OMM returns None."""
        db_path = str(tmp_path / "test.db")
        storage = TLEStorage(db_path)
        result = storage.query_omm_by_norad_id(99999)
        assert result is None


# ── Integration Tests ──

class TestIntegration:
    """End-to-end integration tests."""

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

    def test_omm_all_fields_accessible(self):
        """Verify all OMM record fields are accessible after parsing."""
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
