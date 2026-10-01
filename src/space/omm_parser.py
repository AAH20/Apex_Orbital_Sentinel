"""OMM XML parsing, TLE validation, and database storage."""
from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


class OMMParseError(Exception):
    """Raised when OMM XML parsing fails."""


class ValidationError(Exception):
    """Raised when TLE validation fails."""


@dataclass
class TLERecord:
    """Represents a parsed Two-Line Element set."""

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


@dataclass
class OMMRecord:
    """Represents a parsed OMM (Orbit Mean-Elements Message) record."""

    object_name: str
    object_id: str
    center_name: str
    ref_frame: str
    time_system: str
    mean_element_theory: str
    epoch: str
    mean_motion: float
    eccentricity: float
    inclination: float
    ra_of_asc_node: float
    arg_of_pericenter: float
    mean_anomaly: float
    ephemeris_type: int
    classification: str
    norad_cat_id: int
    element_set_no: int
    rev_at_epoch: int
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    creation_date: str = ""
    originator: str = ""


class OMMParser:
    """Parses OMM (Orbit Mean-Elements Message) XML documents."""

    def parse(self, xml_text: str) -> List[OMMRecord]:
        """Parse OMM XML text and return list of OMMRecord objects."""
        if not xml_text or not xml_text.strip():
            raise OMMParseError("Empty OMM XML text")

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise OMMParseError(f"Invalid XML: {e}") from e

        if root.tag != "omm":
            raise OMMParseError(f"Expected root element 'omm', got '{root.tag}'")

        header = root.find("header")
        creation_date = ""
        originator = ""
        if header is not None:
            cd = header.find("CREATION_DATE")
            if cd is not None and cd.text:
                creation_date = cd.text.strip()
            orig = header.find("ORIGINATOR")
            if orig is not None and orig.text:
                originator = orig.text.strip()

        body = root.find("body")
        if body is None:
            raise OMMParseError("Missing <body> element")

        records: List[OMMRecord] = []
        for segment in body.findall("segment"):
            record = self._parse_segment(segment, creation_date, originator)
            records.append(record)

        if not records:
            raise OMMParseError("No segments found in OMM XML")

        return records

    def parse_file(self, filepath: str) -> List[OMMRecord]:
        """Parse OMM XML from a file path."""
        path = Path(filepath)
        if not path.exists():
            raise OMMParseError(f"File not found: {filepath}")
        text = path.read_text()
        return self.parse(text)

    def _parse_segment(self, segment: ET.Element, creation_date: str, originator: str) -> OMMRecord:
        """Parse a single OMM segment."""
        metadata = segment.find("metadata")
        if metadata is None:
            raise OMMParseError("Missing <metadata> in segment")

        data = segment.find("data")
        if data is None:
            raise OMMParseError("Missing <data> in segment")

        object_name = self._get_text(metadata, "OBJECT_NAME")
        object_id = self._get_text(metadata, "OBJECT_ID")
        center_name = self._get_text(metadata, "CENTER_NAME")
        ref_frame = self._get_text(metadata, "REF_FRAME")
        time_system = self._get_text(metadata, "TIME_SYSTEM")
        mean_element_theory = self._get_text(metadata, "MEAN_ELEMENT_THEORY")

        mean_elements = data.find("meanElements")
        if mean_elements is None:
            raise OMMParseError("Missing <meanElements> in data")

        tle_params = data.find("tleParameters")
        if tle_params is None:
            raise OMMParseError("Missing <tleParameters> in data")

        epoch = self._get_text(mean_elements, "EPOCH")
        mean_motion = self._get_float(mean_elements, "MEAN_MOTION")
        eccentricity = self._get_float(mean_elements, "ECCENTRICITY")
        inclination = self._get_float(mean_elements, "INCLINATION")
        ra_of_asc_node = self._get_float(mean_elements, "RA_OF_ASC_NODE")
        arg_of_pericenter = self._get_float(mean_elements, "ARG_OF_PERICENTER")
        mean_anomaly = self._get_float(mean_elements, "MEAN_ANOMALY")

        ephemeris_type = self._get_int(tle_params, "EPHEMERIS_TYPE")
        classification = self._get_text(tle_params, "CLASSIFICATION")
        norad_cat_id = self._get_int(tle_params, "NORAD_CAT_ID")
        element_set_no = self._get_int(tle_params, "ELEMENT_SET_NO")
        rev_at_epoch = self._get_int(tle_params, "REV_AT_EPOCH")
        bstar = self._get_float(tle_params, "BSTAR")
        mean_motion_dot = self._get_float(tle_params, "MEAN_MOTION_DOT")
        mean_motion_ddot = self._get_float(tle_params, "MEAN_MOTION_DDOT")

        return OMMRecord(
            object_name=object_name,
            object_id=object_id,
            center_name=center_name,
            ref_frame=ref_frame,
            time_system=time_system,
            mean_element_theory=mean_element_theory,
            epoch=epoch,
            mean_motion=mean_motion,
            eccentricity=eccentricity,
            inclination=inclination,
            ra_of_asc_node=ra_of_asc_node,
            arg_of_pericenter=arg_of_pericenter,
            mean_anomaly=mean_anomaly,
            ephemeris_type=ephemeris_type,
            classification=classification,
            norad_cat_id=norad_cat_id,
            element_set_no=element_set_no,
            rev_at_epoch=rev_at_epoch,
            bstar=bstar,
            mean_motion_dot=mean_motion_dot,
            mean_motion_ddot=mean_motion_ddot,
            creation_date=creation_date,
            originator=originator,
        )

    @staticmethod
    def _get_text(parent: ET.Element, tag: str) -> str:
        """Get text content of a child element."""
        elem = parent.find(tag)
        if elem is None or elem.text is None:
            raise OMMParseError(f"Missing required field: {tag}")
        return elem.text.strip()

    @staticmethod
    def _get_float(parent: ET.Element, tag: str) -> float:
        """Get float value of a child element."""
        text = OMMParser._get_text(parent, tag)
        try:
            return float(text)
        except ValueError as e:
            raise OMMParseError(f"Invalid float for {tag}: {text}") from e

    @staticmethod
    def _get_int(parent: ET.Element, tag: str) -> int:
        """Get int value of a child element."""
        text = OMMParser._get_text(parent, tag)
        try:
            return int(text)
        except ValueError as e:
            raise OMMParseError(f"Invalid int for {tag}: {text}") from e


class TLEValidator:
    """Validates TLE records."""

    def validate(self, record: TLERecord) -> bool:
        """Validate a single TLE record. Raises ValidationError on failure."""
        self._validate_checksums(record)
        self._validate_ranges(record)
        return True

    def validate_all(self, records: List[TLERecord]) -> List[TLERecord]:
        """Validate all records, return only valid ones."""
        valid: List[TLERecord] = []
        for record in records:
            try:
                self.validate(record)
                valid.append(record)
            except ValidationError:
                continue
        return valid

    def _validate_checksums(self, record: TLERecord) -> None:
        """Validate line checksums."""
        if record.raw_line1:
            expected = self._calculate_checksum(record.raw_line1[:68])
            if expected != record.line1_checksum:
                raise ValidationError(
                    f"Line 1 checksum mismatch: expected {expected}, got {record.line1_checksum}"
                )
        if record.raw_line2:
            expected = self._calculate_checksum(record.raw_line2[:68])
            if expected != record.line2_checksum:
                raise ValidationError(
                    f"Line 2 checksum mismatch: expected {expected}, got {record.line2_checksum}"
                )

    def _validate_ranges(self, record: TLERecord) -> None:
        """Validate orbital element ranges."""
        if not 0 <= record.inclination <= 180:
            raise ValidationError(f"Inclination out of range [0, 180]: {record.inclination}")
        if not 0 <= record.eccentricity < 1:
            raise ValidationError(f"Eccentricity out of range [0, 1): {record.eccentricity}")
        if record.mean_motion <= 0:
            raise ValidationError(f"Mean motion must be positive: {record.mean_motion}")

    @staticmethod
    def _calculate_checksum(line: str) -> int:
        """Calculate TLE checksum for a line (excluding the checksum character)."""
        total = 0
        for char in line:
            if char.isdigit():
                total += int(char)
            elif char == "-":
                total += 1
        return total % 10


class TLEStorage:
    """SQLite-backed storage for TLE and OMM records."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tles (
                    satellite_number INTEGER PRIMARY KEY,
                    classification TEXT,
                    international_designator TEXT,
                    epoch_year INTEGER,
                    epoch_day REAL,
                    mean_motion_derivative REAL,
                    mean_motion_sec_derivative REAL,
                    bstar_drag REAL,
                    ephemeris_type INTEGER,
                    element_set_number INTEGER,
                    line1_checksum INTEGER,
                    line2_checksum INTEGER,
                    inclination REAL,
                    raan REAL,
                    eccentricity REAL,
                    arg_of_perigee REAL,
                    mean_anomaly REAL,
                    mean_motion REAL,
                    revolution_number INTEGER,
                    name TEXT,
                    raw_line1 TEXT,
                    raw_line2 TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS omm (
                    norad_cat_id INTEGER PRIMARY KEY,
                    object_name TEXT,
                    object_id TEXT,
                    center_name TEXT,
                    ref_frame TEXT,
                    time_system TEXT,
                    mean_element_theory TEXT,
                    epoch TEXT,
                    mean_motion REAL,
                    eccentricity REAL,
                    inclination REAL,
                    ra_of_asc_node REAL,
                    arg_of_pericenter REAL,
                    mean_anomaly REAL,
                    ephemeris_type INTEGER,
                    classification TEXT,
                    element_set_no INTEGER,
                    rev_at_epoch INTEGER,
                    bstar REAL,
                    mean_motion_dot REAL,
                    mean_motion_ddot REAL,
                    creation_date TEXT,
                    originator TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def store(self, record: Union[TLERecord, OMMRecord]) -> None:
        """Store a TLE or OMM record."""
        if isinstance(record, TLERecord):
            self._store_tle(record)
        elif isinstance(record, OMMRecord):
            self._store_omm(record)
        else:
            raise TypeError(f"Unsupported record type: {type(record)}")

    def _store_tle(self, record: TLERecord) -> None:
        """Store a TLE record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tles (
                    satellite_number, classification, international_designator,
                    epoch_year, epoch_day, mean_motion_derivative, mean_motion_sec_derivative,
                    bstar_drag, ephemeris_type, element_set_number,
                    line1_checksum, line2_checksum,
                    inclination, raan, eccentricity, arg_of_perigee, mean_anomaly,
                    mean_motion, revolution_number, name, raw_line1, raw_line2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.satellite_number, record.classification, record.international_designator,
                    record.epoch_year, record.epoch_day, record.mean_motion_derivative,
                    record.mean_motion_sec_derivative, record.bstar_drag,
                    record.ephemeris_type, record.element_set_number,
                    record.line1_checksum, record.line2_checksum,
                    record.inclination, record.raan, record.eccentricity,
                    record.arg_of_perigee, record.mean_anomaly,
                    record.mean_motion, record.revolution_number,
                    record.name, record.raw_line1, record.raw_line2,
                ),
            )
            conn.commit()

    def _store_omm(self, record: OMMRecord) -> None:
        """Store an OMM record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO omm (
                    norad_cat_id, object_name, object_id, center_name, ref_frame,
                    time_system, mean_element_theory, epoch,
                    mean_motion, eccentricity, inclination, ra_of_asc_node,
                    arg_of_pericenter, mean_anomaly,
                    ephemeris_type, classification, element_set_no, rev_at_epoch,
                    bstar, mean_motion_dot, mean_motion_ddot,
                    creation_date, originator
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.norad_cat_id, record.object_name, record.object_id,
                    record.center_name, record.ref_frame, record.time_system,
                    record.mean_element_theory, record.epoch,
                    record.mean_motion, record.eccentricity, record.inclination,
                    record.ra_of_asc_node, record.arg_of_pericenter, record.mean_anomaly,
                    record.ephemeris_type, record.classification,
                    record.element_set_no, record.rev_at_epoch,
                    record.bstar, record.mean_motion_dot, record.mean_motion_ddot,
                    record.creation_date, record.originator,
                ),
            )
            conn.commit()

    def query_by_satellite_number(self, satellite_number: int) -> Optional[TLERecord]:
        """Query TLE by satellite number."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM tles WHERE satellite_number = ?", (satellite_number,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_tle(row)

    def query_omm_by_norad_id(self, norad_cat_id: int) -> Optional[OMMRecord]:
        """Query OMM by NORAD catalog ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM omm WHERE norad_cat_id = ?", (norad_cat_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_omm(row)

    def delete(self, satellite_number: int) -> None:
        """Delete a TLE record by satellite number."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tles WHERE satellite_number = ?", (satellite_number,))
            conn.commit()

    def update(self, record: TLERecord) -> None:
        """Update a TLE record."""
        self._store_tle(record)

    def _row_to_tle(self, row: sqlite3.Row) -> TLERecord:
        """Convert a database row to TLERecord."""
        return TLERecord(
            satellite_number=row["satellite_number"],
            classification=row["classification"],
            international_designator=row["international_designator"],
            epoch_year=row["epoch_year"],
            epoch_day=row["epoch_day"],
            mean_motion_derivative=row["mean_motion_derivative"],
            mean_motion_sec_derivative=row["mean_motion_sec_derivative"],
            bstar_drag=row["bstar_drag"],
            ephemeris_type=row["ephemeris_type"],
            element_set_number=row["element_set_number"],
            line1_checksum=row["line1_checksum"],
            line2_checksum=row["line2_checksum"],
            inclination=row["inclination"],
            raan=row["raan"],
            eccentricity=row["eccentricity"],
            arg_of_perigee=row["arg_of_perigee"],
            mean_anomaly=row["mean_anomaly"],
            mean_motion=row["mean_motion"],
            revolution_number=row["revolution_number"],
            name=row["name"] or "",
            raw_line1=row["raw_line1"] or "",
            raw_line2=row["raw_line2"] or "",
        )

    def _row_to_omm(self, row: sqlite3.Row) -> OMMRecord:
        """Convert a database row to OMMRecord."""
        return OMMRecord(
            object_name=row["object_name"],
            object_id=row["object_id"],
            center_name=row["center_name"],
            ref_frame=row["ref_frame"],
            time_system=row["time_system"],
            mean_element_theory=row["mean_element_theory"],
            epoch=row["epoch"],
            mean_motion=row["mean_motion"],
            eccentricity=row["eccentricity"],
            inclination=row["inclination"],
            ra_of_asc_node=row["ra_of_asc_node"],
            arg_of_pericenter=row["arg_of_pericenter"],
            mean_anomaly=row["mean_anomaly"],
            ephemeris_type=row["ephemeris_type"],
            classification=row["classification"],
            norad_cat_id=row["norad_cat_id"],
            element_set_no=row["element_set_no"],
            rev_at_epoch=row["rev_at_epoch"],
            bstar=row["bstar"],
            mean_motion_dot=row["mean_motion_dot"],
            mean_motion_ddot=row["mean_motion_ddot"],
            creation_date=row["creation_date"] or "",
            originator=row["originator"] or "",
        )
