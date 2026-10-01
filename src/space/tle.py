"""TLE (Two-Line Element) parsing and representation."""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


class TLEParseError(Exception):
    """Raised when TLE parsing fails."""
    pass


@dataclass
class TLE:
    """Two-Line Element set representation."""
    name: str
    line1: str
    line2: str

    @classmethod
    def from_lines(cls, line1: str, line2: str, name: str = "UNKNOWN") -> "TLE":
        """Create TLE from two lines."""
        line1 = line1.strip()
        line2 = line2.strip()
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise TLEParseError("TLE lines must start with '1 ' and '2 '")
        if len(line1) < 69 or len(line2) < 69:
            raise TLEParseError("TLE lines must be at least 69 characters")
        return cls(name=name, line1=line1, line2=line2)

    @property
    def satellite_number(self) -> int:
        return int(self.line2[2:7])

    @property
    def classification(self) -> str:
        return self.line1[7]

    @property
    def epoch_year(self) -> int:
        year = int(self.line1[18:20])
        return 2000 + year if year < 57 else 1900 + year

    @property
    def epoch_day(self) -> float:
        return float(self.line1[20:32])

    @property
    def epoch(self) -> datetime:
        """Convert epoch year/day to datetime."""
        year_start = datetime(self.epoch_year, 1, 1, tzinfo=timezone.utc)
        return year_start + timedelta(days=self.epoch_day - 1)

    @property
    def mean_motion(self) -> float:
        """Mean motion in revolutions per day."""
        return float(self.line2[52:63])

    @property
    def period_seconds(self) -> float:
        """Orbital period in seconds."""
        return 86400.0 / self.mean_motion

    @property
    def inclination(self) -> float:
        """Inclination in degrees."""
        return float(self.line2[8:16])

    @property
    def raan(self) -> float:
        """Right Ascension of Ascending Node in degrees."""
        return float(self.line2[17:25])

    @property
    def eccentricity(self) -> float:
        """Eccentricity (leading decimal implied)."""
        return float("0." + self.line2[26:33].strip())

    @property
    def arg_perigee(self) -> float:
        """Argument of perigee in degrees."""
        return float(self.line2[34:42])

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly in degrees."""
        return float(self.line2[43:51])

    @property
    def revolution_number(self) -> int:
        """Revolution number at epoch."""
        return int(self.line1[63:68])
