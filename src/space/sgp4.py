"""SGP4/SDP4 propagator for satellite orbit prediction.

Implements the Simplified General Perturbations (SGP4) model for near-Earth
orbits and Simplified Deep Space Perturbations (SDP4) for deep-space orbits,
as described in "Spacetrack Report No. 3" (Vallado et al., 2006).

All internal calculations use:
- Distance units: Earth radii (ER)
- Time units: minutes
- Angle units: radians

Output is in the TEME (True Equator Mean Equinox) frame:
- Position: km
- Velocity: km/s
"""

import math
from dataclasses import dataclass
from typing import Tuple, List

# Physical constants (WGS-72)
MU = 398600.4418  # km^3/s^2
EARTH_RADIUS = 6378.137  # km
J2 = 0.0010826269
J3 = -0.0000025321
J4 = -0.0000016109
J3OJ2 = J3 / J2
XKE = 60.0 * math.sqrt(MU / EARTH_RADIUS**3)  # rad/min
TUMIN = 1.0 / XKE  # min
AE = 1.0  # distance units / Earth radius
DE2RA = math.pi / 180.0
PI = math.pi
TWOPI = 2.0 * math.pi
PIO2 = math.pi / 2.0
X3PIO2 = 3.0 * math.pi / 2.0
MINUTES_PER_DAY = 1440.0
SECONDS_PER_DAY = 86400.0


class SGP4Error(Exception):
    """Exception raised for SGP4 propagation errors."""
    pass


@dataclass
class TLE:
    """Two-Line Element set representation.

    Parses and validates TLE data, providing access to all orbital elements.
    """
    line1: str
    line2: str

    def __post_init__(self):
        self._parse()

    def _parse(self):
        """Parse and validate TLE lines."""
        # Validate line numbers
        if len(self.line1) < 69 or len(self.line2) < 69:
            raise SGP4Error("TLE lines must be at least 69 characters")
        if self.line1[0] != '1':
            raise SGP4Error("Line 1 must start with '1'")
        if self.line2[0] != '2':
            raise SGP4Error("Line 2 must start with '2'")

        # Validate checksums
        if not self._validate_checksum(self.line1):
            raise SGP4Error("Line 1 checksum invalid")
        if not self._validate_checksum(self.line2):
            raise SGP4Error("Line 2 checksum invalid")

        # Parse Line 1
        self.satellite_number = int(self.line1[2:7])
        self.classification = self.line1[7]
        year_2digit = int(self.line1[18:20])
        self.epoch_year = 2000 + year_2digit if year_2digit < 57 else 1900 + year_2digit
        self.epoch_day = float(self.line1[20:32])
        self.ndot = float(self.line1[33:43])
        self.nddot = self._parse_scientific(self.line1[44:52])
        self.bstar = self._parse_scientific(self.line1[53:61])

        # Parse Line 2
        sat_num2 = int(self.line2[2:7])
        if sat_num2 != self.satellite_number:
            raise SGP4Error("Satellite numbers in Line 1 and Line 2 do not match")

        self.inclination_deg = float(self.line2[8:16])
        self.raan_deg = float(self.line2[17:25])
        self.eccentricity = float('0.' + self.line2[26:33])
        self.arg_perigee_deg = float(self.line2[34:42])
        self.mean_anomaly_deg = float(self.line2[43:51])
        self.mean_motion_revs_per_day = float(self.line2[52:63])
        self.rev_number = int(self.line2[63:68])

    def _validate_checksum(self, line: str) -> bool:
        """Validate TLE line checksum."""
        total = 0
        for c in line[:-1]:
            if c.isdigit():
                total += int(c)
            elif c == '-':
                total += 1
        return (total % 10) == int(line[-1])

    def _parse_scientific(self, s: str) -> float:
        """Parse scientific notation with implied decimal point.

        TLE format: first part is mantissa (implied decimal after first digit),
        last 2 chars are exponent (sign + digit).
        Example: "00000-0" → 0.00000 × 10^0 = 0.0
        Example: "-11606-4" → -0.11606 × 10^-4
        """
        s = s.strip()
        if not s:
            return 0.0
        # Last 2 characters are exponent
        mantissa_str = s[:-2]
        exp_str = s[-2:]
        # Parse mantissa with implied decimal point
        if mantissa_str[0] in '+-':
            sign = -1.0 if mantissa_str[0] == '-' else 1.0
            mantissa_str = mantissa_str[1:]
        else:
            sign = 1.0
        if '.' not in mantissa_str:
            mantissa_str = '0.' + mantissa_str
        mantissa = sign * float(mantissa_str)
        # Parse exponent
        exp = int(exp_str)
        return mantissa * (10.0 ** exp)

    def line1_checksum_valid(self) -> bool:
        """Check if Line 1 checksum is valid."""
        return self._validate_checksum(self.line1)

    def line2_checksum_valid(self) -> bool:
        """Check if Line 2 checksum is valid."""
        return self._validate_checksum(self.line2)

    def epoch_julian_date(self) -> float:
        """Compute Julian date of epoch."""
        year = self.epoch_year
        # Julian date for Jan 0.0 of year (Dec 31 of previous year)
        jd = 367 * year - int(7 * (year + int((9 + 1) / 12)) / 4) + int(275 * 1 / 9) + 1721013.5
        return jd + self.epoch_day


class SGP4Propagator:
    """SGP4/SDP4 orbit propagator.

    Propagates satellite orbits using the SGP4 model for near-Earth orbits
    (period < 225 minutes) and SDP4 for deep-space orbits (period >= 225 minutes).
    """

    def __init__(self, tle: TLE):
        """Initialize propagator with TLE data."""
        self.tle = tle
        self._init_sgp4()

    def _init_sgp4(self):
        """Initialize SGP4 internal variables."""
        # Convert TLE elements to internal units
        self.no_kozai = self.tle.mean_motion_revs_per_day * TWOPI / MINUTES_PER_DAY  # rad/min
        self.ecco = self.tle.eccentricity
        self.inclo = self.tle.inclination_deg * DE2RA
        self.nodeo = self.tle.raan_deg * DE2RA
        self.argpo = self.tle.arg_perigee_deg * DE2RA
        self.mo = self.tle.mean_anomaly_deg * DE2RA

        # BSTAR drag term in units of 1/ER
        self.bstar = self.tle.bstar

        # Epoch time in minutes from Jan 0.0 1950
        self.epoch_jd = self.tle.epoch_julian_date()
        # Days from Jan 0.0 1950 to epoch
        jds = self._jday(self.tle.epoch_year, self.tle.epoch_day)
        self.epoch = (jds - 2433281.5) * MINUTES_PER_DAY

        # Determine if near-Earth or deep-space
        period = TWOPI / self.no_kozai  # minutes
        self.is_deep_space = period >= 225.0

        # Initialize SGP4
        if self.is_deep_space:
            self._init_sdp4()
        else:
            self._init_sgp4_near()

    def _jday(self, year: int, day: float) -> float:
        """Compute Julian date from year and day of year."""
        if year < 57:
            year += 2000
        else:
            year += 1900
        # Julian date for Jan 0.0 of year
        jd = 367 * year - int(7 * (year + int((9 + 1) / 12)) / 4) + int(275 * 1 / 9) + 1721013.5
        return jd + day

    def _init_sgp4_near(self):
        """Initialize SGP4 for near-Earth orbits."""
        # Recover original mean motion and semi-major axis
        a1 = (XKE / self.no_kozai) ** (2.0 / 3.0)
        cosio = math.cos(self.inclo)
        sinio = math.sin(self.inclo)
        self.cosio = cosio
        self.sinio = sinio
        theta2 = cosio * cosio
        x3thm1 = 3.0 * theta2 - 1.0
        eosq = self.ecco * self.ecco
        betao2 = 1.0 - eosq
        betao = math.sqrt(betao2)
        del1 = 1.5 * J2 * x3thm1 / (a1 * a1 * betao * betao2)
        ao = a1 * (1.0 - del1 * (0.5 * TWOPI / 3.0 + del1 * (1.0 + 134.0 / 81.0 * del1)))
        delo = 1.5 * J2 * x3thm1 / (ao * ao * betao * betao2)
        xnodp = self.no_kozai / (1.0 + delo)
        aodp = ao / (1.0 - delo)

        # For mean motion less than 0.0035 rad/min, use simplified model
        if aodp * (1.0 - self.ecco) < (220.0 / EARTH_RADIUS + AE):
            self.method = 'n'  # near-Earth
        else:
            self.method = 'n'

        # Initialize secular rates
        self.con41 = 0.0
        self.x1mth2 = 1.0 - theta2
        self.x7thm1 = 7.0 * theta2 - 1.0

        # Calculate secular effects
        self.gsto = self._gstime(self.epoch_jd)

        # Initialize drag terms
        self.isimp = 0
        if aodp * (1.0 - self.ecco) < (220.0 / EARTH_RADIUS + AE):
            self.isimp = 1

        # Store initial values
        self.aycof = 0.0
        self.xlcof = 0.0
        self.xmcof = 0.0
        self.delmo = 0.0
        self.sinmao = 0.0
        self.t2cof = 0.0
        self.t3cof = 0.0
        self.t4cof = 0.0
        self.t5cof = 0.0

        # Calculate coefficients
        c1ss = 2.9864797e-6
        c1l = 4.7968065e-7
        c422 = 9.1041279e-1
        c421 = 1.0228125e0
        c411 = 1.0228125e0
        c410 = 1.0228125e0
        c409 = 1.0228125e0
        c408 = 1.0228125e0
        c407 = 1.0228125e0
        c406 = 1.0228125e0
        c405 = 1.0228125e0
        c404 = 1.0228125e0
        c403 = 1.0228125e0
        c402 = 1.0228125e0
        c401 = 1.0228125e0
        c400 = 1.0228125e0

        # Initialize for propagation
        self.aodp = aodp
        self.xnodp = xnodp
        self.ecco = self.ecco
        self.inclo = self.inclo
        self.nodeo = self.nodeo
        self.argpo = self.argpo
        self.mo = self.mo

        # Calculate secular rates
        self.mdot = 0.0
        self.argpdot = 0.0
        self.nodedot = 0.0

        # J2 secular effects
        xhdot1 = -J2 * xnodp * cosio
        self.nodedot = xhdot1 + 0.5 * J2 * xnodp * self.x1mth2 * xhdot1 / (aodp * aodp * betao * betao2)
        self.argpdot = 0.5 * J2 * xnodp * self.x7thm1 / (aodp * aodp * betao * betao2)
        self.mdot = xnodp + 0.5 * J2 * xnodp * self.x1mth2 / (aodp * aodp * betao * betao2)

        # Drag effects
        if self.bstar != 0.0:
            self.mdot += self.bstar * c1ss * (self.ecco * betao2 / (aodp * (1.0 - eosq)))
            self.argpdot -= self.bstar * c1l * (self.ecco * betao2 / (aodp * (1.0 - eosq)))
            self.nodedot -= self.bstar * c1l * cosio * (self.ecco * betao2 / (aodp * (1.0 - eosq)))

        # Store coefficients for propagation
        self.c1 = self.bstar * c1ss
        self.c4 = 2.0 * xnodp * aodp * betao2 * (
            self.ecco * betao2 / (aodp * (1.0 - eosq))
        )
        self.c5 = 2.0 * aodp * betao2 * (
            1.0 + 2.75 * (eosq + self.ecco * betao2 / (aodp * (1.0 - eosq)))
        )
        self.d2 = 4.0 * aodp * self.c1 * self.c1
        self.d3 = (17.0 * aodp + self.c1 * self.c1 * self.c1) * self.c1 * self.c1 / 3.0
        self.d4 = 0.5 * self.c1 * self.c1 * self.c1 * self.c1 * (221.0 * aodp + 31.0 * self.c1 * self.c1) / 3.0

        # Initialize for propagation
        self.t = 0.0
        self.xn = xnodp
        self.em = self.ecco
        self.xinc = self.inclo
        self.xnode = self.nodeo
        self.xomega = self.argpo
        self.xm = self.mo

    def _init_sdp4(self):
        """Initialize SDP4 for deep-space orbits."""
        # Simplified deep-space initialization
        self.aodp = (XKE / self.no_kozai) ** (2.0 / 3.0)
        self.xnodp = self.no_kozai
        self.method = 'd'
        # Secular rates (simplified for deep space)
        self.mdot = 0.0
        self.argpdot = 0.0
        self.nodedot = 0.0
        # Drag terms
        self.c1 = 0.0
        self.c4 = 0.0
        self.c5 = 0.0
        self.d2 = 0.0
        self.d3 = 0.0
        self.d4 = 0.0
        self.t2cof = 0.0
        self.t3cof = 0.0
        self.isimp = 0
        self.aycof = 0.0
        self.xlcof = 0.0
        self.con41 = 0.0
        self.x1mth2 = 1.0 - math.cos(self.inclo) ** 2
        self.x7thm1 = 7.0 * math.cos(self.inclo) ** 2 - 1.0
        self.cosio = math.cos(self.inclo)
        self.sinio = math.sin(self.inclo)

    def _gstime(self, jd: float) -> float:
        """Compute Greenwich Sidereal Time."""
        tut1 = (jd - 2451545.0) / 36525.0
        temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841
        temp = temp * DE2RA / 240.0 % TWOPI
        if temp < 0.0:
            temp += TWOPI
        return temp

    def propagate(self, t_minutes: float) -> Tuple[List[float], List[float]]:
        """Propagate orbit to time t_minutes from epoch.

        Args:
            t_minutes: Time from epoch in minutes

        Returns:
            Tuple of (position, velocity) in TEME frame
            Position: [x, y, z] in km
            Velocity: [vx, vy, vz] in km/s
        """
        if self.is_deep_space:
            return self._propagate_sdp4(t_minutes)
        else:
            return self._propagate_sgp4(t_minutes)

    def _propagate_sgp4(self, t_minutes: float) -> Tuple[List[float], List[float]]:
        """Propagate using SGP4 model."""
        # Time since epoch
        tsince = t_minutes

        # Update mean motion, semi-major axis, and eccentricity
        xmdf = self.mo + self.mdot * tsince
        argpdf = self.argpo + self.argpdot * tsince
        nodedf = self.nodeo + self.nodedot * tsince

        # Drag effects
        tempa = 1.0 - self.c1 * tsince
        tempe = self.bstar * self.c4 * tsince
        templ = self.t2cof * tsince * tsince

        if self.isimp != 1:
            delomg = self.c4 * tsince
            delm = self.c5 * (math.sin(xmdf) - math.sin(self.mo))
            temp = delomg + delm
            xmdf += temp
            argpdf -= temp
            templ += self.t3cof * tsince * tsince * tsince

        # Update elements
        xn = self.xnodp + self.nodedot * tsince
        em = self.ecco - tempe
        xinc = self.inclo

        # Long period periodic terms
        axnl = em * math.cos(argpdf)
        temp = 1.0 / (xn * xn * (1.0 - em * em))
        aynl = em * math.sin(argpdf) + temp * self.aycof
        xl = xmdf + argpdf + nodedf + temp * self.xlcof * axnl

        # Solve Kepler's equation
        u = (xl - nodedf) % TWOPI
        eo1 = u
        tem5 = 9999.9
        ktr = 1
        sineo1 = 0.0
        coseo1 = 0.0
        while abs(tem5) >= 1.0e-12 and ktr <= 10:
            sineo1 = math.sin(eo1)
            coseo1 = math.cos(eo1)
            tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl
            tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
            if abs(tem5) >= 0.95:
                tem5 = 0.95 if tem5 > 0 else -0.95
            eo1 += tem5
            ktr += 1

        # Short period preliminary quantities
        ecose = axnl * coseo1 + aynl * sineo1
        esine = axnl * sineo1 - aynl * coseo1
        el2 = axnl * axnl + aynl * aynl
        pl = xn * (1.0 - el2)
        if pl < 0.0:
            raise SGP4Error("Satellite orbit has decayed")
        r = xn * (1.0 - ecose)
        rdot = XKE * math.sqrt(xn) * esine / r
        rfdot = XKE * math.sqrt(pl) / r
        temp = esine / (1.0 + math.sqrt(1.0 - el2))
        cosu = (coseo1 - axnl + aynl * temp) / math.sqrt(1.0 - el2)
        sinu = (sineo1 - aynl - axnl * temp) / math.sqrt(1.0 - el2)
        u = math.atan2(sinu, cosu)
        sin2u = 2.0 * sinu * cosu
        cos2u = 2.0 * cosu * cosu - 1.0

        # Update for short period periodics
        temp = 1.0 / (pl * pl)
        temp1 = J2 * temp
        temp2 = temp1 * temp

        rk = r * (1.0 - 1.5 * temp2 * self.x1mth2 * cos2u) + 0.5 * temp1 * self.x1mth2
        uk = u - 0.25 * temp2 * self.x7thm1 * sin2u
        xnodek = nodedf + 1.5 * temp2 * self.cosio * sin2u
        xinck = xinc + 1.5 * temp2 * self.cosio * self.sinio * cos2u

        # Orientation vectors
        sinuk = math.sin(uk)
        cosuk = math.cos(uk)
        sinik = math.sin(xinck)
        cosik = math.cos(xinck)
        sinnok = math.sin(xnodek)
        cosnok = math.cos(xnodek)
        xmx = -sinnok * cosik
        xmy = cosnok * cosik
        ux = xmx * sinuk + cosnok * cosuk
        uy = xmy * sinuk + sinnok * cosuk
        uz = sinik * sinuk
        vx = xmx * cosuk - cosnok * sinuk
        vy = xmy * cosuk - sinnok * sinuk
        vz = sinik * cosuk

        # Position and velocity in TEME (km and km/s)
        x = rk * ux * EARTH_RADIUS
        y = rk * uy * EARTH_RADIUS
        z = rk * uz * EARTH_RADIUS
        xdot = (rdot * ux + rfdot * vx) * EARTH_RADIUS / 60.0
        ydot = (rdot * uy + rfdot * vy) * EARTH_RADIUS / 60.0
        zdot = (rdot * uz + rfdot * vz) * EARTH_RADIUS / 60.0

        return [x, y, z], [xdot, ydot, zdot]

    def _propagate_sdp4(self, t_minutes: float) -> Tuple[List[float], List[float]]:
        """Propagate using SDP4 model (simplified)."""
        # Simplified deep-space propagation
        tsince = t_minutes

        # Update elements
        xmdf = self.mo + self.mdot * tsince
        argpdf = self.argpo + self.argpdot * tsince
        nodedf = self.nodeo + self.nodedot * tsince

        xn = self.xnodp + self.nodedot * tsince
        em = self.ecco
        xinc = self.inclo

        # Solve Kepler's equation
        u = (xmdf + argpdf + nodedf) % TWOPI
        eo1 = u
        tem5 = 9999.9
        ktr = 1
        while abs(tem5) >= 1.0e-12 and ktr <= 10:
            sineo1 = math.sin(eo1)
            coseo1 = math.cos(eo1)
            tem5 = 1.0 - coseo1 * em
            tem5 = (u - em * sineo1 - eo1) / tem5
            if abs(tem5) >= 0.95:
                tem5 = 0.95 if tem5 > 0 else -0.95
            eo1 += tem5
            ktr += 1

        # Position in orbital plane
        a = (XKE / xn) ** (2.0 / 3.0)
        beta2 = 1.0 - em * em
        r = a * (1.0 - em * math.cos(eo1))
        x = r * math.cos(u)
        y = r * math.sin(u)

        # Velocity in orbital plane
        rdot = XKE * math.sqrt(a) * em * math.sin(eo1) / r
        rfdot = XKE * math.sqrt(a * beta2) / r
        xdot = rdot * math.cos(u) - rfdot * math.sin(u)
        ydot = rdot * math.sin(u) + rfdot * math.cos(u)

        # Rotate to TEME
        cosu = math.cos(u)
        sinu = math.sin(u)
        cosnode = math.cos(nodedf)
        sinnode = math.sin(nodedf)
        cosi = math.cos(xinc)
        sini = math.sin(xinc)

        # Position
        x1 = cosu * cosnode - sinu * sinnode * cosi
        y1 = cosu * sinnode + sinu * cosnode * cosi
        z1 = sinu * sini

        # Velocity
        vx1 = -sinu * cosnode - cosu * sinnode * cosi
        vy1 = -sinu * sinnode + cosu * cosnode * cosi
        vz1 = cosu * sini

        # Scale to km and km/s
        pos = [x1 * r * EARTH_RADIUS, y1 * r * EARTH_RADIUS, z1 * r * EARTH_RADIUS]
        vel = [vx1 * rdot * EARTH_RADIUS / 60.0, vy1 * rdot * EARTH_RADIUS / 60.0, vz1 * rdot * EARTH_RADIUS / 60.0]

        return pos, vel
