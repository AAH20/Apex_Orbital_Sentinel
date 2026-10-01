"""Full space domain awareness pipeline.

Deepened with:
- Stage-based execution model (PipelineStage)
- Data flow tracking (StageResult, PipelineResult)
- Error handling and graceful recovery (PipelineError, StageError)
- Monitoring and metrics (PipelineMetrics)
"""
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.space.tle import TLE
from src.space.propagator import SGP4Propagator, StateVector
from src.space.conjunction import ConjunctionDetector, ConjunctionEvent


# ── Error Handling ──────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(self, message: str, stage: str = "", original_error: Exception = None):
        self.stage = stage
        self.original_error = original_error
        super().__init__(message)

    def __str__(self):
        if self.stage:
            return f"[{self.stage}] {super().__str__()}"
        return super().__str__()


class StageError(PipelineError):
    """Raised when a pipeline stage fails."""

    def __init__(self, message: str, stage: str, original_error: Exception = None):
        super().__init__(message, stage=stage, original_error=original_error)


# ── Monitoring ──────────────────────────────────────────────────────────────


class PipelineMetrics:
    """Collects and reports pipeline execution metrics."""

    def __init__(self):
        self._stage_counts: Dict[str, int] = {}
        self._stage_durations: Dict[str, List[float]] = {}
        self._error_counts: Dict[str, int] = {}
        self._total_stages = 0
        self._total_errors = 0

    def record_stage(self, name: str, duration_ms: float, success: bool, error: str = None):
        """Record a stage execution."""
        self._total_stages += 1
        self._stage_counts[name] = self._stage_counts.get(name, 0) + 1
        self._stage_durations.setdefault(name, []).append(duration_ms)
        if not success:
            self._total_errors += 1
            self._error_counts[name] = self._error_counts.get(name, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Return a metrics summary."""
        avg_durations = {}
        for name, durations in self._stage_durations.items():
            avg_durations[name] = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_stages": self._total_stages,
            "total_errors": self._total_errors,
            "success_rate": (self._total_stages - self._total_errors) / self._total_stages if self._total_stages > 0 else 1.0,
            "stage_counts": dict(self._stage_counts),
            "avg_duration_ms": avg_durations,
            "error_counts": dict(self._error_counts),
        }

    def reset(self):
        """Reset all metrics."""
        self._stage_counts.clear()
        self._stage_durations.clear()
        self._error_counts.clear()
        self._total_stages = 0
        self._total_errors = 0


# ── Data Flow Tracking ──────────────────────────────────────────────────────


class StageResult:
    """Tracks data flow through a single pipeline stage."""

    def __init__(
        self,
        stage_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool,
        error: Optional[str],
        duration_ms: float,
    ):
        self.stage_name = stage_name
        self.input_data = input_data
        self.output_data = output_data
        self.success = success
        self.error = error
        self.duration_ms = duration_ms


class PipelineResult:
    """Aggregated result from a full pipeline run."""

    def __init__(
        self,
        success: bool,
        stages: List[StageResult],
        final_output: Dict[str, Any],
        total_duration_ms: float,
        errors: List[str],
    ):
        self.success = success
        self.stages = stages
        self.final_output = final_output
        self.total_duration_ms = total_duration_ms
        self.errors = errors


# ── Pipeline Stage Abstraction ──────────────────────────────────────────────


class PipelineStage:
    """A single stage in the pipeline with timing and data flow tracking."""

    def __init__(self, name: str):
        self.name = name

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return output. Override in subclasses."""
        return data

    def execute(self, data: Dict[str, Any]) -> StageResult:
        """Execute the stage with timing and error capture."""
        start = time.perf_counter()
        try:
            output = self.process(data)
            duration_ms = (time.perf_counter() - start) * 1000.0
            output["_stage_name"] = self.name
            output["_duration_ms"] = duration_ms
            return StageResult(
                stage_name=self.name,
                input_data=data,
                output_data=output,
                success=True,
                error=None,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return StageResult(
                stage_name=self.name,
                input_data=data,
                output_data={"_stage_name": self.name, "_duration_ms": duration_ms},
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )


# ── Concrete Stages ─────────────────────────────────────────────────────────


class PropagateStage(PipelineStage):
    """Stage: propagate TLE to state vector."""

    def __init__(self, propagator: SGP4Propagator):
        super().__init__("propagate")
        self.propagator = propagator

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tle = data["tle"]
        when = data.get("when", tle.epoch)
        sv = self.propagator.propagate(tle, when)
        return {
            "state_vector": {
                "position": sv.position,
                "velocity": sv.velocity,
                "time": sv.time,
            },
            "name": tle.name,
            "satellite_number": tle.satellite_number,
        }


class DetectConjunctionsStage(PipelineStage):
    """Stage: detect conjunctions between primary and secondary objects."""

    def __init__(self, detector: ConjunctionDetector):
        super().__init__("detect_conjunctions")
        self.detector = detector

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        primary = data["primary"]
        secondary_objects = data.get("secondary_objects", [])
        duration_minutes = data.get("duration_minutes", 90)

        all_events = []
        for secondary in secondary_objects:
            events = self.detector.detect(
                primary, secondary,
                start=primary.epoch,
                duration_minutes=duration_minutes,
            )
            all_events.extend(events)

        return {
            "conjunctions": all_events,
            "count": len(all_events),
            "primary": primary.name,
        }


class DeriveElementsStage(PipelineStage):
    """Stage: derive orbital elements from a state vector."""

    def __init__(self, propagator: SGP4Propagator):
        super().__init__("derive_elements")
        self.propagator = propagator

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sv_data = data["state_vector"]
        sv = StateVector(
            position=sv_data["position"],
            velocity=sv_data["velocity"],
            time=sv_data["time"],
        )
        elements = self._derive_orbital_elements(sv)
        return {"orbital_elements": elements}

    def _derive_orbital_elements(self, sv: StateVector) -> Dict[str, float]:
        """Derive orbital elements from a state vector."""
        x, y, z = sv.position
        vx, vy, vz = sv.velocity

        r = math.sqrt(x**2 + y**2 + z**2)
        v = math.sqrt(vx**2 + vy**2 + vz**2)

        # Specific angular momentum
        hx = y * vz - z * vy
        hy = z * vx - x * vz
        hz = x * vy - y * vx
        h = math.sqrt(hx**2 + hy**2 + hz**2)

        # Node vector
        nx = -hy
        ny = hx
        n = math.sqrt(nx**2 + ny**2)

        # Eccentricity vector
        rv = x * vx + y * vy + z * vz
        mu = self.propagator.MU
        ex = (v**2 - mu / r) * x / mu - rv * vx / mu
        ey = (v**2 - mu / r) * y / mu - rv * vy / mu
        ez = (v**2 - mu / r) * z / mu - rv * vz / mu
        ecc = math.sqrt(ex**2 + ey**2 + ez**2)

        # Semi-major axis
        energy = v**2 / 2 - mu / r
        a = -mu / (2 * energy)

        # Inclination
        inc = math.degrees(math.acos(hz / h))

        return {
            "semi_major_axis": a,
            "eccentricity": ecc,
            "inclination": inc,
            "specific_angular_momentum": h,
            "orbital_energy": energy,
        }


# ── Main Pipeline ───────────────────────────────────────────────────────────


class SpaceDomainPipeline:
    """End-to-end space domain awareness pipeline.

    Supports stage-based execution with data flow tracking,
    error handling, and monitoring.
    """

    def __init__(
        self,
        conjunction_threshold_km: float = 10.0,
        enable_monitoring: bool = True,
    ):
        self.propagator = SGP4Propagator()
        self.detector = ConjunctionDetector(threshold_km=conjunction_threshold_km)
        self._enable_monitoring = enable_monitoring
        self._metrics = PipelineMetrics()

        # Build stages
        self._propagate_stage = PropagateStage(self.propagator)
        self._detect_stage = DetectConjunctionsStage(self.detector)
        self._derive_stage = DeriveElementsStage(self.propagator)

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics summary."""
        return self._metrics.get_summary()

    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics.reset()

    def run(
        self,
        tle: TLE,
        when: datetime = None,
        secondary_objects: List[TLE] = None,
        duration_minutes: float = 90,
    ) -> PipelineResult:
        """Execute the full pipeline with stage tracking and error handling.

        Args:
            tle: Primary TLE to process.
            when: Propagation time (defaults to TLE epoch).
            secondary_objects: List of secondary TLEs for conjunction detection.
            duration_minutes: Conjunction detection window.

        Returns:
            PipelineResult with stage chain, data flow, and error info.
        """
        if secondary_objects is None:
            secondary_objects = []

        stages: List[StageResult] = []
        errors: List[str] = []
        total_start = time.perf_counter()

        # Stage 1: Propagate
        stage_input = {"tle": tle, "when": when or tle.epoch}
        result = self._propagate_stage.execute(stage_input)
        stages.append(result)
        if self._enable_monitoring:
            self._metrics.record_stage(
                result.stage_name, result.duration_ms, result.success, result.error
            )
        if not result.success:
            errors.append(f"propagate: {result.error}")
            total_ms = (time.perf_counter() - total_start) * 1000.0
            return PipelineResult(
                success=False,
                stages=stages,
                final_output={},
                total_duration_ms=total_ms,
                errors=errors,
            )

        # Stage 2: Derive orbital elements
        stage_input = {"state_vector": result.output_data["state_vector"]}
        result = self._derive_stage.execute(stage_input)
        stages.append(result)
        if self._enable_monitoring:
            self._metrics.record_stage(
                result.stage_name, result.duration_ms, result.success, result.error
            )
        if not result.success:
            errors.append(f"derive_elements: {result.error}")

        # Stage 3: Detect conjunctions (if secondary objects provided)
        if secondary_objects:
            stage_input = {
                "primary": tle,
                "secondary_objects": secondary_objects,
                "duration_minutes": duration_minutes,
            }
            result = self._detect_stage.execute(stage_input)
            stages.append(result)
            if self._enable_monitoring:
                self._metrics.record_stage(
                    result.stage_name, result.duration_ms, result.success, result.error
                )
            if not result.success:
                errors.append(f"detect_conjunctions: {result.error}")

        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Build final output from stage results
        final_output = {}
        for sr in stages:
            if sr.success:
                final_output.update(sr.output_data)

        return PipelineResult(
            success=len(errors) == 0,
            stages=stages,
            final_output=final_output,
            total_duration_ms=total_ms,
            errors=errors,
        )

    def process_single(self, tle: TLE, when: datetime = None) -> dict:
        """Process a single TLE: propagate and return state.

        Legacy method for backward compatibility.
        """
        if when is None:
            when = tle.epoch
        sv = self.propagator.propagate(tle, when)
        return {
            "name": tle.name,
            "satellite_number": tle.satellite_number,
            "state_vector": {
                "position": sv.position,
                "velocity": sv.velocity,
                "time": sv.time,
            },
        }

    def process_batch(self, tles: list, when: datetime = None) -> list:
        """Process multiple TLEs with error recovery.

        Legacy method for backward compatibility.
        """
        results = []
        for tle in tles:
            try:
                results.append(self.process_single(tle, when))
            except Exception:
                # Skip TLEs that fail to propagate
                continue
        return results

    def analyze_conjunctions(
        self,
        primary: TLE,
        secondary_objects: list,
        duration_minutes: float = 90,
    ) -> dict:
        """Analyze conjunctions between primary and secondary objects.

        Legacy method for backward compatibility.
        """
        all_events = []
        for secondary in secondary_objects:
            events = self.detector.detect(
                primary, secondary,
                start=primary.epoch,
                duration_minutes=duration_minutes,
            )
            all_events.extend(events)
        return {
            "primary": primary.name,
            "conjunctions": all_events,
            "count": len(all_events),
        }

    def plan_avoidance_maneuver(
        self,
        primary: TLE,
        threat: TLE,
        duration_minutes: float = 90,
    ) -> dict:
        """Plan an avoidance maneuver.

        Legacy method for backward compatibility.
        """
        events = self.detector.detect(
            primary, threat, start=primary.epoch, duration_minutes=duration_minutes
        )
        if not events:
            return {"maneuver": None, "reason": "No conjunction detected"}

        sv = self.propagator.propagate(primary, primary.epoch)
        v = math.sqrt(sum(x**2 for x in sv.velocity))
        delta_v = 2 * v * math.sin(math.radians(0.1) / 2)

        return {
            "maneuver": {
                "delta_v": delta_v,
                "burn_time": 60.0,
                "direction": "normal",
                "purpose": "Avoid conjunction with " + threat.name,
            },
            "conjunction_count": len(events),
        }

    def derive_orbital_elements(self, sv: StateVector) -> dict:
        """Derive orbital elements from a state vector.

        Legacy method for backward compatibility.
        """
        return self._derive_stage._derive_orbital_elements(sv)
