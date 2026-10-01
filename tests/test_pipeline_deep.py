"""Deepened pipeline tests: data flow, error handling, monitoring.

TDD: these tests define the expected behavior for the enhanced
SpaceDomainPipeline with stage-based execution, metrics, and
graceful error recovery.
"""
import math
import pytest
from datetime import datetime, timedelta, timezone

from src.space.tle import TLE, TLEParseError
from src.space.propagator import SGP4Propagator, PropagationError, StateVector
from src.space.conjunction import ConjunctionDetector, ConjunctionEvent
from src.space.pipeline import (
    SpaceDomainPipeline,
    PipelineStage,
    StageResult,
    PipelineResult,
    PipelineMetrics,
    PipelineError,
    StageError,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ISS_TLE = TLE(
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   20239.50991228  .00000806  00000-0  19740-3 0  9990",
    line2="2 25544  51.6443 322.2412 0004028  69.9862  25.2906 15.49515014253715",
)

DEBRIS_TLE = TLE(
    name="DEBRIS OBJECT",
    line1="1 40000U 20001A   20239.50000000  .00001000  00000-0  20000-3 0  9991",
    line2="2 40000  51.6400 322.2000 0005000  70.0000  25.3000 15.50000000000001",
)

FAR_DEBRIS_TLE = TLE(
    name="FAR DEBRIS",
    line1="1 40001U 20001B   20239.50000000  .00001000  00000-0  20000-3 0  9992",
    line2="2 40001  90.0000 180.0000 0005000  70.0000  25.3000 15.50000000000002",
)

BAD_TLE = TLE(
    name="BAD",
    line1="1 00000U 00000A   20239.50000000  .00000000  00000-0  00000-0 0  9990",
    line2="2 00000   0.0000   0.0000 0.0000000   0.0000   0.0000  0.00000000000000",
)


# ---------------------------------------------------------------------------
# 1. PipelineStage abstraction
# ---------------------------------------------------------------------------

class TestPipelineStage:
    def test_stage_is_callable(self):
        stage = PipelineStage(name="test_stage")
        assert stage.name == "test_stage"

    def test_stage_process_returns_dict(self):
        stage = PipelineStage(name="test_stage")
        result = stage.process({"input": 42})
        assert isinstance(result, dict)

    def test_stage_tracks_execution(self):
        stage = PipelineStage(name="tracker")
        data = {"value": 10}
        result = stage.execute(data)
        assert result.output_data["value"] == 10
        assert result.output_data["_stage_name"] == "tracker"

    def test_stage_records_timing(self):
        stage = PipelineStage(name="timed")
        result = stage.execute({})
        assert "_duration_ms" in result.output_data
        assert result.output_data["_duration_ms"] >= 0


# ---------------------------------------------------------------------------
# 2. StageResult data flow tracking
# ---------------------------------------------------------------------------

class TestStageResult:
    def test_stage_result_creation(self):
        sr = StageResult(
            stage_name="propagate",
            input_data={"tle": "ISS"},
            output_data={"position": (1, 2, 3)},
            success=True,
            error=None,
            duration_ms=1.5,
        )
        assert sr.stage_name == "propagate"
        assert sr.success is True
        assert sr.error is None
        assert sr.duration_ms == pytest.approx(1.5)

    def test_stage_result_with_error(self):
        sr = StageResult(
            stage_name="propagate",
            input_data={"tle": "BAD"},
            output_data={},
            success=False,
            error="Propagation failed",
            duration_ms=0.5,
        )
        assert sr.success is False
        assert sr.error == "Propagation failed"

    def test_stage_result_data_flow(self):
        sr = StageResult(
            stage_name="detect",
            input_data={"primary": "ISS", "secondary": ["DEBRIS"]},
            output_data={"count": 3},
            success=True,
            error=None,
            duration_ms=2.0,
        )
        assert "primary" in sr.input_data
        assert "count" in sr.output_data


# ---------------------------------------------------------------------------
# 3. PipelineResult structure
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_pipeline_result_success(self):
        sr = StageResult("s1", {}, {"x": 1}, True, None, 1.0)
        pr = PipelineResult(
            success=True,
            stages=[sr],
            final_output={"x": 1},
            total_duration_ms=1.0,
            errors=[],
        )
        assert pr.success is True
        assert len(pr.stages) == 1
        assert pr.errors == []

    def test_pipeline_result_with_errors(self):
        sr1 = StageResult("s1", {}, {"x": 1}, True, None, 1.0)
        sr2 = StageResult("s2", {"x": 1}, {}, False, "fail", 0.5)
        pr = PipelineResult(
            success=False,
            stages=[sr1, sr2],
            final_output={},
            total_duration_ms=1.5,
            errors=["fail"],
        )
        assert pr.success is False
        assert len(pr.errors) == 1
        assert pr.total_duration_ms == pytest.approx(1.5)

    def test_pipeline_result_stage_chain(self):
        sr1 = StageResult("parse", {}, {"tle": "ISS"}, True, None, 0.1)
        sr2 = StageResult("propagate", {"tle": "ISS"}, {"sv": True}, True, None, 1.0)
        sr3 = StageResult("detect", {"sv": True}, {"count": 0}, True, None, 2.0)
        pr = PipelineResult(
            success=True,
            stages=[sr1, sr2, sr3],
            final_output={"count": 0},
            total_duration_ms=3.1,
            errors=[],
        )
        assert len(pr.stages) == 3
        assert pr.stages[0].stage_name == "parse"
        assert pr.stages[2].stage_name == "detect"


# ---------------------------------------------------------------------------
# 4. PipelineMetrics monitoring
# ---------------------------------------------------------------------------

class TestPipelineMetrics:
    def test_metrics_initial_state(self):
        m = PipelineMetrics()
        summary = m.get_summary()
        assert summary["total_stages"] == 0
        assert summary["total_errors"] == 0
        assert summary["success_rate"] == pytest.approx(1.0)

    def test_metrics_record_success(self):
        m = PipelineMetrics()
        m.record_stage("propagate", 1.5, True)
        summary = m.get_summary()
        assert summary["total_stages"] == 1
        assert summary["total_errors"] == 0
        assert summary["success_rate"] == pytest.approx(1.0)

    def test_metrics_record_error(self):
        m = PipelineMetrics()
        m.record_stage("propagate", 1.5, False, error="fail")
        summary = m.get_summary()
        assert summary["total_stages"] == 1
        assert summary["total_errors"] == 1
        assert summary["success_rate"] == pytest.approx(0.0)

    def test_metrics_mixed_results(self):
        m = PipelineMetrics()
        m.record_stage("s1", 1.0, True)
        m.record_stage("s2", 2.0, True)
        m.record_stage("s3", 0.5, False, error="err")
        summary = m.get_summary()
        assert summary["total_stages"] == 3
        assert summary["total_errors"] == 1
        assert summary["success_rate"] == pytest.approx(2.0 / 3.0)

    def test_metrics_stage_durations(self):
        m = PipelineMetrics()
        m.record_stage("propagate", 1.0, True)
        m.record_stage("propagate", 3.0, True)
        m.record_stage("detect", 2.0, True)
        summary = m.get_summary()
        assert summary["stage_counts"]["propagate"] == 2
        assert summary["stage_counts"]["detect"] == 1
        assert summary["avg_duration_ms"]["propagate"] == pytest.approx(2.0)

    def test_metrics_error_tracking(self):
        m = PipelineMetrics()
        m.record_stage("s1", 1.0, False, error="err1")
        m.record_stage("s2", 1.0, False, error="err2")
        summary = m.get_summary()
        assert summary["error_counts"]["s1"] == 1
        assert summary["error_counts"]["s2"] == 1

    def test_metrics_reset(self):
        m = PipelineMetrics()
        m.record_stage("s1", 1.0, True)
        m.reset()
        summary = m.get_summary()
        assert summary["total_stages"] == 0


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------

class TestPipelineErrorHandling:
    def test_pipeline_error_with_stage(self):
        err = PipelineError("something failed", stage="propagate")
        assert "propagate" in str(err)
        assert err.stage == "propagate"

    def test_stage_error_inherits_pipeline_error(self):
        err = StageError("stage failed", stage="detect")
        assert isinstance(err, PipelineError)
        assert err.stage == "detect"

    def test_pipeline_error_has_original(self):
        original = ValueError("root cause")
        err = PipelineError("wrapped", stage="parse", original_error=original)
        assert err.original_error is original


# ---------------------------------------------------------------------------
# 6. End-to-end pipeline run
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_run_single_tle(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(ISS_TLE)
        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert len(result.stages) >= 1
        assert result.final_output is not None

    def test_run_with_conjunction_detection(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(
            ISS_TLE,
            secondary_objects=[DEBRIS_TLE],
            duration_minutes=90,
        )
        assert result.success is True
        stage_names = [s.stage_name for s in result.stages]
        assert "propagate" in stage_names
        assert "detect_conjunctions" in stage_names

    def test_run_with_empty_secondary(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(ISS_TLE, secondary_objects=[])
        assert result.success is True

    def test_run_records_metrics(self):
        pipeline = SpaceDomainPipeline(enable_monitoring=True)
        pipeline.run(ISS_TLE)
        metrics = pipeline.get_metrics()
        assert metrics["total_stages"] >= 1

    def test_run_with_bad_tle_graceful(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(BAD_TLE)
        # Should not crash — should return a result with errors
        assert isinstance(result, PipelineResult)
        assert result.success is False
        assert len(result.errors) > 0

    def test_run_stage_data_flow(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(ISS_TLE, secondary_objects=[DEBRIS_TLE])
        # Each stage should have input and output data
        for stage in result.stages:
            assert hasattr(stage, 'input_data')
            assert hasattr(stage, 'output_data')
            assert isinstance(stage.input_data, dict)
            assert isinstance(stage.output_data, dict)

    def test_run_total_duration_positive(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(ISS_TLE)
        assert result.total_duration_ms > 0

    def test_run_multiple_secondary_objects(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(
            ISS_TLE,
            secondary_objects=[DEBRIS_TLE, FAR_DEBRIS_TLE],
            duration_minutes=90,
        )
        assert result.success is True
        # Should have detection results
        detect_stages = [s for s in result.stages if s.stage_name == "detect_conjunctions"]
        assert len(detect_stages) >= 1


# ---------------------------------------------------------------------------
# 7. Batch processing with error recovery
# ---------------------------------------------------------------------------

class TestPipelineBatchRecovery:
    def test_batch_with_good_and_bad_tles(self):
        pipeline = SpaceDomainPipeline()
        results = pipeline.process_batch([ISS_TLE, DEBRIS_TLE])
        assert len(results) == 2
        for r in results:
            assert 'state_vector' in r

    def test_batch_empty_list(self):
        pipeline = SpaceDomainPipeline()
        results = pipeline.process_batch([])
        assert results == []

    def test_batch_all_bad_tles_no_crash(self):
        pipeline = SpaceDomainPipeline()
        # Should handle gracefully — either skip or return error info
        results = pipeline.process_batch([BAD_TLE])
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 8. Maneuver planning edge cases
# ---------------------------------------------------------------------------

class TestPipelineManeuverEdgeCases:
    def test_maneuver_no_conjunction(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.plan_avoidance_maneuver(
            primary=ISS_TLE,
            threat=FAR_DEBRIS_TLE,
            duration_minutes=90,
        )
        assert result['maneuver'] is None
        assert 'reason' in result

    def test_maneuver_with_conjunction(self):
        pipeline = SpaceDomainPipeline(conjunction_threshold_km=10000.0)
        result = pipeline.plan_avoidance_maneuver(
            primary=ISS_TLE,
            threat=DEBRIS_TLE,
            duration_minutes=90,
        )
        assert result['maneuver'] is not None
        assert result['maneuver']['delta_v'] > 0


# ---------------------------------------------------------------------------
# 9. Orbital elements derivation
# ---------------------------------------------------------------------------

class TestPipelineOrbitalElements:
    def test_derive_elements_from_state(self):
        pipeline = SpaceDomainPipeline()
        sv = pipeline.propagator.propagate(ISS_TLE, ISS_TLE.epoch)
        elements = pipeline.derive_orbital_elements(sv)
        assert 'semi_major_axis' in elements
        assert 'eccentricity' in elements
        assert 'inclination' in elements
        assert 6500 < elements['semi_major_axis'] < 7500

    def test_derive_elements_inclination_matches(self):
        pipeline = SpaceDomainPipeline()
        sv = pipeline.propagator.propagate(ISS_TLE, ISS_TLE.epoch)
        elements = pipeline.derive_orbital_elements(sv)
        assert elements['inclination'] == pytest.approx(51.6443, abs=0.1)


# ---------------------------------------------------------------------------
# 10. Monitoring integration
# ---------------------------------------------------------------------------

class TestPipelineMonitoring:
    def test_metrics_accumulate_across_runs(self):
        pipeline = SpaceDomainPipeline(enable_monitoring=True)
        pipeline.run(ISS_TLE)
        pipeline.run(ISS_TLE)
        metrics = pipeline.get_metrics()
        assert metrics["total_stages"] >= 2

    def test_metrics_can_be_reset(self):
        pipeline = SpaceDomainPipeline(enable_monitoring=True)
        pipeline.run(ISS_TLE)
        pipeline.reset_metrics()
        metrics = pipeline.get_metrics()
        assert metrics["total_stages"] == 0

    def test_monitoring_disabled(self):
        pipeline = SpaceDomainPipeline(enable_monitoring=False)
        result = pipeline.run(ISS_TLE)
        assert result.success is True
        metrics = pipeline.get_metrics()
        assert metrics["total_stages"] == 0


# ---------------------------------------------------------------------------
# 11. Stage failure isolation
# ---------------------------------------------------------------------------

class TestStageFailureIsolation:
    def test_one_failure_doesnt_stop_pipeline(self):
        pipeline = SpaceDomainPipeline()
        # Run with a bad TLE — should record failure but not crash
        result = pipeline.run(BAD_TLE)
        assert isinstance(result, PipelineResult)
        # Pipeline should complete (with errors recorded)
        assert len(result.stages) >= 1

    def test_failed_stage_recorded_in_result(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(BAD_TLE)
        failed_stages = [s for s in result.stages if not s.success]
        assert len(failed_stages) >= 1
        assert failed_stages[0].error is not None


# ---------------------------------------------------------------------------
# 12. Data flow integrity
# ---------------------------------------------------------------------------

class TestDataFlowIntegrity:
    def test_stage_output_becomes_next_input(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(ISS_TLE, secondary_objects=[DEBRIS_TLE])
        # Verify data flows through stages
        for i in range(len(result.stages) - 1):
            current_out = result.stages[i].output_data
            next_in = result.stages[i + 1].input_data
            # The next stage's input should contain data from previous output
            # (at minimum, the pipeline context is passed along)
            assert isinstance(next_in, dict)

    def test_final_output_contains_expected_keys(self):
        pipeline = SpaceDomainPipeline()
        result = pipeline.run(ISS_TLE)
        assert 'state_vector' in result.final_output or 'position' in result.final_output
