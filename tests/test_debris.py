"""Tests for orbital debris tracking: catalog, trajectory prediction, risk assessment."""
import math
from datetime import datetime, timedelta

import pytest

from src.space.debris import (
    DebrisObject,
    DebrisCatalog,
    TrajectoryPredictor,
    RiskAssessor,
    RiskLevel,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_debris(
    norad_id="12345",
    name="DEBRIS-1",
    size_m=0.5,
    mass_kg=10.0,
    semi_major_axis_km=7000.0,
    eccentricity=0.001,
    inclination_deg=51.6,
    raan_deg=100.0,
    arg_periapsis_deg=200.0,
    mean_anomaly_deg=0.0,
    epoch=None,
    object_type="debris",
):
    if epoch is None:
        epoch = datetime(2026, 1, 1, 0, 0, 0)
    return DebrisObject(
        norad_id=norad_id,
        name=name,
        size_m=size_m,
        mass_kg=mass_kg,
        semi_major_axis_km=semi_major_axis_km,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        arg_periapsis_deg=arg_periapsis_deg,
        mean_anomaly_deg=mean_anomaly_deg,
        epoch=epoch,
        object_type=object_type,
    )


# ─── DebrisCatalog Tests ──────────────────────────────────────────────────────

class TestDebrisCatalog:
    def test_add_and_get(self):
        catalog = DebrisCatalog()
        obj = make_debris()
        catalog.add(obj)
        assert catalog.get("12345") == obj

    def test_add_duplicate_raises(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris())
        with pytest.raises(ValueError, match="already exists"):
            catalog.add(make_debris())

    def test_remove(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris())
        catalog.remove("12345")
        assert catalog.count() == 0

    def test_remove_nonexistent_raises(self):
        catalog = DebrisCatalog()
        with pytest.raises(KeyError):
            catalog.remove("99999")

    def test_get_nonexistent_raises(self):
        catalog = DebrisCatalog()
        with pytest.raises(KeyError):
            catalog.get("99999")

    def test_list_all(self):
        catalog = DebrisCatalog()
        obj1 = make_debris(norad_id="11111", name="A")
        obj2 = make_debris(norad_id="22222", name="B")
        catalog.add(obj1)
        catalog.add(obj2)
        result = catalog.list_all()
        assert len(result) == 2
        assert obj1 in result
        assert obj2 in result

    def test_filter_by_type(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", object_type="debris"))
        catalog.add(make_debris(norad_id="2", object_type="satellite"))
        catalog.add(make_debris(norad_id="3", object_type="debris"))
        debris = catalog.filter_by_type("debris")
        assert len(debris) == 2
        assert all(d.object_type == "debris" for d in debris)

    def test_filter_by_size(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", size_m=0.1))
        catalog.add(make_debris(norad_id="2", size_m=1.0))
        catalog.add(make_debris(norad_id="3", size_m=5.0))
        result = catalog.filter_by_size(0.5, 2.0)
        assert len(result) == 1
        assert result[0].norad_id == "2"

    def test_count(self):
        catalog = DebrisCatalog()
        assert catalog.count() == 0
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assert catalog.count() == 2


# ─── TrajectoryPredictor Tests ────────────────────────────────────────────────

class TestTrajectoryPredictor:
    def test_predict_position_at_epoch(self):
        """At epoch, position should match the initial state vector."""
        catalog = DebrisCatalog()
        obj = make_debris(
            semi_major_axis_km=7000.0,
            eccentricity=0.0,
            mean_anomaly_deg=0.0,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            inclination_deg=0.0,
        )
        catalog.add(obj)
        predictor = TrajectoryPredictor(catalog)
        pos = predictor.predict_position("12345", obj.epoch)
        # At epoch with zero angles, position is (a, 0, 0)
        assert abs(pos[0] - 7000.0) < 1.0
        assert abs(pos[1]) < 1.0
        assert abs(pos[2]) < 1.0

    def test_predict_position_returns_tuple_of_three(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris())
        predictor = TrajectoryPredictor(catalog)
        pos = predictor.predict_position("12345", datetime(2026, 1, 1))
        assert isinstance(pos, tuple)
        assert len(pos) == 3

    def test_predict_trajectory_length(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris())
        predictor = TrajectoryPredictor(catalog)
        start = datetime(2026, 1, 1)
        end = start + timedelta(hours=1)
        traj = predictor.predict_trajectory("12345", start, end, step_seconds=600)
        # 1 hour / 10 min = 6 steps + 1 for endpoint = 7
        assert len(traj) == 7

    def test_predict_trajectory_positions_are_3d(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris())
        predictor = TrajectoryPredictor(catalog)
        start = datetime(2026, 1, 1)
        traj = predictor.predict_trajectory("12345", start, start + timedelta(minutes=30), step_seconds=600)
        for pos in traj:
            assert len(pos) == 3
            assert all(isinstance(v, float) for v in pos)

    def test_predict_position_unknown_id_raises(self):
        catalog = DebrisCatalog()
        predictor = TrajectoryPredictor(catalog)
        with pytest.raises(KeyError):
            predictor.predict_position("99999", datetime(2026, 1, 1))


# ─── RiskAssessor Tests ───────────────────────────────────────────────────────

class TestRiskAssessor:
    def test_compute_closest_approach_identical_objects(self):
        """Two identical objects at same position have zero distance."""
        catalog = DebrisCatalog()
        obj = make_debris(norad_id="1")
        catalog.add(obj)
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        dist = assessor.compute_closest_approach("1", "2", datetime(2026, 1, 1))
        assert dist < 1.0  # essentially zero

    def test_compute_closest_approach_different_objects(self):
        """Two objects at different semi-major axes have non-zero distance."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=8000.0))
        assessor = RiskAssessor(catalog)
        dist = assessor.compute_closest_approach("1", "2", datetime(2026, 1, 1))
        assert dist > 100.0

    def test_assess_risk_low(self):
        """Objects far apart -> LOW risk."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        risk = assessor.assess_risk("1", "2", datetime(2026, 1, 1))
        assert risk == RiskLevel.LOW

    def test_assess_risk_critical(self):
        """Objects at same position -> CRITICAL risk."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        risk = assessor.assess_risk("1", "2", datetime(2026, 1, 1))
        assert risk == RiskLevel.CRITICAL

    def test_find_conjunctions(self):
        """Find pairs within threshold distance."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=7001.0))
        catalog.add(make_debris(norad_id="3", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        conjunctions = assessor.find_conjunctions(datetime(2026, 1, 1), threshold_km=10.0)
        # Only pair (1,2) should be within 10 km
        assert len(conjunctions) == 1
        pair = conjunctions[0]
        assert (pair[0] == "1" and pair[1] == "2") or (pair[0] == "2" and pair[1] == "1")

    def test_find_conjunctions_empty(self):
        """No conjunctions when all objects are far apart."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        conjunctions = assessor.find_conjunctions(datetime(2026, 1, 1), threshold_km=1.0)
        assert len(conjunctions) == 0

    def test_compute_collision_probability_bounds(self):
        """Collision probability must be between 0 and 1."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        prob = assessor.compute_collision_probability("1", "2", datetime(2026, 1, 1))
        assert 0.0 <= prob <= 1.0
