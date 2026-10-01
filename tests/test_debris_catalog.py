"""Tests for debris_catalog: catalog management, trajectory prediction, collision risk."""
import math
from datetime import datetime, timedelta

import pytest

from src.space.debris_catalog import (
    DebrisObject,
    DebrisCatalog,
    TrajectoryPredictor,
    RiskAssessor,
    RiskLevel,
    ConjunctionEvent,
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

    def test_update_object(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", name="OLD"))
        updated = make_debris(norad_id="1", name="NEW")
        catalog.update(updated)
        assert catalog.get("1").name == "NEW"

    def test_update_nonexistent_raises(self):
        catalog = DebrisCatalog()
        with pytest.raises(KeyError):
            catalog.update(make_debris(norad_id="999"))

    def test_search_by_name(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", name="COSMOS-1234"))
        catalog.add(make_debris(norad_id="2", name="DEBRIS-A"))
        catalog.add(make_debris(norad_id="3", name="COSMOS-5678"))
        results = catalog.search_by_name("COSMOS")
        assert len(results) == 2

    def test_merge_catalogs(self):
        c1 = DebrisCatalog()
        c1.add(make_debris(norad_id="1"))
        c1.add(make_debris(norad_id="2"))
        c2 = DebrisCatalog()
        c2.add(make_debris(norad_id="3"))
        c1.merge(c2)
        assert c1.count() == 3

    def test_merge_with_conflict_raises(self):
        c1 = DebrisCatalog()
        c1.add(make_debris(norad_id="1"))
        c2 = DebrisCatalog()
        c2.add(make_debris(norad_id="1"))
        with pytest.raises(ValueError, match="conflict"):
            c1.merge(c2)

    def test_export_import_roundtrip(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", name="A"))
        catalog.add(make_debris(norad_id="2", name="B"))
        data = catalog.export_to_dict()
        new_catalog = DebrisCatalog.import_from_dict(data)
        assert new_catalog.count() == 2
        assert new_catalog.get("1").name == "A"
        assert new_catalog.get("2").name == "B"

    def test_filter_by_inclination(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", inclination_deg=30.0))
        catalog.add(make_debris(norad_id="2", inclination_deg=60.0))
        catalog.add(make_debris(norad_id="3", inclination_deg=90.0))
        result = catalog.filter_by_inclination(45.0, 75.0)
        assert len(result) == 1
        assert result[0].norad_id == "2"


# ─── TrajectoryPredictor Tests ────────────────────────────────────────────────

class TestTrajectoryPredictor:
    def test_predict_position_at_epoch(self):
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

    def test_orbital_period(self):
        """Orbital period for a=7000 km should be ~97 minutes."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(semi_major_axis_km=7000.0))
        predictor = TrajectoryPredictor(catalog)
        period_s = predictor.orbital_period("12345")
        # T = 2*pi*sqrt(a^3/mu) ≈ 5828 s ≈ 97 min
        assert 5700 < period_s < 6000

    def test_apogee_perigee(self):
        """Apogee and perigee for a=7000, e=0.1."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(semi_major_axis_km=7000.0, eccentricity=0.1))
        predictor = TrajectoryPredictor(catalog)
        apogee, perigee = predictor.apogee_perigee("12345")
        assert abs(apogee - 7700.0) < 1.0
        assert abs(perigee - 6300.0) < 1.0

    def test_predict_position_changes_over_time(self):
        """Position at epoch should differ from position one orbit later."""
        catalog = DebrisCatalog()
        catalog.add(make_debris(semi_major_axis_km=7000.0, eccentricity=0.0))
        predictor = TrajectoryPredictor(catalog)
        epoch = datetime(2026, 1, 1)
        pos1 = predictor.predict_position("12345", epoch)
        period_s = predictor.orbital_period("12345")
        pos2 = predictor.predict_position("12345", epoch + timedelta(seconds=period_s))
        # After one full orbit, should be back near start
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
        assert dist < 1.0


# ─── RiskAssessor Tests ───────────────────────────────────────────────────────

class TestRiskAssessor:
    def test_compute_closest_approach_identical_objects(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        dist = assessor.compute_closest_approach("1", "2", datetime(2026, 1, 1))
        assert dist < 1.0

    def test_compute_closest_approach_different_objects(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=8000.0))
        assessor = RiskAssessor(catalog)
        dist = assessor.compute_closest_approach("1", "2", datetime(2026, 1, 1))
        assert dist > 100.0

    def test_assess_risk_low(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        risk = assessor.assess_risk("1", "2", datetime(2026, 1, 1))
        assert risk == RiskLevel.LOW

    def test_assess_risk_critical(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        risk = assessor.assess_risk("1", "2", datetime(2026, 1, 1))
        assert risk == RiskLevel.CRITICAL

    def test_find_conjunctions(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=7001.0))
        catalog.add(make_debris(norad_id="3", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        conjunctions = assessor.find_conjunctions(datetime(2026, 1, 1), threshold_km=10.0)
        assert len(conjunctions) == 1
        pair = conjunctions[0]
        assert (pair[0] == "1" and pair[1] == "2") or (pair[0] == "2" and pair[1] == "1")

    def test_find_conjunctions_empty(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        conjunctions = assessor.find_conjunctions(datetime(2026, 1, 1), threshold_km=1.0)
        assert len(conjunctions) == 0

    def test_compute_collision_probability_bounds(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        prob = assessor.compute_collision_probability("1", "2", datetime(2026, 1, 1))
        assert 0.0 <= prob <= 1.0

    def test_compute_collision_probability_near_zero_for_distant(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        prob = assessor.compute_collision_probability("1", "2", datetime(2026, 1, 1))
        assert prob < 0.01

    def test_risk_score_is_numeric(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1"))
        catalog.add(make_debris(norad_id="2"))
        assessor = RiskAssessor(catalog)
        score = assessor.risk_score("1", "2", datetime(2026, 1, 1))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_conjunction_event_dataclass(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=7005.0))
        assessor = RiskAssessor(catalog)
        event = assessor.compute_conjunction_event("1", "2", datetime(2026, 1, 1))
        assert isinstance(event, ConjunctionEvent)
        assert event.distance_km > 0
        assert event.sat1_id == "1"
        assert event.sat2_id == "2"

    def test_find_high_risk_pairs(self):
        catalog = DebrisCatalog()
        catalog.add(make_debris(norad_id="1", semi_major_axis_km=7000.0))
        catalog.add(make_debris(norad_id="2", semi_major_axis_km=7001.0))
        catalog.add(make_debris(norad_id="3", semi_major_axis_km=9000.0))
        assessor = RiskAssessor(catalog)
        high_risk = assessor.find_high_risk_pairs(datetime(2026, 1, 1), threshold_km=10.0)
        assert len(high_risk) == 1
