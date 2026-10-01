"""Unit tests for Space C2 — tasking optimization, resource allocation, multi-satellite coordination."""

import unittest
import math
from datetime import datetime, timezone

from src.space.tasking import (
    ResourceAllocator,
    Satellite,
    SatelliteCoordinator,
    SatelliteStatus,
    TaskingOptimizer,
    TaskingPlan,
    TaskingRequest,
    TaskType,
)


def _make_satellite(
    sat_id: str,
    name: str = "SAT",
    capabilities=None,
    power_available=100.0,
    bandwidth_available=1000.0,
    memory_available=1000.0,
    sensor_time_available=60.0,
    lat=0.0,
    lon=0.0,
    status=SatelliteStatus.OPERATIONAL,
):
    """Helper to create a satellite with sensible defaults."""
    return Satellite(
        sat_id=sat_id,
        name=name,
        status=status,
        capabilities=capabilities or {"imaging", "tracking"},
        power_capacity=100.0,
        power_available=power_available,
        bandwidth_capacity=1000.0,
        bandwidth_available=bandwidth_available,
        memory_capacity=1000.0,
        memory_available=memory_available,
        sensor_time_capacity=60.0,
        sensor_time_available=sensor_time_available,
        current_lat=lat,
        current_lon=lon,
    )


def _make_request(
    request_id,
    task_type=TaskType.IMAGING,
    priority=5,
    capabilities=None,
    power=10.0,
    bandwidth=50.0,
    memory=10.0,
    sensor_time=5.0,
    lat=None,
    lon=None,
    dependencies=None,
):
    """Helper to create a tasking request with sensible defaults."""
    return TaskingRequest(
        request_id=request_id,
        task_type=task_type,
        priority=priority,
        required_capabilities=capabilities or {"imaging"},
        power_required=power,
        bandwidth_required=bandwidth,
        memory_required=memory,
        sensor_time_required=sensor_time,
        target_lat=lat,
        target_lon=lon,
        dependencies=dependencies or [],
    )


class TestTaskingOptimizer(unittest.TestCase):
    """Tasking optimizer: satellite management, request submission, optimization."""

    def setUp(self):
        self.opt = TaskingOptimizer()

    def test_add_satellite(self):
        sat = _make_satellite("SAT-1")
        result = self.opt.add_satellite(sat)
        self.assertTrue(result)
        self.assertIn("SAT-1", self.opt.satellites)

    def test_add_duplicate_satellite(self):
        self.opt.add_satellite(_make_satellite("SAT-1"))
        result = self.opt.add_satellite(_make_satellite("SAT-1"))
        self.assertFalse(result)
        self.assertEqual(len(self.opt.satellites), 1)

    def test_remove_satellite(self):
        self.opt.add_satellite(_make_satellite("SAT-1"))
        result = self.opt.remove_satellite("SAT-1")
        self.assertTrue(result)
        self.assertNotIn("SAT-1", self.opt.satellites)

    def test_remove_nonexistent_satellite(self):
        result = self.opt.remove_satellite("SAT-99")
        self.assertFalse(result)

    def test_submit_request(self):
        req = _make_request("REQ-1")
        result = self.opt.submit_request(req)
        self.assertTrue(result)
        self.assertIn("REQ-1", self.opt.requests)

    def test_submit_duplicate_request(self):
        self.opt.submit_request(_make_request("REQ-1"))
        result = self.opt.submit_request(_make_request("REQ-1"))
        self.assertFalse(result)

    def test_optimize_priority_assigns_highest_first(self):
        self.opt.add_satellite(_make_satellite("SAT-1", power_available=15.0))
        self.opt.submit_request(_make_request("LOW", priority=1, power=10.0))
        self.opt.submit_request(_make_request("HIGH", priority=10, power=10.0))
        plan = self.opt.optimize_priority()
        self.assertIn("HIGH", plan.assignments)
        self.assertIn("LOW", plan.unassigned)

    def test_optimize_coverage_assigns_closest(self):
        self.opt.add_satellite(_make_satellite("SAT-FAR", lat=40.0, lon=40.0))
        self.opt.add_satellite(_make_satellite("SAT-NEAR", lat=1.0, lon=1.0))
        self.opt.submit_request(_make_request("REQ-1", lat=0.0, lon=0.0))
        plan = self.opt.optimize_coverage()
        self.assertEqual(plan.assignments["REQ-1"], "SAT-NEAR")

    def test_optimize_minimal_resource(self):
        self.opt.add_satellite(_make_satellite("SAT-1", power_available=100.0))
        self.opt.add_satellite(_make_satellite("SAT-2", power_available=10.0))
        self.opt.submit_request(_make_request("REQ-1", power=5.0))
        plan = self.opt.optimize_minimal_resource()
        self.assertEqual(plan.assignments["REQ-1"], "SAT-1")

    def test_optimize_skips_offline_satellite(self):
        self.opt.add_satellite(
            _make_satellite("SAT-OFF", status=SatelliteStatus.OFFLINE)
        )
        self.opt.submit_request(_make_request("REQ-1"))
        plan = self.opt.optimize_priority()
        self.assertIn("REQ-1", plan.unassigned)

    def test_optimize_skips_insufficient_resources(self):
        self.opt.add_satellite(_make_satellite("SAT-1", power_available=5.0))
        self.opt.submit_request(_make_request("REQ-1", power=50.0))
        plan = self.opt.optimize_priority()
        self.assertIn("REQ-1", plan.unassigned)

    def test_optimize_skips_missing_capabilities(self):
        self.opt.add_satellite(
            _make_satellite("SAT-1", capabilities={"tracking"})
        )
        self.opt.submit_request(
            _make_request("REQ-1", capabilities={"imaging"})
        )
        plan = self.opt.optimize_priority()
        self.assertIn("REQ-1", plan.unassigned)

    def test_resolve_dependencies(self):
        self.opt.submit_request(_make_request("A"))
        self.opt.submit_request(_make_request("B", dependencies=["A"]))
        self.opt.submit_request(_make_request("C", dependencies=["B"]))
        order = self.opt.resolve_dependencies(["A", "B", "C"])
        self.assertEqual(order, ["A", "B", "C"])

    def test_resolve_dependencies_multiple_roots(self):
        self.opt.submit_request(_make_request("A"))
        self.opt.submit_request(_make_request("B"))
        self.opt.submit_request(_make_request("C", dependencies=["A", "B"]))
        order = self.opt.resolve_dependencies(["A", "B", "C"])
        self.assertEqual(len(order), 3)
        self.assertLess(order.index("A"), order.index("C"))
        self.assertLess(order.index("B"), order.index("C"))

    def test_get_satellite_utilization(self):
        sat = _make_satellite("SAT-1", power_available=75.0)
        self.opt.add_satellite(sat)
        util = self.opt.get_satellite_utilization("SAT-1")
        self.assertAlmostEqual(util["power"], 0.25)

    def test_get_satellite_utilization_nonexistent(self):
        util = self.opt.get_satellite_utilization("SAT-99")
        self.assertEqual(util, {})

    def test_plan_stored(self):
        self.opt.add_satellite(_make_satellite("SAT-1"))
        self.opt.submit_request(_make_request("REQ-1"))
        plan = self.opt.optimize_priority()
        self.assertIn(plan.plan_id, self.opt.plans)


class TestResourceAllocator(unittest.TestCase):
    """Resource allocator: allocation, deallocation, power distribution, load balancing."""

    def setUp(self):
        self.allocator = ResourceAllocator()

    def test_allocate(self):
        result = self.allocator.allocate("SAT-1", "power", 30.0)
        self.assertTrue(result)
        self.assertEqual(self.allocator.get_allocation("SAT-1", "power"), 30.0)

    def test_allocate_negative(self):
        result = self.allocator.allocate("SAT-1", "power", -10.0)
        self.assertFalse(result)

    def test_deallocate(self):
        self.allocator.allocate("SAT-1", "power", 50.0)
        result = self.allocator.deallocate("SAT-1", "power", 20.0)
        self.assertTrue(result)
        self.assertEqual(self.allocator.get_allocation("SAT-1", "power"), 30.0)

    def test_deallocate_insufficient(self):
        self.allocator.allocate("SAT-1", "power", 10.0)
        result = self.allocator.deallocate("SAT-1", "power", 20.0)
        self.assertFalse(result)

    def test_deallocate_nonexistent_satellite(self):
        result = self.allocator.deallocate("SAT-99", "power", 10.0)
        self.assertFalse(result)

    def test_get_allocation_nonexistent(self):
        result = self.allocator.get_allocation("SAT-99", "power")
        self.assertEqual(result, 0.0)

    def test_optimize_power_distribution(self):
        sats = [
            _make_satellite("SAT-1", power_available=80.0),
            _make_satellite("SAT-2", power_available=20.0),
        ]
        dist = self.allocator.optimize_power_distribution(sats, {"demand": 50.0})
        self.assertAlmostEqual(dist["SAT-1"], 40.0)
        self.assertAlmostEqual(dist["SAT-2"], 10.0)

    def test_optimize_power_distribution_empty(self):
        dist = self.allocator.optimize_power_distribution([], {"demand": 50.0})
        self.assertEqual(dist, {})

    def test_optimize_power_distribution_zero_demand(self):
        sats = [_make_satellite("SAT-1", power_available=80.0)]
        dist = self.allocator.optimize_power_distribution(sats, {})
        self.assertEqual(dist["SAT-1"], 0.0)

    def test_balance_load(self):
        sats = [
            _make_satellite("SAT-1", power_available=50.0),
            _make_satellite("SAT-2", power_available=50.0),
        ]
        balance = self.allocator.balance_load(sats)
        self.assertAlmostEqual(balance["SAT-1"], 0.0)
        self.assertAlmostEqual(balance["SAT-2"], 0.0)

    def test_balance_load_uneven(self):
        sats = [
            _make_satellite("SAT-1", power_available=20.0),
            _make_satellite("SAT-2", power_available=80.0),
        ]
        balance = self.allocator.balance_load(sats)
        self.assertGreater(balance["SAT-1"], 0)
        self.assertLess(balance["SAT-2"], 0)

    def test_balance_load_empty(self):
        balance = self.allocator.balance_load([])
        self.assertEqual(balance, {})


class TestSatelliteCoordinator(unittest.TestCase):
    """Satellite coordinator: imaging, tracking, handoffs, coverage optimization."""

    def setUp(self):
        self.coord = SatelliteCoordinator()

    def test_coordinate_imaging(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1"))
        targets = [{"lat": 0.0, "lon": 0.0, "priority": 5}]
        plan = self.coord.coordinate_imaging(targets)
        self.assertIn("IMG-0", plan.assignments)

    def test_coordinate_tracking(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1"))
        points = [{"lat": 10.0, "lon": 20.0, "priority": 8}]
        plan = self.coord.coordinate_tracking("TGT-1", points)
        self.assertIn("TRK-TGT-1-0", plan.assignments)

    def test_schedule_handoff(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1"))
        self.coord.optimizer.add_satellite(_make_satellite("SAT-2"))
        handoff = self.coord.schedule_handoff("SAT-1", "SAT-2", "TASK-1")
        self.assertEqual(handoff["from_satellite"], "SAT-1")
        self.assertEqual(handoff["to_satellite"], "SAT-2")
        self.assertEqual(handoff["status"], "scheduled")
        self.assertIn(handoff, self.coord.handovers)

    def test_optimize_constellation_coverage(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1", lat=0.0, lon=0.0))
        points = [{"lat": 0.0, "lon": 0.0}, {"lat": 1.0, "lon": 1.0}]
        result = self.coord.optimize_constellation_coverage(points)
        self.assertGreater(result["coverage"], 0.0)
        self.assertIn("point_0", result["assignments"])

    def test_optimize_constellation_coverage_empty(self):
        result = self.coord.optimize_constellation_coverage([])
        self.assertEqual(result["coverage"], 0.0)

    def test_optimize_constellation_coverage_no_satellites(self):
        points = [{"lat": 0.0, "lon": 0.0}]
        result = self.coord.optimize_constellation_coverage(points)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(len(result["gaps"]), 1)

    def test_detect_coverage_gaps(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1", lat=0.0, lon=0.0))
        points = [{"lat": 0.0, "lon": 0.0}, {"lat": 45.0, "lon": 45.0}]
        gaps = self.coord.detect_coverage_gaps(points)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["lat"], 45.0)

    def test_detect_coverage_gaps_all_covered(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1", lat=0.0, lon=0.0))
        points = [{"lat": 0.0, "lon": 0.0}, {"lat": 1.0, "lon": 1.0}]
        gaps = self.coord.detect_coverage_gaps(points)
        self.assertEqual(len(gaps), 0)

    def test_detect_coverage_gaps_empty(self):
        gaps = self.coord.detect_coverage_gaps([])
        self.assertEqual(gaps, [])

    def test_coordinate_imaging_multiple_targets(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1"))
        self.coord.optimizer.add_satellite(_make_satellite("SAT-2"))
        targets = [
            {"lat": 0.0, "lon": 0.0, "priority": 5},
            {"lat": 10.0, "lon": 10.0, "priority": 8},
            {"lat": 20.0, "lon": 20.0, "priority": 3},
        ]
        plan = self.coord.coordinate_imaging(targets)
        self.assertEqual(len(plan.assignments), 3)

    def test_coordinate_tracking_multiple_points(self):
        self.coord.optimizer.add_satellite(_make_satellite("SAT-1"))
        points = [
            {"lat": 0.0, "lon": 0.0, "priority": 5},
            {"lat": 5.0, "lon": 5.0, "priority": 7},
        ]
        plan = self.coord.coordinate_tracking("TGT-1", points)
        self.assertEqual(len(plan.assignments), 2)


if __name__ == "__main__":
    unittest.main()
