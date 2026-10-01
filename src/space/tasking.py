"""Space C2 — Tasking Optimization, Resource Allocation, and Multi-Satellite Coordination.

Provides:
- Tasking optimization: priority-based scheduling, coverage optimization, minimal resource usage
- Resource allocation: power distribution, load balancing, constraint satisfaction
- Multi-satellite coordination: imaging coordination, tracking coordination, handoff scheduling, coverage optimization
"""

import uuid
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque


class TaskType(Enum):
    """Types of tasking requests."""
    IMAGING = "imaging"
    TRACKING = "tracking"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    MAINTENANCE = "maintenance"


class SatelliteStatus(Enum):
    """Operational status of a satellite."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class Satellite:
    """A satellite in the constellation with resource tracking."""
    sat_id: str
    name: str
    status: SatelliteStatus = SatelliteStatus.OPERATIONAL
    capabilities: Set[str] = field(default_factory=set)
    power_capacity: float = 100.0
    power_available: float = 100.0
    bandwidth_capacity: float = 1000.0
    bandwidth_available: float = 1000.0
    memory_capacity: float = 1000.0
    memory_available: float = 1000.0
    sensor_time_capacity: float = 60.0
    sensor_time_available: float = 60.0
    orbit_period: float = 90.0
    current_lat: float = 0.0
    current_lon: float = 0.0
    altitude: float = 550.0


@dataclass
class TaskingRequest:
    """A tasking request to be optimized and assigned."""
    request_id: str
    task_type: TaskType
    priority: int
    required_capabilities: Set[str] = field(default_factory=set)
    power_required: float = 0.0
    bandwidth_required: float = 0.0
    memory_required: float = 0.0
    sensor_time_required: float = 0.0
    duration: float = 10.0
    earliest_start: Optional[datetime] = None
    latest_start: Optional[datetime] = None
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_satellite: Optional[str] = None
    status: str = "pending"


@dataclass
class TaskingPlan:
    """An optimized tasking plan with assignments and resource usage."""
    plan_id: str
    assignments: Dict[str, str] = field(default_factory=dict)
    schedule: Dict[str, datetime] = field(default_factory=dict)
    total_power_used: float = 0.0
    total_bandwidth_used: float = 0.0
    total_sensor_time_used: float = 0.0
    unassigned: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskingOptimizer:
    """Optimizes tasking requests across a satellite constellation."""

    def __init__(self):
        self.satellites: Dict[str, Satellite] = {}
        self.requests: Dict[str, TaskingRequest] = {}
        self.plans: Dict[str, TaskingPlan] = {}

    def add_satellite(self, satellite: Satellite) -> bool:
        """Add a satellite to the optimizer. Returns False if ID exists."""
        if satellite.sat_id in self.satellites:
            return False
        self.satellites[satellite.sat_id] = satellite
        return True

    def remove_satellite(self, sat_id: str) -> bool:
        """Remove a satellite. Returns False if not found."""
        if sat_id not in self.satellites:
            return False
        del self.satellites[sat_id]
        return True

    def submit_request(self, request: TaskingRequest) -> bool:
        """Submit a tasking request. Returns False if ID exists."""
        if request.request_id in self.requests:
            return False
        self.requests[request.request_id] = request
        return True

    def _can_assign(self, request: TaskingRequest, satellite: Satellite) -> bool:
        """Check if a satellite can handle a request."""
        if satellite.status != SatelliteStatus.OPERATIONAL:
            return False
        if not request.required_capabilities.issubset(satellite.capabilities):
            return False
        if request.power_required > satellite.power_available:
            return False
        if request.bandwidth_required > satellite.bandwidth_available:
            return False
        if request.memory_required > satellite.memory_available:
            return False
        if request.sensor_time_required > satellite.sensor_time_available:
            return False
        return True

    def _assign(self, request: TaskingRequest, satellite: Satellite) -> None:
        """Assign a request to a satellite, consuming resources."""
        satellite.power_available -= request.power_required
        satellite.bandwidth_available -= request.bandwidth_required
        satellite.memory_available -= request.memory_required
        satellite.sensor_time_available -= request.sensor_time_required
        request.assigned_satellite = satellite.sat_id
        request.status = "assigned"

    def optimize_priority(self, request_ids: Optional[List[str]] = None) -> TaskingPlan:
        """Optimize by priority (highest first). Assigns to first capable satellite."""
        plan = TaskingPlan(plan_id=str(uuid.uuid4()))
        ids = request_ids or list(self.requests.keys())

        sorted_requests = sorted(
            [self.requests[rid] for rid in ids if rid in self.requests],
            key=lambda r: r.priority,
            reverse=True,
        )

        for request in sorted_requests:
            for sat in self.satellites.values():
                if self._can_assign(request, sat):
                    self._assign(request, sat)
                    plan.assignments[request.request_id] = sat.sat_id
                    plan.total_power_used += request.power_required
                    plan.total_bandwidth_used += request.bandwidth_required
                    plan.total_sensor_time_used += request.sensor_time_required
                    break
            else:
                plan.unassigned.append(request.request_id)

        self.plans[plan.plan_id] = plan
        return plan

    def optimize_coverage(self, request_ids: Optional[List[str]] = None) -> TaskingPlan:
        """Optimize for coverage by assigning to closest capable satellite."""
        plan = TaskingPlan(plan_id=str(uuid.uuid4()))
        ids = request_ids or list(self.requests.keys())

        sorted_requests = sorted(
            [self.requests[rid] for rid in ids if rid in self.requests],
            key=lambda r: r.priority,
            reverse=True,
        )

        for request in sorted_requests:
            best_sat = None
            best_dist = float("inf")

            for sat in self.satellites.values():
                if not self._can_assign(request, sat):
                    continue
                if request.target_lat is not None and request.target_lon is not None:
                    dist = math.sqrt(
                        (sat.current_lat - request.target_lat) ** 2
                        + (sat.current_lon - request.target_lon) ** 2
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_sat = sat
                else:
                    best_sat = sat
                    break

            if best_sat:
                self._assign(request, best_sat)
                plan.assignments[request.request_id] = best_sat.sat_id
                plan.total_power_used += request.power_required
                plan.total_bandwidth_used += request.bandwidth_required
                plan.total_sensor_time_used += request.sensor_time_required
            else:
                plan.unassigned.append(request.request_id)

        self.plans[plan.plan_id] = plan
        return plan

    def optimize_minimal_resource(self, request_ids: Optional[List[str]] = None) -> TaskingPlan:
        """Optimize to minimize total resource consumption."""
        plan = TaskingPlan(plan_id=str(uuid.uuid4()))
        ids = request_ids or list(self.requests.keys())

        sorted_requests = sorted(
            [self.requests[rid] for rid in ids if rid in self.requests],
            key=lambda r: r.priority,
            reverse=True,
        )

        for request in sorted_requests:
            best_sat = None
            best_cost = float("inf")

            for sat in self.satellites.values():
                if not self._can_assign(request, sat):
                    continue
                cost = (
                    request.power_required / max(sat.power_capacity, 1e-9)
                    + request.bandwidth_required / max(sat.bandwidth_capacity, 1e-9)
                    + request.memory_required / max(sat.memory_capacity, 1e-9)
                    + request.sensor_time_required / max(sat.sensor_time_capacity, 1e-9)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_sat = sat

            if best_sat:
                self._assign(request, best_sat)
                plan.assignments[request.request_id] = best_sat.sat_id
                plan.total_power_used += request.power_required
                plan.total_bandwidth_used += request.bandwidth_required
                plan.total_sensor_time_used += request.sensor_time_required
            else:
                plan.unassigned.append(request.request_id)

        self.plans[plan.plan_id] = plan
        return plan

    def resolve_dependencies(self, request_ids: List[str]) -> List[str]:
        """Topologically sort requests by dependencies using Kahn's algorithm."""
        in_degree = {rid: 0 for rid in request_ids}
        graph = {rid: [] for rid in request_ids}

        for rid in request_ids:
            if rid not in self.requests:
                continue
            for dep in self.requests[rid].dependencies:
                if dep in request_ids:
                    graph[dep].append(rid)
                    in_degree[rid] += 1

        queue = deque(sorted([rid for rid in request_ids if in_degree[rid] == 0]))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(graph[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_satellite_utilization(self, sat_id: str) -> Dict[str, float]:
        """Get resource utilization for a satellite (0.0 to 1.0)."""
        sat = self.satellites.get(sat_id)
        if not sat:
            return {}
        return {
            "power": 1.0 - (sat.power_available / sat.power_capacity) if sat.power_capacity > 0 else 0.0,
            "bandwidth": 1.0 - (sat.bandwidth_available / sat.bandwidth_capacity) if sat.bandwidth_capacity > 0 else 0.0,
            "memory": 1.0 - (sat.memory_available / sat.memory_capacity) if sat.memory_capacity > 0 else 0.0,
            "sensor_time": 1.0 - (sat.sensor_time_available / sat.sensor_time_capacity) if sat.sensor_time_capacity > 0 else 0.0,
        }


class ResourceAllocator:
    """Allocates resources across satellites with constraint satisfaction."""

    def __init__(self):
        self.allocations: Dict[str, Dict[str, float]] = {}

    def allocate(self, sat_id: str, resource_type: str, amount: float) -> bool:
        """Allocate a resource. Returns False for negative amounts."""
        if amount < 0:
            return False
        if sat_id not in self.allocations:
            self.allocations[sat_id] = {}
        self.allocations[sat_id][resource_type] = self.allocations[sat_id].get(resource_type, 0.0) + amount
        return True

    def deallocate(self, sat_id: str, resource_type: str, amount: float) -> bool:
        """Deallocate a resource. Returns False if insufficient allocation."""
        if amount < 0:
            return False
        if sat_id not in self.allocations:
            return False
        current = self.allocations[sat_id].get(resource_type, 0.0)
        if current < amount:
            return False
        self.allocations[sat_id][resource_type] = current - amount
        return True

    def get_allocation(self, sat_id: str, resource_type: str) -> float:
        """Get current allocation for a resource on a satellite."""
        if sat_id not in self.allocations:
            return 0.0
        return self.allocations[sat_id].get(resource_type, 0.0)

    def optimize_power_distribution(
        self, satellites: List[Satellite], demands: Dict[str, float]
    ) -> Dict[str, float]:
        """Distribute power demands proportionally to available capacity."""
        if not satellites:
            return {}

        total_available = sum(s.power_available for s in satellites)
        total_demand = sum(demands.values())

        if total_demand == 0 or total_available == 0:
            return {s.sat_id: 0.0 for s in satellites}

        distribution = {}
        for sat in satellites:
            share = (sat.power_available / total_available) * min(total_demand, total_available)
            distribution[sat.sat_id] = min(share, sat.power_available)

        return distribution

    def balance_load(self, satellites: List[Satellite]) -> Dict[str, float]:
        """Compute load balancing adjustments (positive = overloaded)."""
        if not satellites:
            return {}

        utilizations = {}
        for sat in satellites:
            util = (1.0 - sat.power_available / sat.power_capacity) if sat.power_capacity > 0 else 0.0
            utilizations[sat.sat_id] = util

        avg_util = sum(utilizations.values()) / len(utilizations)

        return {sat_id: util - avg_util for sat_id, util in utilizations.items()}


class SatelliteCoordinator:
    """Coordinates multi-satellite operations including imaging, tracking, and handoffs."""

    def __init__(self):
        self.optimizer = TaskingOptimizer()
        self.allocator = ResourceAllocator()
        self.handovers: List[Dict[str, Any]] = []

    def coordinate_imaging(self, targets: List[Dict[str, float]]) -> TaskingPlan:
        """Coordinate imaging of multiple targets across satellites."""
        requests = []
        for i, target in enumerate(targets):
            req = TaskingRequest(
                request_id=f"IMG-{i}",
                task_type=TaskType.IMAGING,
                priority=target.get("priority", 5),
                required_capabilities={"imaging"},
                power_required=target.get("power", 20.0),
                bandwidth_required=target.get("bandwidth", 100.0),
                sensor_time_required=target.get("sensor_time", 5.0),
                target_lat=target.get("lat"),
                target_lon=target.get("lon"),
            )
            requests.append(req)
            self.optimizer.submit_request(req)

        return self.optimizer.optimize_coverage([r.request_id for r in requests])

    def coordinate_tracking(
        self, target_id: str, tracking_points: List[Dict[str, float]]
    ) -> TaskingPlan:
        """Coordinate tracking of a target across multiple satellites."""
        requests = []
        for i, point in enumerate(tracking_points):
            req = TaskingRequest(
                request_id=f"TRK-{target_id}-{i}",
                task_type=TaskType.TRACKING,
                priority=point.get("priority", 5),
                required_capabilities={"tracking"},
                power_required=point.get("power", 10.0),
                bandwidth_required=point.get("bandwidth", 50.0),
                sensor_time_required=point.get("sensor_time", 3.0),
                target_lat=point.get("lat"),
                target_lon=point.get("lon"),
            )
            requests.append(req)
            self.optimizer.submit_request(req)

        return self.optimizer.optimize_priority([r.request_id for r in requests])

    def schedule_handoff(self, from_sat: str, to_sat: str, task_id: str) -> Dict[str, Any]:
        """Schedule a handoff between satellites."""
        handoff = {
            "handoff_id": str(uuid.uuid4()),
            "from_satellite": from_sat,
            "to_satellite": to_sat,
            "task_id": task_id,
            "status": "scheduled",
            "scheduled_at": datetime.now(timezone.utc),
        }
        self.handovers.append(handoff)
        return handoff

    def optimize_constellation_coverage(
        self, ground_points: List[Dict[str, float]], max_coverage_radius: float = 15.0
    ) -> Dict[str, Any]:
        """Optimize constellation coverage of ground points."""
        if not ground_points:
            return {"coverage": 0.0, "assignments": {}, "gaps": []}

        if not self.optimizer.satellites:
            return {"coverage": 0.0, "assignments": {}, "gaps": list(ground_points)}

        assignments = {}
        covered = set()

        for i, point in enumerate(ground_points):
            best_sat = None
            best_dist = float("inf")

            for sat in self.optimizer.satellites.values():
                if sat.status != SatelliteStatus.OPERATIONAL:
                    continue
                dist = math.sqrt(
                    (sat.current_lat - point["lat"]) ** 2
                    + (sat.current_lon - point["lon"]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_sat = sat

            if best_sat and best_dist <= max_coverage_radius:
                assignments[f"point_{i}"] = best_sat.sat_id
                covered.add(i)

        coverage = len(covered) / len(ground_points) if ground_points else 0.0
        gaps = [ground_points[i] for i in range(len(ground_points)) if i not in covered]

        return {
            "coverage": coverage,
            "assignments": assignments,
            "gaps": gaps,
        }

    def detect_coverage_gaps(self, ground_points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Detect ground points not covered by any satellite (10-degree radius)."""
        gaps = []
        for point in ground_points:
            covered = False
            for sat in self.optimizer.satellites.values():
                if sat.status != SatelliteStatus.OPERATIONAL:
                    continue
                dist = math.sqrt(
                    (sat.current_lat - point["lat"]) ** 2
                    + (sat.current_lon - point["lon"]) ** 2
                )
                if dist <= 10.0:
                    covered = True
                    break
            if not covered:
                gaps.append(point)
        return gaps
