"""Cluster detection, proximity analysis, and collision chain prediction."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Cluster:
    """A group of spatially co-located orbital objects."""

    members: Set[str]
    centroid: Tuple[float, float, float]
    radius_km: float
    density: float


@dataclass
class ProximityEvent:
    """A close approach between two orbital objects."""

    object_a: str
    object_b: str
    distance_km: float
    time_of_closest_approach_s: float
    miss_distance_km: float
    relative_velocity_km_s: float


@dataclass
class CollisionChain:
    """A predicted sequence of collision events."""

    initial_pair: Tuple[str, str]
    objects_affected: Set[str]
    chain_sequence: List[Tuple[str, str]]
    cascade_probability: float


def _euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _relative_velocity(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    """Magnitude of relative velocity between two velocity vectors."""
    dx = v1[0] - v2[0]
    dy = v1[1] - v2[1]
    dz = v1[2] - v2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _compute_tca_and_miss_distance(
    pos_a: Tuple[float, float, float],
    vel_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
    vel_b: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Compute time of closest approach and miss distance.

    Uses linear relative motion approximation.
    Returns (tca_seconds, miss_distance_km).
    """
    # Relative position and velocity
    rx = pos_b[0] - pos_a[0]
    ry = pos_b[1] - pos_a[1]
    rz = pos_b[2] - pos_a[2]
    vx = vel_b[0] - vel_a[0]
    vy = vel_b[1] - vel_a[1]
    vz = vel_b[2] - vel_a[2]

    v_sq = vx * vx + vy * vy + vz * vz

    if v_sq < 1e-12:
        # Relative velocity is zero: TCA is now, miss is current distance
        return 0.0, math.sqrt(rx * rx + ry * ry + rz * rz)

    # TCA = -(r . v) / (v . v)
    tca = -(rx * vx + ry * vy + rz * vz) / v_sq

    # Position at TCA
    dx = rx + vx * tca
    dy = ry + vy * tca
    dz = rz + vz * tca
    miss = math.sqrt(dx * dx + dy * dy + dz * dz)

    return tca, miss


class ClusterAnalyzer:
    """Analyzes orbital objects for clusters, proximity, and collision chains."""

    def __init__(self, objects: list) -> None:
        self._objects = {obj.id: obj for obj in objects}

    # ── Cluster Detection ────────────────────────────────────────────────

    def detect_clusters(self, threshold_km: float) -> List[Cluster]:
        """Detect spatial clusters using union-find over proximity pairs."""
        if not self._objects:
            return []

        ids = sorted(self._objects.keys())
        parent: Dict[str, str] = {nid: nid for nid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Union objects within threshold
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                dist = _euclidean_distance(
                    self._objects[a_id].position,
                    self._objects[b_id].position,
                )
                if dist <= threshold_km:
                    union(a_id, b_id)

        # Group by root
        groups: Dict[str, Set[str]] = {}
        for nid in ids:
            root = find(nid)
            groups.setdefault(root, set()).add(nid)

        return [self._build_cluster(members) for members in groups.values()]

    def _build_cluster(self, members: Set[str]) -> Cluster:
        """Build a Cluster dataclass from a set of member IDs."""
        positions = [self._objects[mid].position for mid in members]
        n = len(positions)

        # Centroid
        cx = sum(p[0] for p in positions) / n
        cy = sum(p[1] for p in positions) / n
        cz = sum(p[2] for p in positions) / n
        centroid = (cx, cy, cz)

        # Radius: max distance from centroid to any member
        radius = max(
            _euclidean_distance(centroid, p) for p in positions
        )

        # Density: fraction of pairs within threshold (use radius as proxy)
        # Actually compute density as connected pairs / total pairs
        if n < 2:
            density = 0.0
        else:
            connected = 0
            total = 0
            member_list = sorted(members)
            for i, a in enumerate(member_list):
                for b in member_list[i + 1:]:
                    total += 1
                    dist = _euclidean_distance(
                        self._objects[a].position,
                        self._objects[b].position,
                    )
                    # Use 2*radius as threshold for density
                    if dist <= 2 * radius:
                        connected += 1
            density = connected / total if total > 0 else 0.0

        return Cluster(
            members=members,
            centroid=centroid,
            radius_km=radius,
            density=density,
        )

    # ── Proximity Analysis ───────────────────────────────────────────────

    def find_proximity_events(
        self,
        threshold_km: float,
        time_horizon_s: Optional[float] = None,
    ) -> List[ProximityEvent]:
        """Find all proximity events within threshold and time horizon."""
        events: List[ProximityEvent] = []
        ids = sorted(self._objects.keys())

        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                obj_a = self._objects[a_id]
                obj_b = self._objects[b_id]

                dist = _euclidean_distance(obj_a.position, obj_b.position)
                if dist > threshold_km:
                    continue

                tca, miss = _compute_tca_and_miss_distance(
                    obj_a.position, obj_a.velocity,
                    obj_b.position, obj_b.velocity,
                )

                # Filter by time horizon
                if time_horizon_s is not None:
                    if tca < 0 or tca > time_horizon_s:
                        continue

                rel_vel = _relative_velocity(obj_a.velocity, obj_b.velocity)

                events.append(ProximityEvent(
                    object_a=a_id,
                    object_b=b_id,
                    distance_km=dist,
                    time_of_closest_approach_s=tca,
                    miss_distance_km=miss,
                    relative_velocity_km_s=rel_vel,
                ))

        events.sort(key=lambda e: e.distance_km)
        return events

    # ── Collision Chain Prediction ───────────────────────────────────────

    def predict_collision_chains(
        self,
        collision_threshold_km: float,
        cascade_probability: float = 0.3,
        max_chain_length: int = 5,
    ) -> List[CollisionChain]:
        """Predict collision chains using cascade model.

        A collision between A and B can create debris that hits nearby objects.
        """
        if len(self._objects) < 2:
            return []

        # Find all potential collision pairs
        collision_pairs: List[Tuple[str, str, float]] = []
        ids = sorted(self._objects.keys())

        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                obj_a = self._objects[a_id]
                obj_b = self._objects[b_id]

                tca, miss = _compute_tca_and_miss_distance(
                    obj_a.position, obj_a.velocity,
                    obj_b.position, obj_b.velocity,
                )

                if miss <= collision_threshold_km and tca >= 0:
                    collision_pairs.append((a_id, b_id, tca))

        if not collision_pairs:
            return []

        # Sort by TCA
        collision_pairs.sort(key=lambda x: x[2])

        chains: List[CollisionChain] = []
        consumed: Set[str] = set()

        for a_id, b_id, tca in collision_pairs:
            if a_id in consumed or b_id in consumed:
                continue

            chain_sequence: List[Tuple[str, str]] = [(a_id, b_id)]
            affected: Set[str] = {a_id, b_id}
            current_prob = cascade_probability

            # Find cascade targets near the collision point
            collision_pos = self._interpolate_position(a_id, b_id, tca)

            for other_id in ids:
                if other_id in affected:
                    continue
                if len(chain_sequence) >= max_chain_length:
                    break

                other = self._objects[other_id]
                dist_to_collision = _euclidean_distance(
                    other.position, collision_pos
                )

                # If object is near collision point, it may be hit by debris
                if dist_to_collision <= collision_threshold_km * 10:
                    chain_sequence.append((b_id, other_id))
                    affected.add(other_id)
                    current_prob *= cascade_probability

            consumed.update(affected)
            chains.append(CollisionChain(
                initial_pair=(a_id, b_id),
                objects_affected=affected,
                chain_sequence=chain_sequence,
                cascade_probability=current_prob,
            ))

        return chains

    def _interpolate_position(
        self,
        a_id: str,
        b_id: str,
        t: float,
    ) -> Tuple[float, float, float]:
        """Estimate collision position as midpoint of two objects at time t."""
        obj_a = self._objects[a_id]
        obj_b = self._objects[b_id]

        # Position at time t
        ax = obj_a.position[0] + obj_a.velocity[0] * t
        ay = obj_a.position[1] + obj_a.velocity[1] * t
        az = obj_a.position[2] + obj_a.velocity[2] * t

        bx = obj_b.position[0] + obj_b.velocity[0] * t
        by = obj_b.position[1] + obj_b.velocity[1] * t
        bz = obj_b.position[2] + obj_b.velocity[2] * t

        return ((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2)
