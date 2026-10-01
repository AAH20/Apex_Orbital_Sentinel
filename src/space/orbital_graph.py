"""Orbital graph intelligence: graph-based relationships, proximity, clusters."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class OrbitalObject:
    """A node in the orbital graph — satellite, debris, station, etc."""

    id: str
    position: Tuple[float, float, float]  # km (ECI)
    velocity: Tuple[float, float, float]  # km/s
    regime: str = "LEO"  # LEO, MEO, GEO, HEO, etc.
    inclination_deg: float = 0.0
    metadata: dict = field(default_factory=dict)


class OrbitalGraph:
    """Graph-based model of orbital relationships.

    Nodes are orbital objects; edges represent proximity or other
    relationships with a weight (typically distance in km).
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, OrbitalObject] = {}
        self._adj: Dict[str, Dict[str, float]] = {}

    # ── Node Management ───────────────────────────────────────────────────

    def add_node(self, obj: OrbitalObject) -> None:
        """Add or replace a node by its id."""
        self._nodes[obj.id] = obj
        self._adj.setdefault(obj.id, {})

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        del self._nodes[node_id]
        del self._adj[node_id]
        for adj in self._adj.values():
            adj.pop(node_id, None)

    def get_node(self, node_id: str) -> OrbitalObject:
        """Return the node with the given id."""
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self) -> List[OrbitalObject]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    # ── Edge Management ───────────────────────────────────────────────────

    def add_edge(self, a: str, b: str, weight: float) -> None:
        """Add an undirected edge between nodes a and b."""
        if a not in self._nodes:
            raise KeyError(f"Node '{a}' not found")
        if b not in self._nodes:
            raise KeyError(f"Node '{b}' not found")
        self._adj[a][b] = weight
        self._adj[b][a] = weight

    def remove_edge(self, a: str, b: str) -> None:
        self._adj[a].pop(b, None)
        self._adj[b].pop(a, None)

    def has_edge(self, a: str, b: str) -> bool:
        return b in self._adj.get(a, {})

    def get_edge_weight(self, a: str, b: str) -> float:
        return self._adj[a][b]

    def edge_count(self) -> int:
        return sum(len(v) for v in self._adj.values()) // 2

    def neighbors(self, node_id: str) -> Set[str]:
        return set(self._adj.get(node_id, {}).keys())

    def degree(self, node_id: str) -> int:
        return len(self._adj.get(node_id, {}))

    # ── Proximity Analysis ───────────────────────────────────────────────

    @staticmethod
    def distance(a: OrbitalObject, b: OrbitalObject) -> float:
        """Euclidean distance between two orbital objects in km."""
        dx = a.position[0] - b.position[0]
        dy = a.position[1] - b.position[1]
        dz = a.position[2] - b.position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def find_proximity(
        self, node_id: str, threshold_km: float
    ) -> List[Tuple[str, float]]:
        """Find all nodes within threshold_km of the given node.

        Returns list of (node_id, distance) tuples sorted by distance.
        """
        source = self.get_node(node_id)
        results: List[Tuple[str, float]] = []
        for nid, obj in self._nodes.items():
            if nid == node_id:
                continue
            d = self.distance(source, obj)
            if d <= threshold_km:
                results.append((nid, d))
        results.sort(key=lambda x: x[1])
        return results

    def find_all_proximity_pairs(
        self, threshold_km: float
    ) -> List[Tuple[str, str, float]]:
        """Find all pairs of nodes within threshold_km of each other.

        Returns list of (id_a, id_b, distance) with id_a < id_b.
        """
        results: List[Tuple[str, str, float]] = []
        ids = sorted(self._nodes.keys())
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1 :]:
                d = self.distance(self._nodes[a_id], self._nodes[b_id])
                if d <= threshold_km:
                    results.append((a_id, b_id, d))
        results.sort(key=lambda x: x[2])
        return results

    def closest_pair(self) -> Optional[Tuple[str, str, float]]:
        """Return the pair of nodes with the smallest distance, or None."""
        pairs = self.find_all_proximity_pairs(float("inf"))
        return pairs[0] if pairs else None

    def farthest_pair(self) -> Optional[Tuple[str, str, float]]:
        """Return the pair of nodes with the largest distance, or None."""
        pairs = self.find_all_proximity_pairs(float("inf"))
        return pairs[-1] if pairs else None

    # ── Cluster Detection ────────────────────────────────────────────────

    def connected_components(self) -> List[Set[str]]:
        """Return connected components of the graph (via edges)."""
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for node_id in self._nodes:
            if node_id in visited:
                continue
            component: Set[str] = set()
            stack = [node_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in self._adj.get(current, {}):
                    if neighbor not in visited:
                        stack.append(neighbor)
            components.append(component)
        return components

    def find_clusters(self, threshold_km: float) -> List[Set[str]]:
        """Cluster nodes by proximity: nodes within threshold_km are connected.

        Uses union-find over proximity pairs.
        """
        parent: Dict[str, str] = {nid: nid for nid in self._nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a_id, b_id, _ in self.find_all_proximity_pairs(threshold_km):
            union(a_id, b_id)

        clusters: Dict[str, Set[str]] = {}
        for nid in self._nodes:
            root = find(nid)
            clusters.setdefault(root, set()).add(nid)
        return list(clusters.values())

    def largest_cluster(self, threshold_km: float) -> Set[str]:
        """Return the largest cluster by proximity threshold."""
        clusters = self.find_clusters(threshold_km)
        return max(clusters, key=len) if clusters else set()

    def cluster_density(self, cluster: Set[str]) -> float:
        """Return edge density of a cluster: 2*E / (N*(N-1))."""
        n = len(cluster)
        if n < 2:
            return 0.0
        edges = sum(
            1
            for a in cluster
            for b in self._adj.get(a, {})
            if b in cluster and a < b
        )
        return edges / (n * (n - 1) / 2)

    def is_connected(self) -> bool:
        """True if the graph is a single connected component."""
        return len(self.connected_components()) <= 1

    # ── Graph Metrics ─────────────────────────────────────────────────────

    def average_degree(self) -> float:
        """Average degree across all nodes."""
        n = self.node_count()
        if n == 0:
            return 0.0
        return sum(self.degree(nid) for nid in self._nodes) / n

    def regime_distribution(self) -> Dict[str, int]:
        """Count of nodes per orbital regime."""
        dist: Dict[str, int] = {}
        for obj in self._nodes.values():
            dist[obj.regime] = dist.get(obj.regime, 0) + 1
        return dist

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the graph to a plain dict."""
        return {
            "nodes": [
                {
                    "id": obj.id,
                    "position": list(obj.position),
                    "velocity": list(obj.velocity),
                    "regime": obj.regime,
                    "inclination_deg": obj.inclination_deg,
                    "metadata": obj.metadata,
                }
                for obj in self._nodes.values()
            ],
            "edges": [
                {"a": a, "b": b, "weight": w}
                for a, neighbors in self._adj.items()
                for b, w in neighbors.items()
                if a < b
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrbitalGraph:
        """Deserialize a graph from a dict produced by to_dict()."""
        g = cls()
        for node_data in data.get("nodes", []):
            obj = OrbitalObject(
                id=node_data["id"],
                position=tuple(node_data["position"]),
                velocity=tuple(node_data["velocity"]),
                regime=node_data.get("regime", "LEO"),
                inclination_deg=node_data.get("inclination_deg", 0.0),
                metadata=node_data.get("metadata", {}),
            )
            g.add_node(obj)
        for edge in data.get("edges", []):
            g.add_edge(edge["a"], edge["b"], edge["weight"])
        return g
