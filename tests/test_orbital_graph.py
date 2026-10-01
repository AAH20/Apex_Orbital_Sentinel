"""Tests for orbital_graph module."""
import pytest

from src.space.orbital_graph import OrbitalGraph, OrbitalObject


def make_obj(id, x=0, y=0, z=0, vx=0, vy=0, vz=0, regime="LEO", inc=0.0):
    return OrbitalObject(
        id=id,
        position=(x, y, z),
        velocity=(vx, vy, vz),
        regime=regime,
        inclination_deg=inc,
    )


# ── Node Management ──────────────────────────────────────────────────────────


class TestNodeManagement:
    def test_add_node_increments_count(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        assert g.node_count() == 1

    def test_add_duplicate_node_overwrites(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=1))
        g.add_node(make_obj("a", x=2))
        assert g.node_count() == 1
        assert g.get_node("a").position[0] == 2

    def test_remove_node(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.remove_node("a")
        assert g.node_count() == 0
        assert not g.has_node("a")

    def test_remove_nonexistent_node_raises(self):
        g = OrbitalGraph()
        with pytest.raises(KeyError):
            g.remove_node("nonexistent")

    def test_get_nonexistent_node_raises(self):
        g = OrbitalGraph()
        with pytest.raises(KeyError):
            g.get_node("nonexistent")

    def test_nodes_returns_all(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        ids = {n.id for n in g.nodes()}
        assert ids == {"a", "b"}


# ── Edge Management ──────────────────────────────────────────────────────────


class TestEdgeManagement:
    def test_add_edge(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_edge("a", "b", 1.5)
        assert g.has_edge("a", "b")
        assert g.has_edge("b", "a")
        assert g.get_edge_weight("a", "b") == 1.5

    def test_edge_count(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_node(make_obj("c"))
        g.add_edge("a", "b", 1.0)
        g.add_edge("b", "c", 2.0)
        assert g.edge_count() == 2

    def test_remove_edge(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_edge("a", "b", 1.0)
        g.remove_edge("a", "b")
        assert not g.has_edge("a", "b")
        assert g.edge_count() == 0

    def test_neighbors(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_node(make_obj("c"))
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "c", 2.0)
        assert set(g.neighbors("a")) == {"b", "c"}

    def test_degree(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_node(make_obj("c"))
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "c", 2.0)
        assert g.degree("a") == 2
        assert g.degree("b") == 1

    def test_add_edge_nonexistent_node_raises(self):
        g = OrbitalGraph()
        with pytest.raises(KeyError):
            g.add_edge("a", "b", 1.0)


# ── Proximity Analysis ──────────────────────────────────────────────────────


class TestProximity:
    def test_distance(self):
        a = make_obj("a", x=0, y=0, z=0)
        b = make_obj("b", x=3, y=4, z=0)
        assert OrbitalGraph.distance(a, b) == 5.0

    def test_find_proximity(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=10, y=0, z=0))
        g.add_node(make_obj("c", x=100, y=0, z=0))
        result = g.find_proximity("a", threshold_km=20)
        assert len(result) == 1
        assert result[0][0] == "b"
        assert result[0][1] == 10.0

    def test_find_all_proximity_pairs(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=5, y=0, z=0))
        g.add_node(make_obj("c", x=100, y=0, z=0))
        pairs = g.find_all_proximity_pairs(threshold_km=10)
        assert len(pairs) == 1
        assert pairs[0][0] == "a"
        assert pairs[0][1] == "b"

    def test_closest_pair(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=3, y=0, z=0))
        g.add_node(make_obj("c", x=100, y=0, z=0))
        result = g.closest_pair()
        assert result is not None
        assert result[0] == "a"
        assert result[1] == "b"
        assert result[2] == 3.0

    def test_closest_pair_empty_graph(self):
        g = OrbitalGraph()
        assert g.closest_pair() is None

    def test_farthest_pair(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=3, y=0, z=0))
        g.add_node(make_obj("c", x=100, y=0, z=0))
        result = g.farthest_pair()
        assert result is not None
        assert result[2] == 100.0


# ── Cluster Detection ────────────────────────────────────────────────────────


class TestClusters:
    def test_connected_components(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_node(make_obj("c"))
        g.add_edge("a", "b", 1.0)
        components = g.connected_components()
        assert len(components) == 2
        sizes = sorted(len(c) for c in components)
        assert sizes == [1, 2]

    def test_find_clusters_by_proximity(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=5, y=0, z=0))
        g.add_node(make_obj("c", x=100, y=0, z=0))
        clusters = g.find_clusters(threshold_km=10)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 2]

    def test_largest_cluster(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=5, y=0, z=0))
        g.add_node(make_obj("c", x=6, y=0, z=0))
        g.add_node(make_obj("d", x=100, y=0, z=0))
        largest = g.largest_cluster(threshold_km=10)
        assert len(largest) == 3

    def test_cluster_density(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=0, y=0, z=0))
        g.add_node(make_obj("b", x=5, y=0, z=0))
        g.add_edge("a", "b", 1.0)
        density = g.cluster_density(["a", "b"])
        assert density == 1.0

    def test_is_connected_true(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_edge("a", "b", 1.0)
        assert g.is_connected()

    def test_is_connected_false(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        assert not g.is_connected()


# ── Graph Metrics ────────────────────────────────────────────────────────────


class TestMetrics:
    def test_average_degree(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a"))
        g.add_node(make_obj("b"))
        g.add_node(make_obj("c"))
        g.add_edge("a", "b", 1.0)
        g.add_edge("b", "c", 2.0)
        assert g.average_degree() == pytest.approx(4.0 / 3.0)

    def test_regime_distribution(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", regime="LEO"))
        g.add_node(make_obj("b", regime="LEO"))
        g.add_node(make_obj("c", regime="GEO"))
        dist = g.regime_distribution()
        assert dist == {"LEO": 2, "GEO": 1}


# ── Serialization ────────────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        g = OrbitalGraph()
        g.add_node(make_obj("a", x=1, y=2, z=3, regime="LEO"))
        g.add_node(make_obj("b", x=4, y=5, z=6, regime="GEO"))
        g.add_edge("a", "b", 7.0)
        data = g.to_dict()
        g2 = OrbitalGraph.from_dict(data)
        assert g2.node_count() == 2
        assert g2.edge_count() == 1
        assert g2.has_edge("a", "b")
        assert g2.get_node("a").position == (1.0, 2.0, 3.0)
        assert g2.get_node("b").regime == "GEO"
