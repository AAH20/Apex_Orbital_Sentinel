"""Tests for cluster_analysis module."""
import math
import pytest

from src.space.cluster_analysis import (
    Cluster,
    ClusterAnalyzer,
    CollisionChain,
    ProximityEvent,
    _compute_tca_and_miss_distance,
    _euclidean_distance,
    _relative_velocity,
)
from src.space.orbital_graph import OrbitalObject


def make_obj(id, x=0, y=0, z=0, vx=0, vy=0, vz=0, regime="LEO", inc=0.0):
    return OrbitalObject(
        id=id,
        position=(x, y, z),
        velocity=(vx, vy, vz),
        regime=regime,
        inclination_deg=inc,
    )


# ── Helper Functions ──────────────────────────────────────────────────────────


class TestEuclideanDistance:
    def test_distance_zero(self):
        assert _euclidean_distance((0, 0, 0), (0, 0, 0)) == 0.0

    def test_distance_3d(self):
        assert _euclidean_distance((0, 0, 0), (1, 2, 2)) == pytest.approx(3.0)

    def test_distance_symmetric(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 5.0, 6.0)
        assert _euclidean_distance(a, b) == pytest.approx(_euclidean_distance(b, a))


class TestRelativeVelocity:
    def test_relative_velocity_zero(self):
        assert _relative_velocity((1, 2, 3), (1, 2, 3)) == 0.0

    def test_relative_velocity_basic(self):
        assert _relative_velocity((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    def test_relative_velocity_symmetric(self):
        v1 = (1.0, 2.0, 3.0)
        v2 = (4.0, 5.0, 6.0)
        assert _relative_velocity(v1, v2) == pytest.approx(_relative_velocity(v2, v1))


class TestTCAAndMissDistance:
    def test_parallel_motion(self):
        """Objects moving in parallel: TCA=0, miss=distance."""
        pos_a = (0, 0, 0)
        vel_a = (1, 0, 0)
        pos_b = (0, 5, 0)
        vel_b = (1, 0, 0)
        tca, miss = _compute_tca_and_miss_distance(pos_a, vel_a, pos_b, vel_b)
        assert tca == pytest.approx(0.0)
        assert miss == pytest.approx(5.0)

    def test_head_on_collision(self):
        """Objects on collision course: miss distance ~0."""
        pos_a = (0, 0, 0)
        vel_a = (1, 0, 0)
        pos_b = (10, 0, 0)
        vel_b = (-1, 0, 0)
        tca, miss = _compute_tca_and_miss_distance(pos_a, vel_a, pos_b, vel_b)
        assert tca == pytest.approx(5.0)
        assert miss == pytest.approx(0.0, abs=1e-9)

    def test_receding_objects(self):
        """Objects moving apart: TCA in the past (negative)."""
        pos_a = (0, 0, 0)
        vel_a = (-1, 0, 0)
        pos_b = (10, 0, 0)
        vel_b = (1, 0, 0)
        tca, miss = _compute_tca_and_miss_distance(pos_a, vel_a, pos_b, vel_b)
        assert tca == pytest.approx(-5.0)

    def test_perpendicular_miss(self):
        """Objects passing at right angles."""
        pos_a = (0, 0, 0)
        vel_a = (1, 0, 0)
        pos_b = (5, 3, 0)
        vel_b = (0, 0, 0)
        tca, miss = _compute_tca_and_miss_distance(pos_a, vel_a, pos_b, vel_b)
        assert tca == pytest.approx(5.0)
        assert miss == pytest.approx(3.0)


# ── Cluster Detection ─────────────────────────────────────────────────────────


class TestDetectClusters:
    def test_empty_objects(self):
        analyzer = ClusterAnalyzer([])
        assert analyzer.detect_clusters(threshold_km=100) == []

    def test_single_object(self):
        analyzer = ClusterAnalyzer([make_obj("a", x=0, y=0, z=0)])
        clusters = analyzer.detect_clusters(threshold_km=100)
        assert len(clusters) == 1
        assert clusters[0].members == {"a"}

    def test_two_close_objects_one_cluster(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=5, y=0, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=10)
        assert len(clusters) == 1
        assert clusters[0].members == {"a", "b"}

    def test_two_distant_objects_two_clusters(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=100, y=0, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=10)
        assert len(clusters) == 2

    def test_chain_clustering_transitive(self):
        """A-B close, B-C close, A-C far => all one cluster."""
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=8, y=0, z=0),
            make_obj("c", x=16, y=0, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=10)
        assert len(clusters) == 1
        assert clusters[0].members == {"a", "b", "c"}

    def test_multiple_separate_clusters(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=5, y=0, z=0),
            make_obj("c", x=100, y=0, z=0),
            make_obj("d", x=105, y=0, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=10)
        assert len(clusters) == 2
        sizes = sorted(len(c.members) for c in clusters)
        assert sizes == [2, 2]

    def test_cluster_centroid(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=10, y=0, z=0),
            make_obj("c", x=0, y=10, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=20)
        assert len(clusters) == 1
        centroid = clusters[0].centroid
        assert centroid[0] == pytest.approx(10.0 / 3.0)
        assert centroid[1] == pytest.approx(10.0 / 3.0)
        assert centroid[2] == pytest.approx(0.0)

    def test_cluster_radius(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=10, y=0, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=20)
        assert len(clusters) == 1
        # Radius is max distance from centroid to any member
        assert clusters[0].radius_km == pytest.approx(5.0)

    def test_cluster_density(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=5, y=0, z=0),
            make_obj("c", x=0, y=5, z=0),
        ])
        clusters = analyzer.detect_clusters(threshold_km=20)
        assert len(clusters) == 1
        # 3 objects, max 3 pairs, all within threshold => density 1.0
        assert clusters[0].density == pytest.approx(1.0)


# ── Proximity Analysis ────────────────────────────────────────────────────────


class TestProximityEvents:
    def test_empty_objects(self):
        analyzer = ClusterAnalyzer([])
        assert analyzer.find_proximity_events(threshold_km=100) == []

    def test_single_object_no_events(self):
        analyzer = ClusterAnalyzer([make_obj("a")])
        assert analyzer.find_proximity_events(threshold_km=100) == []

    def test_two_objects_within_threshold(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=5, y=0, z=0),
        ])
        events = analyzer.find_proximity_events(threshold_km=10)
        assert len(events) == 1
        assert events[0].object_a == "a"
        assert events[0].object_b == "b"
        assert events[0].distance_km == pytest.approx(5.0)

    def test_two_objects_beyond_threshold(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=50, y=0, z=0),
        ])
        events = analyzer.find_proximity_events(threshold_km=10)
        assert len(events) == 0

    def test_proximity_event_tca(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=0, z=0, vx=-1, vy=0, vz=0),
        ])
        events = analyzer.find_proximity_events(threshold_km=100)
        assert len(events) == 1
        assert events[0].time_of_closest_approach_s == pytest.approx(5.0)

    def test_proximity_event_miss_distance(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=3, z=0, vx=-1, vy=0, vz=0),
        ])
        events = analyzer.find_proximity_events(threshold_km=100)
        assert len(events) == 1
        assert events[0].miss_distance_km == pytest.approx(3.0)

    def test_proximity_event_relative_velocity(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=0, z=0, vx=-1, vy=0, vz=0),
        ])
        events = analyzer.find_proximity_events(threshold_km=100)
        assert len(events) == 1
        assert events[0].relative_velocity_km_s == pytest.approx(2.0)

    def test_proximity_events_sorted_by_distance(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=5, y=0, z=0),
            make_obj("c", x=2, y=0, z=0),
        ])
        events = analyzer.find_proximity_events(threshold_km=100)
        assert len(events) == 3
        distances = [e.distance_km for e in events]
        assert distances == sorted(distances)

    def test_proximity_events_time_horizon(self):
        """Events beyond time horizon should be excluded."""
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=0.001, vy=0, vz=0),
            make_obj("b", x=1000, y=0, z=0, vx=0, vy=0, vz=0),
        ])
        # TCA = 1000 / 0.001 = 1,000,000 seconds
        # With horizon of 100s, should be excluded
        events = analyzer.find_proximity_events(
            threshold_km=2000, time_horizon_s=100
        )
        assert len(events) == 0

    def test_proximity_events_within_time_horizon(self):
        """Events within time horizon should be included."""
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=100, y=0, z=0, vx=0, vy=0, vz=0),
        ])
        # TCA = 100 seconds
        events = analyzer.find_proximity_events(
            threshold_km=200, time_horizon_s=200
        )
        assert len(events) == 1


# ── Collision Chain Prediction ────────────────────────────────────────────────


class TestCollisionChains:
    def test_no_collisions(self):
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0),
            make_obj("b", x=1000, y=0, z=0),
        ])
        chains = analyzer.predict_collision_chains(collision_threshold_km=1.0)
        assert len(chains) == 0

    def test_direct_collision(self):
        """Two objects on collision course."""
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=0, z=0, vx=-1, vy=0, vz=0),
        ])
        chains = analyzer.predict_collision_chains(collision_threshold_km=1.0)
        assert len(chains) == 1
        assert set(chains[0].initial_pair) == {"a", "b"}
        assert chains[0].objects_affected == {"a", "b"}

    def test_collision_chain_cascade(self):
        """A-B collision creates debris that hits C."""
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=0, z=0, vx=-1, vy=0, vz=0),
            make_obj("c", x=10, y=5, z=0, vx=0, vy=0, vz=0),
        ])
        chains = analyzer.predict_collision_chains(
            collision_threshold_km=1.0,
            cascade_probability=0.5,
        )
        assert len(chains) >= 1
        # The chain should include C as affected
        found_cascade = any("c" in chain.objects_affected for chain in chains)
        assert found_cascade

    def test_collision_chain_length_limit(self):
        """Chain length should be limited by max_chain_length."""
        objects = [
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=0, z=0, vx=-1, vy=0, vz=0),
        ]
        # Add many nearby objects
        for i in range(10):
            objects.append(make_obj(f"c{i}", x=10, y=i, z=0, vx=0, vy=0, vz=0))
        analyzer = ClusterAnalyzer(objects)
        chains = analyzer.predict_collision_chains(
            collision_threshold_km=1.0,
            cascade_probability=1.0,
            max_chain_length=3,
        )
        for chain in chains:
            assert len(chain.chain_sequence) <= 3

    def test_collision_chain_probability_decreases(self):
        """Cascade probability should decrease with chain length."""
        analyzer = ClusterAnalyzer([
            make_obj("a", x=0, y=0, z=0, vx=1, vy=0, vz=0),
            make_obj("b", x=10, y=0, z=0, vx=-1, vy=0, vz=0),
            make_obj("c", x=10, y=5, z=0, vx=0, vy=0, vz=0),
        ])
        chains = analyzer.predict_collision_chains(
            collision_threshold_km=1.0,
            cascade_probability=0.5,
        )
        for chain in chains:
            # Probability should be <= cascade_probability
            assert chain.cascade_probability <= 0.5

    def test_collision_chain_empty_objects(self):
        analyzer = ClusterAnalyzer([])
        chains = analyzer.predict_collision_chains(collision_threshold_km=1.0)
        assert chains == []

    def test_collision_chain_single_object(self):
        analyzer = ClusterAnalyzer([make_obj("a")])
        chains = analyzer.predict_collision_chains(collision_threshold_km=1.0)
        assert chains == []
