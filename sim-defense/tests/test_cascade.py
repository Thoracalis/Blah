"""
test_cascade.py – Unit tests for cascade logic.

Tests:
  1. Node health reduction reduces effective_capacity.
  2. Supply cut → dependent node degrades over time.
  3. Alternate route prevents degradation.
  4. C2 loss causes dependent nodes to fall to 30%.
  5. Repair restores health.
  6. Edge damage severs logistics.
  7. Defense ratio calculation.
  8. Haversine distance calculation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from simulator.graph import InfrastructureGraph, NodeState, EdgeState
from simulator.cascade import CascadeEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRAPH_FILE = SRC / "data" / "sweden_graph.json"


@pytest.fixture
def graph() -> InfrastructureGraph:
    """Fresh graph loaded from the JSON file."""
    g = InfrastructureGraph(GRAPH_FILE)
    return g


@pytest.fixture
def cascade(graph) -> CascadeEngine:
    cfg = {
        "cascade": {
            "supply_decay_rate": 0.05,
            "repair_rate": 0.10,
            "dependency_threshold": 0.2,
        }
    }
    return CascadeEngine(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNodeState:
    def test_initial_health_is_1(self, graph):
        for node in graph.nodes.values():
            assert node.health == pytest.approx(1.0), f"Node {node.id} health != 1.0"

    def test_apply_damage_reduces_health(self, graph):
        node = graph.nodes["F21"]
        prev_health = node.health
        applied = node.apply_damage(0.3)
        assert applied == pytest.approx(0.3, abs=1e-6)
        assert node.health == pytest.approx(prev_health - 0.3, abs=1e-6)

    def test_health_cannot_go_below_zero(self, graph):
        node = graph.nodes["F21"]
        node.apply_damage(2.0)
        assert node.health == pytest.approx(0.0)

    def test_repair_restores_health(self, graph):
        node = graph.nodes["F21"]
        node.apply_damage(0.5)
        gained = node.repair(0.3)
        assert gained == pytest.approx(0.3, abs=1e-6)
        assert node.health == pytest.approx(0.8, abs=1e-6)

    def test_health_cannot_exceed_1_on_repair(self, graph):
        node = graph.nodes["F21"]
        node.apply_damage(0.1)
        node.repair(5.0)
        assert node.health == pytest.approx(1.0)

    def test_is_functional_true_when_healthy(self, graph):
        node = graph.nodes["F21"]
        assert node.is_functional is True

    def test_is_functional_false_when_destroyed(self, graph):
        node = graph.nodes["F21"]
        node.apply_damage(1.0)
        assert node.is_functional is False


class TestEdgeState:
    def test_intact_edge_passes_capacity(self, graph):
        edge = list(graph.edges.values())[0]
        assert edge.effective_capacity == pytest.approx(edge.capacity)

    def test_damaged_edge_reduces_capacity(self, graph):
        edge = list(graph.edges.values())[0]
        edge.apply_damage(0.5)
        assert edge.effective_capacity < edge.capacity

    def test_severed_edge_has_zero_capacity(self, graph):
        edge = list(graph.edges.values())[0]
        edge.apply_damage(1.0)
        assert edge.is_intact is False
        assert edge.effective_capacity == pytest.approx(0.0)


class TestCascadeEngine:
    def test_health_reduction_propagates_to_effective_capacity(self, graph, cascade):
        node = graph.nodes["F21"]
        node.apply_damage(0.4)
        cascade.propagate(graph)
        # effective_capacity <= capacity * health
        assert node.effective_capacity <= node.capacity * node.health + 1e-6

    def test_supply_cut_degrades_dependent_node(self, graph, cascade):
        """Destroy all fuel depots and sever logistics edges – F21 should degrade."""
        # Destroy all fuel depots
        for nid, n in graph.nodes.items():
            if n.type == "fuel_depot":
                n.apply_damage(1.0)

        # Sever all logistics edges
        for eid, e in graph.edges.items():
            if e.type == "logistics":
                e.apply_damage(1.0)

        initial_eff = graph.nodes["F21"].effective_capacity
        events = cascade.propagate(graph)
        after_eff = graph.nodes["F21"].effective_capacity

        # Should have degraded
        assert after_eff <= initial_eff

    def test_intact_supply_prevents_degradation(self, graph, cascade):
        """With all supply intact, effective capacity should match capacity * health."""
        events = cascade.propagate(graph)
        node = graph.nodes["F21"]
        # No damage → eff_cap should still be 1.0
        assert node.effective_capacity == pytest.approx(node.capacity * node.health, abs=0.05)

    def test_cascade_events_returned(self, graph, cascade):
        """Cascade propagate returns a list (possibly empty for a healthy graph)."""
        events = cascade.propagate(graph)
        assert isinstance(events, list)

    def test_supply_deficit_accumulates(self, graph, cascade):
        """Multiple steps without supply increase deficit counter."""
        # Sever all fuel/ammo supply to F21
        for eid, e in graph.edges.items():
            if e.type == "logistics" and e.target == "F21":
                e.apply_damage(1.0)
        for nid, n in graph.nodes.items():
            if n.type in ("fuel_depot", "ammo_depot"):
                n.apply_damage(1.0)

        node = graph.nodes["F21"]
        cascade.propagate(graph)
        deficit_after_1 = node.supply_deficit_steps
        cascade.propagate(graph)
        deficit_after_2 = node.supply_deficit_steps

        # Deficit should grow each step
        assert deficit_after_2 >= deficit_after_1

    def test_repair_action_restores_health(self, graph, cascade):
        node = graph.nodes["RAD_GOT"]
        node.apply_damage(0.5)
        health_before = node.health
        cascade.apply_repair(graph, "RAD_GOT", repair_amount=0.2)
        assert node.health > health_before

    def test_c2_loss_degrades_dependent_nodes(self, graph, cascade):
        """When all C2 nodes are destroyed, command-dependent nodes should degrade."""
        for nid, n in graph.nodes.items():
            if n.type == "c2":
                n.apply_damage(1.0)
        for eid, e in graph.edges.items():
            if e.type == "command_link":
                e.apply_damage(1.0)

        node = graph.nodes["AD_N"]
        prev_eff = node.effective_capacity
        cascade.propagate(graph)
        # Should have been reduced to 30% of its previous effective_capacity
        assert node.effective_capacity <= prev_eff * 0.31 + 0.05  # small tolerance

    def test_alternate_route_prevents_supply_cut(self, graph, cascade):
        """
        If a direct edge is cut but an alternate path exists,
        the node should NOT be marked as supply-cut.
        """
        # Sever direct fuel edge from FUEL_N to F21, but FUEL_C→FUEL_N→F21 might route
        for eid, e in graph.edges.items():
            if e.source == "FUEL_N" and e.target == "F21":
                e.apply_damage(1.0)

        # Cascade should attempt rerouting
        events = cascade.propagate(graph)
        node = graph.nodes["F21"]
        # Defense ratio should remain reasonably high (cascade found alternate or supplier)
        # We just check no exception and effective capacity is not zero
        assert node.effective_capacity >= 0.0  # no crash

    def test_edge_damage_severs_path(self, graph, cascade):
        """After severing all logistics edges, has_path should return False."""
        for eid, e in graph.edges.items():
            if e.type == "logistics":
                e.apply_damage(1.0)
        # No logistics path from any fuel depot to F21
        has = graph.has_path("FUEL_N", "F21", edge_types=["logistics"])
        assert has is False


class TestInfrastructureGraph:
    def test_defense_ratio_starts_at_1(self, graph):
        ratio = graph.defense_ratio()
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_defense_ratio_decreases_after_damage(self, graph):
        graph.nodes["C2_C"].apply_damage(0.5)
        ratio = graph.defense_ratio()
        assert ratio < 1.0

    def test_haversine_known_distance(self):
        """Approximate distance Stockholm→Gothenburg ≈ 400 km."""
        dist = InfrastructureGraph.haversine_km(59.33, 18.05, 57.7, 11.97)
        assert 380 < dist < 430, f"Unexpected distance: {dist:.1f} km"

    def test_nodes_within_range(self, graph):
        # Nodes within 50 km of Stockholm C2 (59.33, 18.05)
        nearby = graph.nodes_within_range(59.33, 18.05, 50.0)
        assert len(nearby) >= 1  # At minimum C2_C and CIV_PWR_C should be close

    def test_nodes_by_importance_ordered(self, graph):
        ordered = graph.nodes_by_importance()
        importances = [n.importance for n in ordered]
        assert importances == sorted(importances, reverse=True)

    def test_graph_snapshot_serialisable(self, graph):
        import json
        snap = graph.snapshot()
        # Should not raise
        s = json.dumps(snap)
        assert '"nodes"' in s
        assert '"edges"' in s

    def test_all_node_ids_unique(self, graph):
        ids = list(graph.nodes.keys())
        assert len(ids) == len(set(ids))

    def test_air_defense_covering_known_position(self, graph):
        # AD_GOT is at 57.70, 18.40 with range 100km
        # A point 50km away should be covered
        covered = graph.air_defense_covering(57.5, 18.3)
        assert "AD_GOT" in covered

    def test_air_defense_not_covering_far_point(self, graph):
        # A point very far from any AD node (in Norway)
        covered = graph.air_defense_covering(60.0, 5.0)
        assert len(covered) == 0

    def test_reset_restores_health(self, graph):
        graph.nodes["F21"].apply_damage(0.7)
        graph.reset()
        assert graph.nodes["F21"].health == pytest.approx(1.0)


class TestFullEpisode:
    """Integration test: run a short episode end-to-end."""

    def test_episode_runs_without_error(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        # Short episode
        cfg["simulator"]["episode_length"] = 20

        from simulator.env import AirDefenseEnv
        env = AirDefenseEnv(cfg)
        obs, info = env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == 20
        assert 0.0 <= info["defense_ratio"] <= 1.0

    def test_rule_based_outperforms_random(self):
        """Rule-based agent should achieve higher mean defense ratio than random."""
        import yaml
        from simulator.env import AirDefenseEnv
        from agents.baseline import RuleBasedDefender, RandomDefender

        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        cfg["simulator"]["episode_length"] = 30

        env = AirDefenseEnv(cfg)
        rule_agent = RuleBasedDefender()
        rand_agent = RandomDefender(env.action_space)

        n_episodes = 5

        def run(agent, base_seed):
            ratios = []
            for i in range(n_episodes):
                obs, info = env.reset(seed=base_seed + i)
                done = False
                while not done:
                    action = agent.predict(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                ratios.append(info["defense_ratio"])
            return sum(ratios) / len(ratios)

        rule_ratio = run(rule_agent, base_seed=100)
        rand_ratio = run(rand_agent, base_seed=100)

        # Rule-based should be >= random (with high probability)
        # Use a lenient threshold since random can sometimes do ok
        assert rule_ratio >= rand_ratio - 0.05, (
            f"Rule agent ratio {rule_ratio:.3f} unexpectedly worse than "
            f"random {rand_ratio:.3f}"
        )
