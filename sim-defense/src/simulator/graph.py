"""
graph.py – Infrastructure graph model for the Air Defense Resilience Simulator.

Nodes represent military/infrastructure assets; edges represent
command links, sensor links, and logistics routes.

All data is FICTIONAL and for educational/research purposes only.
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

NODE_TYPES = {
    "airbase":      {"symbol": "▲", "color_key": "cyan"},
    "air_defense":  {"symbol": "◆", "color_key": "cyan"},
    "radar":        {"symbol": "○", "color_key": "cyan"},
    "fuel_depot":   {"symbol": "■", "color_key": "yellow"},
    "ammo_depot":   {"symbol": "■", "color_key": "yellow"},
    "c2":           {"symbol": "★", "color_key": "cyan"},
    "civil_infra":  {"symbol": "□", "color_key": "white"},
}

EDGE_TYPES = {
    "command_link": {"dashed": False, "color_key": "cyan"},
    "sensor_link":  {"dashed": False, "color_key": "green"},
    "logistics":    {"dashed": False, "color_key": "yellow"},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    id: str
    name: str
    type: str
    lat: float
    lon: float
    health: float                      # 0.0 – 1.0
    capacity: float                    # base capacity when fully healthy
    importance: float                  # used in reward calculation
    protection_level: float            # 0.0 – 1.0, set by defender agent
    effective_capacity: float = 1.0   # computed after cascade
    supply_deficit_steps: int = 0      # steps without required supply
    dependencies: list[str] = field(default_factory=list)
    # Optional type-specific fields
    range_km: float | None = None
    coverage_radius_km: float | None = None

    @property
    def is_functional(self) -> bool:
        return self.health > 0.15 and self.effective_capacity > 0.1

    def apply_damage(self, damage: float) -> float:
        """Apply damage [0,1] to health. Returns actual damage applied."""
        prev = self.health
        self.health = max(0.0, self.health - damage)
        return prev - self.health

    def repair(self, amount: float) -> float:
        """Repair by amount. Returns actual health gained."""
        prev = self.health
        self.health = min(1.0, self.health + amount)
        return self.health - prev

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "lat": self.lat,
            "lon": self.lon,
            "health": round(self.health, 4),
            "capacity": round(self.capacity, 4),
            "effective_capacity": round(self.effective_capacity, 4),
            "importance": self.importance,
            "protection_level": round(self.protection_level, 4),
            "is_functional": self.is_functional,
            "range_km": self.range_km,
            "coverage_radius_km": self.coverage_radius_km,
        }


@dataclass
class EdgeState:
    id: str
    source: str
    target: str
    type: str
    capacity: float
    health: float      # 1.0 = intact, 0.0 = severed

    @property
    def is_intact(self) -> bool:
        return self.health > 0.2

    @property
    def effective_capacity(self) -> float:
        return self.capacity * self.health if self.is_intact else 0.0

    def apply_damage(self, damage: float) -> None:
        self.health = max(0.0, self.health - damage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "health": round(self.health, 4),
            "is_intact": self.is_intact,
            "effective_capacity": round(self.effective_capacity, 4),
        }


# ---------------------------------------------------------------------------
# Infrastructure graph
# ---------------------------------------------------------------------------

class InfrastructureGraph:
    """
    Wraps a NetworkX DiGraph with typed nodes/edges and helper methods
    for capacity propagation, path-finding, and state serialisation.
    """

    def __init__(self, graph_file: str | Path) -> None:
        self._graph_file = Path(graph_file)
        self.nodes: dict[str, NodeState] = {}
        self.edges: dict[str, EdgeState] = {}
        self._nx: nx.DiGraph = nx.DiGraph()
        self._load()
        logger.info("InfrastructureGraph loaded: %d nodes, %d edges",
                    len(self.nodes), len(self.edges))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with self._graph_file.open() as fh:
            data = json.load(fh)

        for nd in data["nodes"]:
            ns = NodeState(
                id=nd["id"],
                name=nd["name"],
                type=nd["type"],
                lat=nd["lat"],
                lon=nd["lon"],
                health=nd.get("health", 1.0),
                capacity=nd.get("capacity", 1.0),
                importance=nd.get("importance", 0.5),
                protection_level=nd.get("protection_level", 0.3),
                effective_capacity=nd.get("capacity", 1.0),
                dependencies=nd.get("dependencies", []),
                range_km=nd.get("range_km"),
                coverage_radius_km=nd.get("coverage_radius_km"),
            )
            self.nodes[ns.id] = ns
            self._nx.add_node(ns.id, state=ns)

        for ed in data["edges"]:
            es = EdgeState(
                id=ed["id"],
                source=ed["source"],
                target=ed["target"],
                type=ed["type"],
                capacity=ed.get("capacity", 1.0),
                health=ed.get("health", 1.0),
            )
            self.edges[es.id] = es
            self._nx.add_edge(es.source, es.target, state=es, key=es.id)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reload from disk to restore pristine state."""
        self.nodes.clear()
        self.edges.clear()
        self._nx.clear()
        self._load()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the full graph state."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
        }

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in km between two lat/lon points."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def nodes_within_range(self, lat: float, lon: float, radius_km: float,
                           node_types: list[str] | None = None) -> list[str]:
        """Return node IDs within radius_km of (lat, lon)."""
        result = []
        for nid, n in self.nodes.items():
            if node_types and n.type not in node_types:
                continue
            if self.haversine_km(lat, lon, n.lat, n.lon) <= radius_km:
                result.append(nid)
        return result

    def air_defense_covering(self, lat: float, lon: float) -> list[str]:
        """Return IDs of functional air-defense nodes that cover (lat, lon)."""
        result = []
        for nid, n in self.nodes.items():
            if n.type != "air_defense" or not n.is_functional:
                continue
            if n.range_km is None:
                continue
            dist = self.haversine_km(lat, lon, n.lat, n.lon)
            if dist <= n.range_km * n.effective_capacity:  # range degrades with capacity
                result.append(nid)
        return result

    def radars_covering(self, lat: float, lon: float) -> list[str]:
        """Return IDs of functional radar nodes whose coverage envelope includes (lat, lon)."""
        result = []
        for nid, n in self.nodes.items():
            if n.type != "radar" or not n.is_functional:
                continue
            if n.coverage_radius_km is None:
                continue
            dist = self.haversine_km(lat, lon, n.lat, n.lon)
            if dist <= n.coverage_radius_km * n.effective_capacity:
                result.append(nid)
        return result

    # ------------------------------------------------------------------
    # Network / flow helpers
    # ------------------------------------------------------------------

    def _active_subgraph(self, edge_types: list[str] | None = None) -> nx.DiGraph:
        """Build a subgraph containing only intact edges (optionally filtered by type)."""
        g = nx.DiGraph()
        g.add_nodes_from(self._nx.nodes)
        for es in self.edges.values():
            if not es.is_intact:
                continue
            if edge_types and es.type not in edge_types:
                continue
            g.add_edge(es.source, es.target, weight=1.0 / max(es.effective_capacity, 0.01))
        return g

    def has_path(self, source: str, target: str,
                 edge_types: list[str] | None = None) -> bool:
        g = self._active_subgraph(edge_types)
        return nx.has_path(g, source, target)

    def shortest_path(self, source: str, target: str,
                      edge_types: list[str] | None = None) -> list[str] | None:
        g = self._active_subgraph(edge_types)
        try:
            return nx.shortest_path(g, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def supply_reachability(self, target_id: str,
                            supplier_types: list[str]) -> float:
        """
        Compute the fraction of required supply types that can still reach
        `target_id` via intact logistics edges.

        Returns a value in [0, 1] where 1 = all suppliers reachable.
        """
        suppliers = [nid for nid, n in self.nodes.items()
                     if n.type in supplier_types and n.is_functional]
        if not suppliers:
            return 0.0
        reachable = sum(
            1 for s in suppliers
            if self.has_path(s, target_id, edge_types=["logistics"])
        )
        return reachable / len(suppliers)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def total_defense_capacity(self) -> float:
        """Weighted sum of health × importance across all nodes."""
        return sum(n.health * n.importance for n in self.nodes.values())

    def max_defense_capacity(self) -> float:
        return sum(n.importance for n in self.nodes.values())

    def defense_ratio(self) -> float:
        mx = self.max_defense_capacity()
        return self.total_defense_capacity() / mx if mx > 0 else 0.0

    def nodes_by_importance(self) -> list[NodeState]:
        return sorted(self.nodes.values(), key=lambda n: n.importance, reverse=True)

    def nodes_by_type(self, node_type: str) -> list[NodeState]:
        return [n for n in self.nodes.values() if n.type == node_type]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"InfrastructureGraph(nodes={len(self.nodes)}, "
                f"edges={len(self.edges)}, "
                f"defense_ratio={self.defense_ratio():.2f})")
