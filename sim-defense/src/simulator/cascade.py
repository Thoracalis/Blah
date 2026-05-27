"""
cascade.py – Dependency propagation and cascade failure logic.

When a node is damaged:
  1. Its effective_capacity is reduced proportionally.
  2. Dependent nodes lose supply → their effective_capacity degrades.
  3. If a transport edge is cut, rerouting is attempted via NetworkX.
  4. If no route exists, dependent nodes accumulate supply_deficit_steps.

All values FICTIONAL for educational purposes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import InfrastructureGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cascade result
# ---------------------------------------------------------------------------

@dataclass
class CascadeEvent:
    node_id: str
    event_type: str          # "capacity_reduced" | "supply_cut" | "rerouted" | "degraded"
    detail: str
    effective_capacity_before: float
    effective_capacity_after: float

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "event_type": self.event_type,
            "detail": self.detail,
            "eff_cap_before": round(self.effective_capacity_before, 4),
            "eff_cap_after":  round(self.effective_capacity_after, 4),
        }


# ---------------------------------------------------------------------------
# Supply type requirements per node type
# ---------------------------------------------------------------------------

SUPPLY_REQUIREMENTS: dict[str, list[str]] = {
    "airbase":      ["fuel_depot", "ammo_depot"],
    "air_defense":  ["ammo_depot"],
    "radar":        [],  # Only needs command link (C2), handled separately
    "fuel_depot":   [],
    "ammo_depot":   [],
    "c2":           [],
    "civil_infra":  [],
}

# Node types that require a C2 command link to function
COMMAND_DEPENDENT_TYPES = {"airbase", "air_defense", "radar"}


# ---------------------------------------------------------------------------
# Cascade engine
# ---------------------------------------------------------------------------

class CascadeEngine:
    """
    Propagates the effects of node/edge damage across the dependency graph.
    Called once per timestep after all engagements are resolved.
    """

    def __init__(self, cfg: dict) -> None:
        cas_cfg = cfg.get("cascade", {})
        self.supply_decay_rate: float = cas_cfg.get("supply_decay_rate", 0.05)
        self.repair_rate: float = cas_cfg.get("repair_rate", 0.08)
        self.dependency_threshold: float = cas_cfg.get("dependency_threshold", 0.2)

    def propagate(self, graph: "InfrastructureGraph") -> list[CascadeEvent]:
        """
        Full cascade propagation pass.
        Returns a list of events describing what changed.
        """
        events: list[CascadeEvent] = []

        # Pass 1: Update each node's effective_capacity based on its own health
        for node in graph.nodes.values():
            old_eff = node.effective_capacity
            node.effective_capacity = node.capacity * node.health
            if abs(node.effective_capacity - old_eff) > 0.01:
                events.append(CascadeEvent(
                    node_id=node.id,
                    event_type="capacity_reduced",
                    detail=f"health={node.health:.3f}",
                    effective_capacity_before=old_eff,
                    effective_capacity_after=node.effective_capacity,
                ))

        # Pass 2: Propagate supply cuts through the dependency graph
        events.extend(self._propagate_supply(graph))

        # Pass 3: Propagate command-link dependency
        events.extend(self._propagate_command(graph))

        return events

    # ------------------------------------------------------------------

    def _propagate_supply(self, graph: "InfrastructureGraph") -> list[CascadeEvent]:
        events: list[CascadeEvent] = []
        for node in graph.nodes.values():
            required_types = SUPPLY_REQUIREMENTS.get(node.type, [])
            if not required_types:
                continue

            old_eff = node.effective_capacity

            # Check reachability for each required supply type
            supply_factor = 1.0
            for stype in required_types:
                # Find suppliers of this type
                suppliers = [
                    nid for nid, n in graph.nodes.items()
                    if n.type == stype and n.health > self.dependency_threshold
                ]
                if not suppliers:
                    # No suppliers of this type exist at all
                    supply_factor = min(supply_factor, 0.0)
                    continue

                # Check if at least one supplier can reach this node via logistics
                reachable = any(
                    graph.has_path(s, node.id, edge_types=["logistics"])
                    for s in suppliers
                )

                if not reachable:
                    # Supply cut – accumulate deficit
                    node.supply_deficit_steps += 1
                    degradation = min(
                        self.supply_decay_rate * node.supply_deficit_steps,
                        0.9,  # cap at 90% degradation
                    )
                    supply_factor = min(supply_factor, 1.0 - degradation)

                    if node.supply_deficit_steps == 1:
                        events.append(CascadeEvent(
                            node_id=node.id,
                            event_type="supply_cut",
                            detail=f"No {stype} supply route to {node.id}",
                            effective_capacity_before=old_eff,
                            effective_capacity_after=node.effective_capacity,
                        ))
                    else:
                        events.append(CascadeEvent(
                            node_id=node.id,
                            event_type="degraded",
                            detail=(f"Supply deficit step {node.supply_deficit_steps}, "
                                    f"factor={supply_factor:.3f}"),
                            effective_capacity_before=old_eff,
                            effective_capacity_after=node.effective_capacity,
                        ))
                else:
                    # Supply reachable – restore deficit counter if previously cut
                    if node.supply_deficit_steps > 0:
                        node.supply_deficit_steps = 0
                        events.append(CascadeEvent(
                            node_id=node.id,
                            event_type="rerouted",
                            detail=f"{stype} supply restored via alternate route",
                            effective_capacity_before=old_eff,
                            effective_capacity_after=node.effective_capacity,
                        ))

            # Apply supply factor to effective capacity
            node.effective_capacity = max(0.0, node.effective_capacity * supply_factor)
        return events

    def _propagate_command(self, graph: "InfrastructureGraph") -> list[CascadeEvent]:
        events: list[CascadeEvent] = []
        for node in graph.nodes.values():
            if node.type not in COMMAND_DEPENDENT_TYPES:
                continue

            old_eff = node.effective_capacity

            # Find C2 nodes
            c2_nodes = [
                nid for nid, n in graph.nodes.items()
                if n.type == "c2" and n.health > self.dependency_threshold
            ]
            if not c2_nodes:
                node.effective_capacity = 0.0
                events.append(CascadeEvent(
                    node_id=node.id,
                    event_type="supply_cut",
                    detail="No functional C2 nodes",
                    effective_capacity_before=old_eff,
                    effective_capacity_after=0.0,
                ))
                continue

            # Check if any C2 can reach this node via command_link
            c2_reachable = any(
                graph.has_path(c2, node.id, edge_types=["command_link"])
                for c2 in c2_nodes
            )
            if not c2_reachable:
                # Degraded command and control – reduced to 30% effectiveness
                node.effective_capacity *= 0.30
                if abs(node.effective_capacity - old_eff) > 0.01:
                    events.append(CascadeEvent(
                        node_id=node.id,
                        event_type="degraded",
                        detail="No C2 command link – operating autonomously at 30%",
                        effective_capacity_before=old_eff,
                        effective_capacity_after=node.effective_capacity,
                    ))
        return events

    # ------------------------------------------------------------------
    # Repair actions
    # ------------------------------------------------------------------

    def apply_repair(self, graph: "InfrastructureGraph",
                     node_id: str, repair_amount: float | None = None) -> float:
        """
        Apply a repair action to the given node.
        Returns the actual health gained.
        """
        node = graph.nodes.get(node_id)
        if node is None:
            logger.warning("Repair action on unknown node %s", node_id)
            return 0.0
        amount = repair_amount if repair_amount is not None else self.repair_rate
        gained = node.repair(amount)
        if gained > 0:
            node.supply_deficit_steps = max(0, node.supply_deficit_steps - 1)
            logger.debug("Repaired %s by %.3f (health now %.3f)", node_id, gained, node.health)
        return gained

    # ------------------------------------------------------------------
    # Edge damage (e.g., convoy/infrastructure strike)
    # ------------------------------------------------------------------

    def damage_edge(self, graph: "InfrastructureGraph",
                    edge_id: str, damage: float) -> None:
        edge = graph.edges.get(edge_id)
        if edge is None:
            return
        edge.apply_damage(damage)
        logger.debug("Edge %s damaged by %.3f (health now %.3f)",
                     edge_id, damage, edge.health)
