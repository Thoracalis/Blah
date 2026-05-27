"""
threats.py – Threat / attacker wave generator.

The attacker is scripted (not RL) in Steps 1–6.
All threat values are FICTIONAL.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .graph import InfrastructureGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threat dataclass
# ---------------------------------------------------------------------------

@dataclass
class Threat:
    id: str
    type: str                  # cruise_missile | ballistic_missile | drone_swarm | strike_aircraft
    target_node_id: str
    spawn_lat: float
    spawn_lon: float
    speed: float               # grid-units per step (approx degrees per step)
    detection_difficulty: float  # 0–1, higher = harder to detect/intercept
    damage_potential: float    # 0–1 if it hits unimpeded
    swarm_size: int = 1        # only > 1 for drone_swarm

    # State
    current_lat: float = 0.0
    current_lon: float = 0.0
    eta: int = 0               # steps until impact
    active: bool = True        # False once intercepted or impacted
    intercepted: bool = False
    impacted: bool = False
    engaged_by: list[str] = field(default_factory=list)  # AD node IDs

    def __post_init__(self) -> None:
        self.current_lat = self.spawn_lat
        self.current_lon = self.spawn_lon

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "target_node_id": self.target_node_id,
            "current_lat": round(self.current_lat, 4),
            "current_lon": round(self.current_lon, 4),
            "eta": self.eta,
            "active": self.active,
            "intercepted": self.intercepted,
            "impacted": self.impacted,
            "detection_difficulty": self.detection_difficulty,
            "damage_potential": self.damage_potential,
            "swarm_size": self.swarm_size,
        }


# ---------------------------------------------------------------------------
# Wave strategy
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "name": "saturation",
        "type_weights": {"cruise_missile": 0.10, "ballistic_missile": 0.05,
                         "drone_swarm": 0.80, "strike_aircraft": 0.05},
        "target_preference": ["air_defense", "radar"],
    },
    {
        "name": "precision",
        "type_weights": {"cruise_missile": 0.50, "ballistic_missile": 0.35,
                         "drone_swarm": 0.05, "strike_aircraft": 0.10},
        "target_preference": ["airbase", "c2", "ammo_depot"],
    },
    {
        "name": "suppression",
        "type_weights": {"cruise_missile": 0.40, "ballistic_missile": 0.10,
                         "drone_swarm": 0.10, "strike_aircraft": 0.40},
        "target_preference": ["radar", "airbase"],
    },
]

# Default threat type parameters (can be overridden by config)
DEFAULT_TYPE_PARAMS: dict[str, dict] = {
    "cruise_missile": {
        "speed": 0.8,
        "detection_difficulty": 0.3,
        "damage_potential": 0.45,
        "swarm_size": 1,
    },
    "ballistic_missile": {
        "speed": 3.0,
        "detection_difficulty": 0.6,
        "damage_potential": 0.75,
        "swarm_size": 1,
    },
    "drone_swarm": {
        "speed": 0.5,
        "detection_difficulty": 0.4,
        "damage_potential": 0.20,
        "swarm_size": 4,
    },
    "strike_aircraft": {
        "speed": 2.0,
        "detection_difficulty": 0.5,
        "damage_potential": 0.60,
        "swarm_size": 1,
    },
}


# ---------------------------------------------------------------------------
# Threat generator
# ---------------------------------------------------------------------------

class ThreatGenerator:
    """
    Generates attacker waves according to scripted strategies.

    The attacker cycles through strategies and targets high-value nodes,
    adjusting priorities based on which defences have been degraded.
    """

    def __init__(self, cfg: dict, rng: random.Random, np_rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng
        self._np_rng = np_rng
        self._threat_counter = 0
        self._strategy_index = 0
        self._type_params = DEFAULT_TYPE_PARAMS.copy()

        # Override with config if provided
        if "types" in cfg.get("threats", {}):
            for ttype, params in cfg["threats"]["types"].items():
                if ttype in self._type_params:
                    self._type_params[ttype].update(params)

        spawn_cfg = cfg.get("threats", {}).get("spawn_region", {})
        self._spawn_lat_range = (
            spawn_cfg.get("lat_min", 59.0),
            spawn_cfg.get("lat_max", 68.0),
        )
        self._spawn_lon_range = (
            spawn_cfg.get("lon_min", 26.0),
            spawn_cfg.get("lon_max", 32.0),
        )
        self._max_per_wave = cfg.get("threats", {}).get("max_threats_per_wave", 6)
        self._wave_interval = cfg.get("threats", {}).get("wave_interval", 8)

    def reset(self) -> None:
        self._threat_counter = 0
        self._strategy_index = 0

    def should_launch_wave(self, step: int) -> bool:
        return step > 0 and step % self._wave_interval == 0

    def generate_wave(self, graph: InfrastructureGraph, step: int) -> list[Threat]:
        """Generate a wave of threats targeting nodes in `graph`."""
        strategy = STRATEGIES[self._strategy_index % len(STRATEGIES)]
        self._strategy_index += 1

        # Adaptively adjust target preference: prefer degraded defenders
        target_prefs = strategy["target_preference"]
        candidate_nodes = self._select_targets(graph, target_prefs, n=self._max_per_wave)

        if not candidate_nodes:
            logger.warning("No valid targets found for wave at step %d", step)
            return []

        # Determine wave size: 1..max_per_wave
        wave_size = self._rng.randint(
            max(1, self._max_per_wave // 2),
            self._max_per_wave,
        )

        threats: list[Threat] = []
        type_weights = strategy["type_weights"]
        types = list(type_weights.keys())
        weights = list(type_weights.values())

        for _ in range(wave_size):
            threat_type = self._rng.choices(types, weights=weights, k=1)[0]
            target = self._rng.choice(candidate_nodes)
            threat = self._spawn_threat(threat_type, target, graph)
            threats.append(threat)

        logger.info("Wave at step %d: strategy=%s, %d threats",
                    step, strategy["name"], len(threats))
        return threats

    # ------------------------------------------------------------------

    def _select_targets(self, graph: InfrastructureGraph,
                        preferred_types: list[str], n: int) -> list[str]:
        """
        Select up to n target node IDs.
        Prefer nodes of preferred_types; fall back to all functional nodes.
        Weight towards high-importance, low-health nodes (attacker exploits weakness).
        """
        candidates = []
        for nid, node in graph.nodes.items():
            if node.type in preferred_types:
                # Attacker prefers targets that are still functional (worth hitting)
                score = node.importance * node.health + 0.1
                candidates.append((nid, score))

        if not candidates:
            # Fall back to all nodes
            candidates = [(nid, n.importance) for nid, n in graph.nodes.items()]

        if not candidates:
            return []

        node_ids, scores = zip(*candidates)
        total = sum(scores)
        probs = [s / total for s in scores]
        k = min(n, len(node_ids))
        chosen = self._rng.choices(list(node_ids), weights=list(probs), k=k)
        return list(set(chosen))  # deduplicate

    def _spawn_threat(self, threat_type: str, target_id: str,
                      graph: InfrastructureGraph) -> Threat:
        params = self._type_params[threat_type]
        target = graph.nodes[target_id]

        # Spawn location: random point in eastern spawn box
        spawn_lat = self._np_rng.uniform(*self._spawn_lat_range)
        spawn_lon = self._np_rng.uniform(*self._spawn_lon_range)

        # Estimate ETA in steps
        dist_km = graph.haversine_km(spawn_lat, spawn_lon, target.lat, target.lon)
        # speed is in degrees/step roughly; convert via ~111 km/degree at these latitudes
        speed_km_per_step = params["speed"] * 111.0
        eta = max(1, int(dist_km / speed_km_per_step))

        # Add small noise to parameters for realism
        dmg = float(self._np_rng.normal(params["damage_potential"], 0.05))
        dmg = float(np.clip(dmg, 0.05, 1.0))

        self._threat_counter += 1
        threat_id = f"THR-{self._threat_counter:04d}-{threat_type[:3].upper()}"

        return Threat(
            id=threat_id,
            type=threat_type,
            target_node_id=target_id,
            spawn_lat=spawn_lat,
            spawn_lon=spawn_lon,
            speed=params["speed"],
            detection_difficulty=params["detection_difficulty"],
            damage_potential=dmg,
            swarm_size=params.get("swarm_size", 1),
            current_lat=spawn_lat,
            current_lon=spawn_lon,
            eta=eta,
        )

    def advance_threats(self, threats: list[Threat],
                        graph: InfrastructureGraph) -> None:
        """
        Move all active threats one step closer to their targets.
        Updates current_lat, current_lon, and eta in-place.
        """
        for threat in threats:
            if not threat.active:
                continue
            target = graph.nodes.get(threat.target_node_id)
            if target is None:
                threat.active = False
                continue

            # Linear interpolation toward target
            dlat = target.lat - threat.current_lat
            dlon = target.lon - threat.current_lon
            dist = max((dlat ** 2 + dlon ** 2) ** 0.5, 1e-9)

            step_size = threat.speed * 0.1   # 0.1 degree per speed unit per step
            if dist <= step_size:
                threat.current_lat = target.lat
                threat.current_lon = target.lon
                threat.eta = 0
            else:
                threat.current_lat += step_size * dlat / dist
                threat.current_lon += step_size * dlon / dist
                threat.eta = max(0, threat.eta - 1)
