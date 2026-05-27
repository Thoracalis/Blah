"""
env.py – Gymnasium environment wrapping the Air Defense Resilience Simulator.

Observation space: Dict with node features, edge index, incoming threats, global resources.
Action space:      MultiDiscrete – hierarchical discrete choices per step.

All values FICTIONAL. Educational / research tool only.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .graph import InfrastructureGraph
from .threats import ThreatGenerator, Threat
from .engagement import EngagementEngine, EngagementResult
from .cascade import CascadeEngine, CascadeEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action spec constants
# ---------------------------------------------------------------------------

# Action 0: Protect top-N nodes (which group of nodes to buff)
#   0 = protect top-3 by importance
#   1 = protect top-5 by importance
#   2 = protect air_defense nodes
#   3 = protect radar nodes
#   4 = protect c2 nodes

N_PROTECT_CHOICES = 5

# Action 1: Relocate AD (index of AD node to reposition toward most threatened area)
#   0 = no relocation
#   1..n = move AD unit n-1 toward most threatened sector

N_AD_NODES = 6  # approximate; padded

# Action 2: Repair priority
#   0 = no repair
#   1 = repair most damaged node
#   2 = repair most damaged c2
#   3 = repair most damaged radar
#   4 = repair most damaged airbase

N_REPAIR_CHOICES = 5

# Action 3: Sortie launch
#   0 = no sortie
#   1 = launch sortie against strike_aircraft
#   2 = launch sortie for CAP (general deterrence)

N_SORTIE_CHOICES = 3

MAX_THREATS_OBS = 12    # padded threat list length
N_NODE_FEATURES  = 6    # health, eff_cap, importance, protection, lat_norm, lon_norm


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AirDefenseEnv(gym.Env):
    """
    Gymnasium environment for the Air Defense Resilience Simulator.

    The agent plays the role of the Swedish air-defence commander.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, cfg: dict, render_mode: str | None = None) -> None:
        super().__init__()
        self._cfg = cfg
        self.render_mode = render_mode

        sim_cfg = cfg.get("simulator", {})
        self._episode_length = sim_cfg.get("episode_length", 150)
        self._graph_file = Path(sim_cfg.get("graph_file", "src/data/sweden_graph.json"))
        res_cfg = sim_cfg.get("resources", {})
        self._total_protection_budget = res_cfg.get("total_protection_budget", 10.0)
        self._max_relocations = res_cfg.get("max_air_defense_relocations", 2)
        self._max_repairs = res_cfg.get("max_repair_actions", 3)
        self._sortie_pool_max = res_cfg.get("sortie_pool", 20)
        self._sortie_recharge = res_cfg.get("sortie_recharge_per_step", 1)

        rew_cfg = sim_cfg.get("reward", {})
        self._rew_defense_weight  = rew_cfg.get("defense_capacity_weight", 1.0)
        self._rew_stop_weight     = rew_cfg.get("stop_rate_weight", 2.0)
        self._rew_crit_penalty    = rew_cfg.get("critical_damage_penalty", 5.0)
        self._rew_waste_penalty   = rew_cfg.get("resource_waste_penalty", 0.1)
        self._rew_survival_bonus  = rew_cfg.get("survival_bonus", 50.0)
        self._rew_survival_thresh = rew_cfg.get("survival_threshold", 0.70)

        # Spaces
        n_nodes = 29  # from graph JSON; we pad/truncate observations

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(n_nodes, N_NODE_FEATURES),
                dtype=np.float32,
            ),
            "threats": spaces.Box(
                low=0.0, high=1.0,
                shape=(MAX_THREATS_OBS, 6),   # type_enc(4), eta_norm, dmg_potential
                dtype=np.float32,
            ),
            "global": spaces.Box(
                low=0.0, high=1.0,
                shape=(4,),  # defense_ratio, stop_rate, sorties_left_norm, step_norm
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.MultiDiscrete([
            N_PROTECT_CHOICES,
            N_AD_NODES,
            N_REPAIR_CHOICES,
            N_SORTIE_CHOICES,
        ])

        # Will be initialised in reset()
        self._graph: InfrastructureGraph | None = None
        self._threat_gen: ThreatGenerator | None = None
        self._engagement: EngagementEngine | None = None
        self._cascade: CascadeEngine | None = None

        self._step_count = 0
        self._active_threats: list[Threat] = []
        self._sortie_pool = self._sortie_pool_max
        self._last_stop_rate = 0.0
        self._episode_events: list[dict] = []

        # Node order list (stable across steps within episode)
        self._node_order: list[str] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        base_seed = seed if seed is not None else self._cfg.get("simulator", {}).get("seed", 42)
        py_rng = random.Random(base_seed)
        np_rng = np.random.default_rng(base_seed)

        self._graph = InfrastructureGraph(self._graph_file)
        self._threat_gen = ThreatGenerator(self._cfg.get("simulator", {}), py_rng, np_rng)
        self._engagement = EngagementEngine(self._cfg.get("simulator", {}), np_rng)
        self._cascade = CascadeEngine(self._cfg.get("simulator", {}))

        self._step_count = 0
        self._active_threats = []
        self._sortie_pool = self._sortie_pool_max
        self._last_stop_rate = 0.0
        self._episode_events = []
        self._node_order = list(self._graph.nodes.keys())

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        assert self._graph is not None, "Call reset() first"
        self._step_count += 1

        # --- 1. Apply defender actions ---
        self._apply_action(action)

        # --- 2. Generate new threats ---
        if self._threat_gen.should_launch_wave(self._step_count):
            new_threats = self._threat_gen.generate_wave(self._graph, self._step_count)
            self._active_threats.extend(new_threats)
            for t in new_threats:
                self._log_event(f"T+{self._step_count:04d}: Wave: {t.id} → {t.target_node_id}")

        # --- 3. Advance threat positions ---
        self._threat_gen.advance_threats(self._active_threats, self._graph)

        # --- 4. Resolve engagements ---
        eng_results = self._engagement.resolve_step(self._active_threats, self._graph)
        n_intercepted = sum(1 for r in eng_results if r.intercepted)
        n_total = len(eng_results)
        self._last_stop_rate = n_intercepted / n_total if n_total > 0 else 1.0

        for r in eng_results:
            if r.intercepted:
                self._log_event(
                    f"T+{self._step_count:04d}: INTERCEPTED {r.threat_id} "
                    f"(p={r.intercept_probability:.2f}, by {r.engaged_by})"
                )
            elif r.miss:
                self._log_event(f"T+{self._step_count:04d}: MISS {r.threat_id}")
            else:
                self._log_event(
                    f"T+{self._step_count:04d}: HIT {r.threat_id} → "
                    f"{r.hit_node_id} dmg={r.damage_applied:.2f}"
                )

        # --- 5. Cascade propagation ---
        cascade_events = self._cascade.propagate(self._graph)
        for ev in cascade_events:
            if ev.event_type in ("supply_cut", "degraded"):
                self._log_event(
                    f"T+{self._step_count:04d}: CASCADE {ev.node_id}: {ev.detail}"
                )

        # --- 6. Sortie recharge ---
        self._sortie_pool = min(
            self._sortie_pool_max,
            self._sortie_pool + self._sortie_recharge,
        )

        # --- 7. Prune resolved threats ---
        self._active_threats = [t for t in self._active_threats if t.active]

        # --- 8. Compute reward ---
        reward = self._compute_reward(eng_results, cascade_events)

        # --- 9. Check termination ---
        terminated = self._step_count >= self._episode_length
        if terminated and self._graph.defense_ratio() >= self._rew_survival_thresh:
            reward += self._rew_survival_bonus

        obs = self._get_obs()
        info = self._get_info(eng_results=eng_results, cascade_events=cascade_events)

        if self.render_mode == "human":
            self._render_terminal()

        return obs, reward, terminated, False, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self._render_terminal(return_str=True)
        return None

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        protect_choice, ad_reloc, repair_choice, sortie_choice = int(action[0]), int(action[1]), int(action[2]), int(action[3])

        # --- Protect ---
        self._apply_protect(protect_choice)

        # --- Repair ---
        self._apply_repair(repair_choice)

        # --- Sortie ---
        self._apply_sortie(sortie_choice)

        # AD relocation handled implicitly (future: with delay cost)

    def _apply_protect(self, choice: int) -> None:
        """Redistribute protection budget based on chosen strategy."""
        nodes = self._graph.nodes
        budget = self._total_protection_budget
        targets: list[str] = []

        if choice == 0:
            targets = [n.id for n in self._graph.nodes_by_importance()[:3]]
        elif choice == 1:
            targets = [n.id for n in self._graph.nodes_by_importance()[:5]]
        elif choice == 2:
            targets = [n.id for n in self._graph.nodes_by_type("air_defense")]
        elif choice == 3:
            targets = [n.id for n in self._graph.nodes_by_type("radar")]
        elif choice == 4:
            targets = [n.id for n in self._graph.nodes_by_type("c2")]

        # Reset all protection levels proportionally
        per_node = min(1.0, budget / max(len(targets), 1))
        for nid, node in nodes.items():
            node.protection_level = per_node if nid in targets else 0.1

    def _apply_repair(self, choice: int) -> None:
        if choice == 0:
            return

        candidates: list[str] = []
        if choice == 1:
            candidates = [n.id for n in sorted(self._graph.nodes.values(),
                                               key=lambda n: n.health)
                          if n.health < 0.95][:self._max_repairs]
        elif choice == 2:
            candidates = [n.id for n in self._graph.nodes_by_type("c2")
                          if n.health < 0.95][:self._max_repairs]
        elif choice == 3:
            candidates = [n.id for n in self._graph.nodes_by_type("radar")
                          if n.health < 0.95][:self._max_repairs]
        elif choice == 4:
            candidates = [n.id for n in self._graph.nodes_by_type("airbase")
                          if n.health < 0.95][:self._max_repairs]

        for nid in candidates:
            self._cascade.apply_repair(self._graph, nid)

    def _apply_sortie(self, choice: int) -> None:
        if choice == 0 or self._sortie_pool <= 0:
            return

        if choice == 1:
            # Launch against strike_aircraft
            strike_threats = [t for t in self._active_threats
                              if t.active and t.type == "strike_aircraft"]
            for t in strike_threats[:2]:
                if self._sortie_pool <= 0:
                    break
                t.intercepted = True
                t.active = False
                self._sortie_pool -= 1
                self._log_event(
                    f"T+{self._step_count:04d}: SORTIE intercepted {t.id}"
                )
        elif choice == 2:
            # CAP: slightly boost intercept on next wave
            self._sortie_pool = max(0, self._sortie_pool - 1)

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        n_nodes = len(self._node_order)
        # Lat/lon normalisation bounds (rough Sweden bounding box)
        lat_min, lat_max = 55.0, 69.0
        lon_min, lon_max = 11.0, 24.0

        node_feats = np.zeros((n_nodes, N_NODE_FEATURES), dtype=np.float32)
        for i, nid in enumerate(self._node_order):
            n = self._graph.nodes[nid]
            node_feats[i] = [
                n.health,
                n.effective_capacity,
                n.importance,
                n.protection_level,
                (n.lat - lat_min) / (lat_max - lat_min),
                (n.lon - lon_min) / (lon_max - lon_min),
            ]

        # Pad to fixed size
        obs_nodes = np.zeros((29, N_NODE_FEATURES), dtype=np.float32)
        obs_nodes[:n_nodes] = node_feats

        # Threats
        TYPE_ENC = {
            "cruise_missile":    [1, 0, 0, 0],
            "ballistic_missile": [0, 1, 0, 0],
            "drone_swarm":       [0, 0, 1, 0],
            "strike_aircraft":   [0, 0, 0, 1],
        }
        threat_obs = np.zeros((MAX_THREATS_OBS, 6), dtype=np.float32)
        active = [t for t in self._active_threats if t.active]
        for i, t in enumerate(active[:MAX_THREATS_OBS]):
            enc = TYPE_ENC.get(t.type, [0, 0, 0, 0])
            threat_obs[i] = enc + [
                min(1.0, t.eta / 20.0),
                t.damage_potential,
            ]

        global_obs = np.array([
            self._graph.defense_ratio(),
            self._last_stop_rate,
            self._sortie_pool / self._sortie_pool_max,
            self._step_count / self._episode_length,
        ], dtype=np.float32)

        return {
            "node_features": obs_nodes,
            "threats": threat_obs,
            "global": global_obs,
        }

    def _get_info(self, eng_results: list[EngagementResult] | None = None,
                  cascade_events: list[CascadeEvent] | None = None) -> dict:
        return {
            "step": self._step_count,
            "defense_ratio": self._graph.defense_ratio(),
            "active_threats": len([t for t in self._active_threats if t.active]),
            "sortie_pool": self._sortie_pool,
            "last_stop_rate": self._last_stop_rate,
            "graph_snapshot": self._graph.snapshot(),
            "engagement_results": [r.to_dict() for r in (eng_results or [])],
            "cascade_events": [e.to_dict() for e in (cascade_events or [])],
            "event_log": list(self._episode_events[-20:]),  # last 20 events
            "threats_state": [t.to_dict() for t in self._active_threats if t.active],
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, eng_results: list[EngagementResult],
                        cascade_events: list[CascadeEvent]) -> float:
        r = 0.0

        # Defense capacity maintained
        r += self._rew_defense_weight * self._graph.defense_ratio()

        # Stop rate this step
        r += self._rew_stop_weight * self._last_stop_rate

        # Penalty for critical nodes hit
        critical_types = {"c2", "radar", "air_defense"}
        for res in eng_results:
            if res.hit_node_id and not res.miss:
                node = self._graph.nodes.get(res.hit_node_id)
                if node and node.type in critical_types:
                    r -= self._rew_crit_penalty * res.damage_applied

        return r

    # ------------------------------------------------------------------
    # Logging / rendering
    # ------------------------------------------------------------------

    def _log_event(self, msg: str) -> None:
        self._episode_events.append(msg)
        logger.info(msg)

    def _render_terminal(self, return_str: bool = False) -> str | None:
        lines = [
            f"\n{'═' * 60}",
            f"  AIR DEFENSE SIMULATOR  |  Step {self._step_count}/{self._episode_length}",
            f"  Defense Ratio: {self._graph.defense_ratio():.3f}  |  "
            f"Stop Rate: {self._last_stop_rate:.2f}  |  "
            f"Sorties: {self._sortie_pool}",
            f"{'─' * 60}",
        ]
        for node in self._graph.nodes_by_importance()[:8]:
            bar = "█" * int(node.health * 10) + "░" * (10 - int(node.health * 10))
            lines.append(
                f"  {node.id:<12} [{bar}] {node.health:.2f}  "
                f"ec={node.effective_capacity:.2f}"
            )
        lines.append(f"{'─' * 60}")
        active = [t for t in self._active_threats if t.active]
        lines.append(f"  Active threats: {len(active)}")
        for t in active[:4]:
            lines.append(f"    {t.id:<20} → {t.target_node_id}  ETA={t.eta}")
        lines.append(f"{'═' * 60}")
        output = "\n".join(lines)
        if return_str:
            return output
        print(output)
        return None
