"""
engagement.py – Stochastic air-defense engagement resolution.

No physics. Probability tables only.
All values are FICTIONAL for educational purposes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .graph import InfrastructureGraph
    from .threats import Threat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engagement result
# ---------------------------------------------------------------------------

@dataclass
class EngagementResult:
    threat_id: str
    intercepted: bool
    hit_node_id: str | None
    damage_applied: float         # 0 if intercepted or miss
    miss: bool                    # True if threat got through but still missed
    intercept_probability: float  # what we computed before rolling
    engaged_by: list[str]         # AD node IDs that participated

    def to_dict(self) -> dict:
        return {
            "threat_id": self.threat_id,
            "intercepted": self.intercepted,
            "hit_node_id": self.hit_node_id,
            "damage_applied": round(self.damage_applied, 4),
            "miss": self.miss,
            "intercept_probability": round(self.intercept_probability, 4),
            "engaged_by": self.engaged_by,
        }


# ---------------------------------------------------------------------------
# Base intercept probabilities (fictional)
# ---------------------------------------------------------------------------

BASE_INTERCEPT_PROB: dict[str, float] = {
    "cruise_missile":    0.72,
    "ballistic_missile": 0.40,
    "drone_swarm":       0.60,
    "strike_aircraft":   0.55,
}

RADAR_COVERAGE_BONUS      = 0.15   # per radar covering the engagement zone
SATURATION_PENALTY        = 0.07   # per simultaneous threat above 1
HEALTH_FACTOR_WEIGHT      = 0.6    # lerp(weight, 1.0, health)
PROTECTION_BONUS          = 0.10   # per unit of protection_level (max 0.30 at 1.0)

MISS_PROB                 = 0.60
PARTIAL_HIT_PROB          = 0.25   # remaining 0.15 = full hit
PARTIAL_DAMAGE_RANGE      = (0.10, 0.30)
FULL_DAMAGE_RANGE         = (0.50, 0.80)


# ---------------------------------------------------------------------------
# Engagement engine
# ---------------------------------------------------------------------------

class EngagementEngine:
    """
    Resolves engagements between incoming threats and air-defense units.

    For each active threat that has reached or is near its target, we:
    1. Find all AD units in range.
    2. Compute intercept probability (base × modifiers).
    3. Roll for intercept.
    4. If not intercepted, roll for hit/miss/partial.
    5. Apply damage to the target node.
    """

    def __init__(self, cfg: dict, np_rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._np_rng = np_rng
        eng_cfg = cfg.get("engagement", {})
        self._base_probs = {**BASE_INTERCEPT_PROB}
        self._base_probs.update(eng_cfg.get("base_intercept_probability", {}))
        self._radar_bonus = eng_cfg.get("radar_coverage_bonus", RADAR_COVERAGE_BONUS)
        self._sat_penalty = eng_cfg.get("saturation_penalty_per_threat", SATURATION_PENALTY)
        self._health_weight = eng_cfg.get("health_factor_weight", HEALTH_FACTOR_WEIGHT)

        dmg_cfg = cfg.get("engagement", {})
        self._miss_prob = dmg_cfg.get("miss_probability", MISS_PROB)
        self._partial_prob = dmg_cfg.get("partial_hit_probability", PARTIAL_HIT_PROB)
        partial_range = dmg_cfg.get("partial_damage_range", list(PARTIAL_DAMAGE_RANGE))
        full_range = dmg_cfg.get("full_damage_range", list(FULL_DAMAGE_RANGE))
        self._partial_range = tuple(partial_range)
        self._full_range = tuple(full_range)

    def resolve_step(self, active_threats: list["Threat"],
                     graph: "InfrastructureGraph") -> list[EngagementResult]:
        """
        Resolve all threats that have reached eta == 0 (impact imminent).
        Returns a list of EngagementResult, one per resolved threat.
        """
        # Group threats by target to compute saturation
        from_target: dict[str, list["Threat"]] = {}
        for t in active_threats:
            if t.active and t.eta == 0:
                from_target.setdefault(t.target_node_id, []).append(t)

        results: list[EngagementResult] = []
        for target_id, threats_group in from_target.items():
            n_simultaneous = len(threats_group)
            for threat in threats_group:
                result = self._resolve_single(
                    threat, graph, n_simultaneous - 1  # -1 for self
                )
                results.append(result)
                # Mark threat resolved
                threat.active = False
                threat.intercepted = result.intercepted
                threat.impacted = not result.intercepted

        return results

    # ------------------------------------------------------------------

    def _resolve_single(self, threat: "Threat", graph: "InfrastructureGraph",
                        concurrent_others: int) -> EngagementResult:
        target_node = graph.nodes.get(threat.target_node_id)

        # Find AD units in range of the threat's current position
        ad_covering = graph.air_defense_covering(
            threat.current_lat, threat.current_lon
        )
        threat.engaged_by = list(ad_covering)

        # Find radars covering this position
        radars = graph.radars_covering(threat.current_lat, threat.current_lon)

        # --- Compute intercept probability ---
        p = self._base_probs.get(threat.type, 0.5)

        # Radar bonus
        p += self._radar_bonus * len(radars)

        # Saturation penalty
        p -= self._sat_penalty * concurrent_others

        # AD health modifier (best AD unit health in range)
        if ad_covering:
            best_ad_health = max(graph.nodes[nid].health for nid in ad_covering)
            # lerp: at health=0 → p*weight, at health=1 → p
            health_multiplier = (1 - self._health_weight) + self._health_weight * best_ad_health
            p *= health_multiplier
        else:
            # No AD coverage → very low intercept chance (only VSHORAD / luck)
            p *= 0.08

        # Target protection level adds some intercept chance
        if target_node:
            p += PROTECTION_BONUS * target_node.protection_level

        # Detection difficulty reduces intercept probability
        p *= (1.0 - 0.5 * threat.detection_difficulty)

        p = float(np.clip(p, 0.0, 0.98))

        # --- Roll intercept ---
        intercepted = bool(self._np_rng.random() < p)

        if intercepted:
            return EngagementResult(
                threat_id=threat.id,
                intercepted=True,
                hit_node_id=None,
                damage_applied=0.0,
                miss=False,
                intercept_probability=p,
                engaged_by=ad_covering,
            )

        # --- Threat got through – roll hit/miss/partial/full ---
        roll = float(self._np_rng.random())
        if roll < self._miss_prob:
            # Miss – lands in terrain
            return EngagementResult(
                threat_id=threat.id,
                intercepted=False,
                hit_node_id=None,
                damage_applied=0.0,
                miss=True,
                intercept_probability=p,
                engaged_by=ad_covering,
            )
        elif roll < self._miss_prob + self._partial_prob:
            # Partial hit
            damage = float(self._np_rng.uniform(*self._partial_range))
            damage *= threat.damage_potential
        else:
            # Full hit
            damage = float(self._np_rng.uniform(*self._full_range))
            damage *= threat.damage_potential

        damage = float(np.clip(damage, 0.0, 1.0))

        # For drone swarms, damage scales with swarm size that got through
        if threat.type == "drone_swarm" and threat.swarm_size > 1:
            # Each drone in the swarm has independent hit chance
            n_hitting = int(self._np_rng.binomial(threat.swarm_size, 1 - p))
            n_hitting = max(1, n_hitting)  # at least 1 got through
            damage = min(1.0, damage * n_hitting / threat.swarm_size * 1.5)

        # Apply damage to target node
        if target_node:
            actual_damage = target_node.apply_damage(damage)
            logger.debug("Threat %s hit %s for %.3f damage (health now %.3f)",
                         threat.id, target_node.id, actual_damage, target_node.health)
        else:
            actual_damage = damage

        return EngagementResult(
            threat_id=threat.id,
            intercepted=False,
            hit_node_id=threat.target_node_id if target_node else None,
            damage_applied=actual_damage,
            miss=False,
            intercept_probability=p,
            engaged_by=ad_covering,
        )
