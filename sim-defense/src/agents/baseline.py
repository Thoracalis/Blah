"""
baseline.py – Rule-based defender agent (Step 3 baseline).

Strategy:
  - Always protect top-N most important nodes.
  - Repair the most damaged critical node each step.
  - Launch sorties against strike aircraft immediately.
  - Never wastes resources on already-healthy nodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RuleBasedDefender:
    """
    Deterministic rule-based agent that always:
      1. Protects the top-5 most important nodes.
      2. Repairs the most damaged node.
      3. Launches sorties vs strike aircraft.
    """

    def __init__(self) -> None:
        self.name = "RuleBasedDefender"

    def predict(self, obs: dict[str, Any], info: dict[str, Any] | None = None
                ) -> np.ndarray:
        """Return action given observation and optional info dict."""
        action = np.zeros(4, dtype=np.int64)

        # Action 0: Protect top-5 by importance
        action[0] = 1  # top-5 protection

        # Action 1: No relocation (simplification)
        action[1] = 0

        # Action 2: Repair most damaged node
        action[2] = 1  # repair most damaged

        # Action 3: If there are strike_aircraft threats, launch sorties
        if info is not None:
            threats = info.get("threats_state", [])
            has_strike = any(t.get("type") == "strike_aircraft" for t in threats)
            action[3] = 1 if has_strike else 0
        else:
            # From observation alone: if any threat slot has type [0,0,0,1] → strike_aircraft
            threat_obs = obs.get("threats", np.zeros((12, 6)))
            # strike_aircraft enc = [0,0,0,1,...], so col 3 = 1
            if np.any(threat_obs[:, 3] > 0.5):
                action[3] = 1

        return action

    def __repr__(self) -> str:
        return f"RuleBasedDefender(strategy=top5+repair+sortie)"


class RandomDefender:
    """Uniformly random defender (serves as lower bound)."""

    def __init__(self, action_space: Any) -> None:
        self._action_space = action_space
        self.name = "RandomDefender"

    def predict(self, obs: dict[str, Any], info: dict[str, Any] | None = None
                ) -> np.ndarray:
        return self._action_space.sample()

    def __repr__(self) -> str:
        return "RandomDefender()"
