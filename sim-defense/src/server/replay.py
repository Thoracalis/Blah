"""
replay.py – Episode replay store.

Records full episode frames so the frontend can replay any finished episode.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_STORED_EPISODES = 10


class ReplayStore:
    def __init__(self, save_dir: str | Path | None = None) -> None:
        self._episodes: deque[dict] = deque(maxlen=MAX_STORED_EPISODES)
        self._current_frames: list[dict] = []
        self._save_dir = Path(save_dir) if save_dir else None
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def start_episode(self, episode_id: str, metadata: dict | None = None) -> None:
        self._current_frames = []
        self._current_meta = {
            "episode_id": episode_id,
            "metadata": metadata or {},
        }

    def record_frame(self, step: int, state: dict) -> None:
        self._current_frames.append({"step": step, "state": state})

    def finish_episode(self, final_info: dict | None = None) -> str:
        ep = {
            **self._current_meta,
            "frames": self._current_frames,
            "final_info": final_info or {},
            "n_frames": len(self._current_frames),
        }
        self._episodes.append(ep)
        ep_id = self._current_meta["episode_id"]

        if self._save_dir:
            fpath = self._save_dir / f"episode_{ep_id}.json"
            with fpath.open("w") as fh:
                json.dump(ep, fh)
            logger.info("Episode %s saved to %s", ep_id, fpath)

        self._current_frames = []
        return ep_id

    # ------------------------------------------------------------------

    def list_episodes(self) -> list[dict]:
        return [
            {"episode_id": ep["episode_id"],
             "n_frames": ep["n_frames"],
             "final_defense_ratio": ep["final_info"].get("defense_ratio", 0)}
            for ep in self._episodes
        ]

    def get_episode(self, episode_id: str) -> dict | None:
        for ep in self._episodes:
            if ep["episode_id"] == episode_id:
                return ep
        return None

    def get_frame(self, episode_id: str, step: int) -> dict | None:
        ep = self.get_episode(episode_id)
        if ep is None:
            return None
        for frame in ep["frames"]:
            if frame["step"] == step:
                return frame
        return None
