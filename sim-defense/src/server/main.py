"""
main.py – FastAPI server for the Air Defense Resilience Simulator.

Endpoints:
  GET  /               → serves index.html
  GET  /api/status     → current simulation status
  GET  /api/graph      → initial graph layout
  POST /api/control    → pause / play / reset / set_speed
  GET  /api/replay     → list stored episodes
  GET  /api/replay/{id}→ get full episode frames
  WS   /ws             → real-time state stream

Usage:
  cd sim-defense
  uvicorn src.server.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Allow running from sim-defense root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from simulator.env import AirDefenseEnv
from agents.baseline import RuleBasedDefender, RandomDefender
from server.websocket import ConnectionManager
from server.replay import ReplayStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open() as fh:
        return yaml.safe_load(fh)

CFG = _load_cfg()

app = FastAPI(title="Air Defense Resilience Simulator", version="0.1.0")

# Mount frontend static files
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

ws_manager = ConnectionManager()
replay_store = ReplayStore(save_dir=ROOT / "logs" / "replays")

# Simulation state (single-session for now)
_sim_state = {
    "env": None,
    "agent": None,
    "running": False,
    "paused": False,
    "speed": 1.0,          # steps per second
    "step": 0,
    "episode_id": None,
    "last_info": None,
    "last_obs": None,
}
_sim_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env() -> AirDefenseEnv:
    env = AirDefenseEnv(CFG, render_mode=None)
    return env


def _make_agent(agent_type: str) -> Any:
    env = _sim_state["env"]
    if agent_type == "rule":
        return RuleBasedDefender()
    elif agent_type == "random":
        return RandomDefender(env.action_space)
    # Future: load PPO checkpoint
    return RandomDefender(env.action_space)


def _build_state_message(info: dict, obs: dict | None = None) -> dict:
    """Build the WS state payload sent to clients each step."""
    msg: dict[str, Any] = {
        "type": "state",
        "step": _sim_state["step"],
        "episode_id": _sim_state["episode_id"],
        "defense_ratio": info.get("defense_ratio", 0),
        "active_threats": info.get("active_threats", 0),
        "last_stop_rate": info.get("last_stop_rate", 0),
        "sortie_pool": info.get("sortie_pool", 0),
        "graph": info.get("graph_snapshot", {}),
        "threats": info.get("threats_state", []),
        "engagement_results": info.get("engagement_results", []),
        "cascade_events": info.get("cascade_events", []),
        "event_log": info.get("event_log", []),
    }
    return msg


async def _run_episode(agent_type: str = "rule") -> None:
    """Background task: run one episode and stream state via WebSocket."""
    async with _sim_lock:
        env = _make_env()
        _sim_state["env"] = env
        _sim_state["agent"] = _make_agent(agent_type)
        episode_id = str(uuid.uuid4())[:8]
        _sim_state["episode_id"] = episode_id
        _sim_state["step"] = 0
        _sim_state["running"] = True
        _sim_state["paused"] = False

    seed = int(time.time()) % 10000
    obs, info = _sim_state["env"].reset(seed=seed)
    _sim_state["last_obs"] = obs
    _sim_state["last_info"] = info

    replay_store.start_episode(episode_id, metadata={"agent": agent_type, "seed": seed})

    # Broadcast initial graph layout
    await ws_manager.broadcast({
        "type": "episode_start",
        "episode_id": episode_id,
        "graph": info.get("graph_snapshot", {}),
        "config": {
            "episode_length": CFG["simulator"]["episode_length"],
        },
    })

    done = False
    while not done and _sim_state["running"]:
        # Pause loop
        while _sim_state["paused"] and _sim_state["running"]:
            await asyncio.sleep(0.1)

        if not _sim_state["running"]:
            break

        agent = _sim_state["agent"]
        action = agent.predict(obs, info)
        obs, reward, terminated, truncated, info = _sim_state["env"].step(action)
        done = terminated or truncated

        _sim_state["step"] += 1
        _sim_state["last_obs"] = obs
        _sim_state["last_info"] = info

        msg = _build_state_message(info, obs)
        await ws_manager.broadcast(msg)

        replay_store.record_frame(_sim_state["step"], msg)

        # Speed control
        speed = max(0.1, _sim_state["speed"])
        sleep_time = 1.0 / speed
        await asyncio.sleep(sleep_time)

    # Episode finished
    _sim_state["running"] = False
    final_ep_id = replay_store.finish_episode(final_info=_sim_state["last_info"])
    await ws_manager.broadcast({
        "type": "episode_end",
        "episode_id": episode_id,
        "defense_ratio": _sim_state["last_info"].get("defense_ratio", 0),
        "steps": _sim_state["step"],
    })
    logger.info("Episode %s finished. Defense ratio: %.3f",
                episode_id, _sim_state["last_info"].get("defense_ratio", 0))


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = ROOT / "frontend" / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text())
    return HTMLResponse(content="<h1>Simulator starting…</h1><p>Frontend not found.</p>")


@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "running": _sim_state["running"],
        "paused": _sim_state["paused"],
        "step": _sim_state["step"],
        "episode_id": _sim_state["episode_id"],
        "speed": _sim_state["speed"],
        "ws_clients": ws_manager.n_connected,
        "defense_ratio": (
            _sim_state["last_info"].get("defense_ratio", 0)
            if _sim_state["last_info"] else 0
        ),
    })


@app.get("/api/graph")
async def get_graph():
    """Return the current graph snapshot (or initial layout if not running)."""
    if _sim_state["last_info"]:
        return JSONResponse(_sim_state["last_info"]["graph_snapshot"])
    # Return raw JSON from file
    graph_file = ROOT / "src" / "data" / "sweden_graph.json"
    import json
    return JSONResponse(json.loads(graph_file.read_text()))


@app.post("/api/control")
async def control(body: dict):
    """
    Control commands:
      {"cmd": "play", "agent": "rule"}
      {"cmd": "pause"}
      {"cmd": "resume"}
      {"cmd": "reset"}
      {"cmd": "set_speed", "speed": 2.0}
    """
    cmd = body.get("cmd")

    if cmd == "play":
        if _sim_state["running"]:
            return JSONResponse({"status": "already_running"})
        agent_type = body.get("agent", "rule")
        asyncio.create_task(_run_episode(agent_type))
        return JSONResponse({"status": "started", "agent": agent_type})

    elif cmd == "pause":
        _sim_state["paused"] = True
        return JSONResponse({"status": "paused"})

    elif cmd == "resume":
        _sim_state["paused"] = False
        return JSONResponse({"status": "resumed"})

    elif cmd == "reset":
        _sim_state["running"] = False
        _sim_state["paused"] = False
        await asyncio.sleep(0.2)
        agent_type = body.get("agent", "rule")
        asyncio.create_task(_run_episode(agent_type))
        return JSONResponse({"status": "reset_and_started"})

    elif cmd == "set_speed":
        speed = float(body.get("speed", 1.0))
        _sim_state["speed"] = max(0.1, min(20.0, speed))
        return JSONResponse({"status": "speed_set", "speed": _sim_state["speed"]})

    return JSONResponse({"status": "unknown_command"}, status_code=400)


@app.get("/api/replay")
async def list_replays():
    return JSONResponse({"episodes": replay_store.list_episodes()})


@app.get("/api/replay/{episode_id}")
async def get_replay(episode_id: str):
    ep = replay_store.get_episode(episode_id)
    if ep is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(ep)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        # Send current state immediately on connect
        if _sim_state["last_info"]:
            msg = _build_state_message(_sim_state["last_info"])
            await ws_manager.send_to(ws, msg)

        while True:
            # Keep connection alive; clients can send commands via WS too
            data = await ws.receive_text()
            try:
                import json
                cmd = json.loads(data)
                if cmd.get("cmd") == "ping":
                    await ws_manager.send_to(ws, {"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as exc:
        logger.warning("WS error: %s", exc)
        ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    logger.info("Air Defense Resilience Simulator server started.")
    logger.info("DISCLAIMER: All simulation data is fictional. Educational tool only.")
    # Auto-start a rule-based episode
    asyncio.create_task(_run_episode("rule"))
