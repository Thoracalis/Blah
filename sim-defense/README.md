# Air Defense Resilience Simulator

> **DISCLAIMER: This is an abstract pedagogical / research model.**
> All data (coordinates, unit names, capabilities, probabilities) are **fictional approximations**
> used for educational purposes only. This tool does not represent actual Swedish or Russian
> military capabilities, doctrine, or infrastructure. It contains no classified or sensitive
> operational information. Do not use for any operational or policy purposes.

---

## Overview

A reinforcement-learning simulator that trains an agent to optimise the resilience of a
fictional air-defence network against a scripted attacker. The focus is on:

- **Stochastic engagement outcomes** (probabilistic intercept tables)
- **Cascade / dependency failures** (what happens when a radar node goes dark)
- **Flow re-routing** (logistics adapt via NetworkX shortest-path)
- **RL agent** (PPO + GNN-policy learns to balance protection and adaptation)

The front-end looks like a NATO/NORAD Command-and-Control display rendered in a browser.

---

## Architecture

```
C2 Visualisation (Browser / Canvas + D3)
        ↑  WebSocket state stream
FastAPI Server  (run episodes, stream state, pause/play/reset)
        ↑
Simulator  (Gymnasium env – graph, threats, engagement, cascade)
        ↑
RL Agent  (PPO + PyTorch Geometric GNN policy)
```

---

## Quick Start

```bash
# 1. Install (editable)
pip install -e ".[dev]"

# 2. Run the simulator in terminal mode (Step 1)
python -m simulator.run_terminal

# 3. Start the web server (Step 2+)
uvicorn server.main:app --reload --port 8000
# Open http://localhost:8000

# 4. Train the PPO agent (Step 4+)
python -m training.train_ppo
```

---

## Configuration

All simulator parameters live in `config.yaml`. Edit there — nothing is hard-coded.

---

## Project Structure

```
sim-defense/
├── README.md
├── pyproject.toml
├── config.yaml
├── src/
│   ├── simulator/
│   │   ├── graph.py          # Nodes, edges, dependency graph
│   │   ├── threats.py        # Threat / attacker wave generator
│   │   ├── engagement.py     # Stochastic combat resolution
│   │   ├── cascade.py        # Flow & dependency propagation
│   │   └── env.py            # Gymnasium wrapper
│   ├── agents/
│   │   ├── baseline.py       # Rule-based defender
│   │   ├── mlp_policy.py     # PPO with MLP head
│   │   └── gnn_policy.py     # PPO with PyG GNN head
│   ├── training/
│   │   └── train_ppo.py
│   ├── server/
│   │   ├── main.py           # FastAPI application
│   │   ├── websocket.py      # WS connection manager
│   │   └── replay.py         # Episode replay store
│   └── data/
│       └── sweden_graph.json # Predefined node/edge layout
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── c2.js                 # Main C2 display
│   └── map.js                # Sweden map rendering
├── tests/
├── checkpoints/
└── logs/
```

---

## Development Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Simulator core (terminal) |
| 2 | ✅ | C2 web visualisation + random defender |
| 3 | 🔲 | Gymnasium wrapper + rule-based baseline |
| 4 | 🔲 | PPO + MLP policy |
| 5 | 🔲 | GNN policy |
| 6 | 🔲 | Live inference in UI |
| 7 | 🔲 | Self-play (optional) |

---

*Built as an abstract research/educational project. Not affiliated with any military organisation.*
