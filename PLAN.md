# Space Robotics Research Project — Master Plan

## Project Overview

A unified framework for planetary rover trajectory planning combining:

1. **Learning-augmented optimization** — neural network warm-starts an SCP trajectory optimizer
2. **Delay-aware planning** — accounts for Mars/Moon communication delays, enables multi-hour autonomous operation
3. **Risk-aware objectives** — CVaR replaces simple cost functions, bounding probability of mission failure

Formalized as a **POMDP with delayed observations**, evaluated on real NASA HiRISE/LRO terrain data.

**Target publication:** RA-L with ICRA/IROS co-submission (8-page format)

---

## Tech Stack

### Backend (Python)

- `FastAPI` — serves terrain data and planner results
- `rasterio` / `numpy` — HiRISE DEM ingestion and cost map generation
- `CVXPY` — Sequential Convex Programming (SCP) solver
- `Gymnasium` — custom RL environment for rover navigation
- Baselines: A*, RRT*, PPO

### Frontend

- `CesiumJS` or `Three.js` — 3D terrain rendering
- Click-to-plan UI: select start/end point → call backend → render path overlay

---

## Phases

### Phase 1 — Visual Hook & Data Foundation

**Goal:** Working end-to-end demo on real Mars terrain

- [ ] Ingest NASA HiRISE DEM data
- [ ] Generate 3D mesh from DEM
- [ ] Generate traversability cost map (slope, rock density, slip risk)
- [ ] Implement A\* planner over cost map
- [ ] FastAPI backend serving terrain + planner results
- [ ] Frontend: 3D terrain viewer, click two points, render optimal path

**Deliverable:** Web dashboard — click two points on actual Mars terrain, get a mathematically optimal path avoiding steep slopes.

---

### Phase 2 — Core AI / RL Engine

**Goal:** Physics-based simulation with learning-augmented planner

- [ ] Formalize POMDP problem definition with delay model (do this before coding)
- [ ] Integrate terrain data into physics simulator (MuJoCo or PyBullet)
- [ ] Build custom Gymnasium environment for rover navigation
- [ ] Train RL agent (PPO/SAC) with energy, slip, and risk constraints
- [ ] Implement neural warm-start network (supervised on solved trajectories)
- [ ] Integrate SCP refinement (CVXPY) after neural warm-start
- [ ] Integrate CVaR into SCP cost function

**Deliverable:** Working simulation of rover navigating complex terrain under real physical constraints using the hybrid learning + optimization pipeline.

---

### Phase 3 — Agentic & Risk-Aware Layer

**Goal:** Full uncertainty handling and end-to-end autonomy

- [ ] Add sensor noise and incomplete map simulation
- [ ] Implement communication delay model
- [ ] Online replanning trigger logic
- [ ] Belief-state updates (POMDP-lite or full POMDP)
- [ ] Evaluation harness: repeatable benchmarks across baselines (A*, RRT*, pure PPO)
- [ ] Ablation experiments (with/without warm-start, with/without delay model, CVaR vs expected cost)
- [ ] Paper draft

**Deliverable:** End-to-end system managing full lifecycle of autonomous scientific objective under risk and uncertainty. Paper submitted.

---

## Research Thesis (One Sentence)

> Build the system that proves a rover can plan safely and efficiently under communication delay and terrain uncertainty — using learning to make optimization fast, and risk-awareness to make it trustworthy.

---

## Guiding Principles

- Use real HiRISE/LRO terrain from day one — no toy data
- Write the formal POMDP definition before writing any Phase 2 code
- Publish the core planner paper before building the full platform
- Frontend is a visualization tool, not a research contribution — build it once, freeze it
- Baselines must be honest and well-tuned or reviewers will reject on that alone

---

## Progress Log

| Date       | Milestone                 | Notes                                                 |
| ---------- | ------------------------- | ----------------------------------------------------- |
| 2026-03-29 | Project planning complete | Stack decided, phases defined, ready to start Phase 1 |

---

## Current Status

**Active Phase:** Phase 1 — Visual Hook & Data Foundation

**Next immediate step:** Set up project structure and ingest first HiRISE DEM dataset.
