# Formal POMDP Problem Definition
## Delay-Tolerant, Risk-Aware Planetary Rover Trajectory Planning

---

## 1. Motivation

A Mars rover cannot be controlled in real time. The one-way communication delay between Earth and Mars ranges from **3 to 22 minutes** (average ~10 min), making round-trip command latency 6–44 minutes. As a result:

- Ground operators upload a **plan** for a multi-hour autonomous execution window.
- The rover executes that plan without human input, observing terrain and updating its own beliefs.
- If something goes wrong mid-traverse, the rover must **detect and replan autonomously** — it cannot ask for help.

Standard path planning (A*, RRT*) ignores this entirely. A rover operating under real mission constraints needs a planning framework that models:

1. **Partial observability** — the rover's terrain map is incomplete; it only sees what its sensors cover.
2. **Communication delay** — ground commands and observations are temporally stale.
3. **Stochastic transitions** — wheel slip, unexpected obstacles, and sensor noise create outcome uncertainty.
4. **Risk-bounded objectives** — we don't just want expected-optimal; we want plans that bound the probability of catastrophic failure.

This calls for a **Partially Observable Markov Decision Process (POMDP)** extended with a communication delay model.

---

## 2. The POMDP Tuple

A POMDP is defined by the tuple:

```
M = (S, A, O, T, Z, R, γ, b₀, d)
```

| Symbol | Name | Description |
|--------|------|-------------|
| S | State space | All possible world states |
| A | Action space | Rover control inputs |
| O | Observation space | Sensor readings available to the rover |
| T | Transition model | T(s' \| s, a) — how the world evolves |
| Z | Observation model | Z(o \| s', a) — sensor likelihood |
| R | Reward / cost | R(s, a) — immediate signal |
| γ | Discount factor | γ ∈ (0, 1] — how much future reward matters |
| b₀ | Initial belief | Prior distribution over starting state |
| d | Comm delay | One-way delay in timesteps |

---

## 3. State Space  S

The state at time t is a tuple:

```
s_t = (p_t, m_t, e_t, τ_t)
```

### 3.1 Rover Pose  `p_t = (x, y, θ)`

- `(x, y)` — continuous position on the terrain grid, in meters
- `θ ∈ [0, 2π)` — heading

For Phase 2 implementation we discretise to the 1 m/pixel HiRISE grid, reducing pose to `(row, col, θ)` with 8 discrete headings.

### 3.2 Terrain Belief Map  `m_t`

The terrain is **not fully known**. Each cell `(i, j)` has:

- `h_ij` — true elevation (unknown until observed)
- `σ_ij` — slope derived from elevation (unknown until observed)

The rover maintains a **belief map**: a Gaussian over each cell's elevation:

```
m_t(i,j) = 𝒩(μ_ij^t, Σ_ij^t)
```

Cells within sensor range get their belief updated at each timestep. Cells outside sensor range retain their prior (initialised from the coarse orbital DEM).

### 3.3 Energy Level  `e_t ∈ [0, E_max]`

Battery state in Wh. Decreases with motion (slope-dependent) and idle power draw. Must not reach 0 before a safe stopping point or solar charging zone.

### 3.4 Mission Time  `τ_t ∈ {0, 1, …, H}`

Discrete timestep within the current autonomous execution window of length H.
At `τ_t = H`, the rover must be in a safe state (reachable by ground for the next uplink).

---

## 4. Action Space  A

Discrete actions at each timestep:

| Action | Description |
|--------|-------------|
| `MOVE_N/NE/E/SE/S/SW/W/NW` | Move one cell in the given direction |
| `STOP` | Remain stationary (safe fallback) |

Each `MOVE_*` action attempts to advance one grid cell (~1 m). Whether the rover actually moves as commanded depends on the terrain (transition model, Section 6).

**Action sequence:** Ground uplinks a sequence of actions `a_0, a_1, …, a_{H-1}` at time 0. The rover executes this sequence autonomously, with the option to trigger a **replan event** if its onboard belief diverges significantly from plan assumptions (Section 9).

---

## 5. Observation Space  O

At each timestep, the rover receives:

```
o_t = (ô_terrain, ô_slip, ô_energy)
```

### 5.1 Terrain Observation  `ô_terrain`

Local elevation readings within sensor footprint radius `r_s` (e.g., 5 m for a stereo camera):

```
ô_h(i,j) = h(i,j) + ε_h,    ε_h ~ 𝒩(0, σ_sensor²)
```

Cells outside `r_s` are **not observed** — their beliefs are unchanged.

### 5.2 Slip Observation  `ô_slip ∈ {0, 1}`

Whether the rover detected wheel slip this step:

```
P(ô_slip = 1 | slope) = sigmoid(k · (slope - slope_threshold))
```

Slip is a noisy binary signal. Persistent slip triggers a risk flag.

### 5.3 Energy Observation  `ô_energy`

Measured battery level with small sensor noise:

```
ô_energy = e_t + ε_e,    ε_e ~ 𝒩(0, σ_e²)
```

---

## 6. Transition Model  T(s' | s, a)

### 6.1 Nominal Motion

If action is `MOVE_dir` and the target cell `(x', y')` is traversable:

```
p_t+1 = (x', y', θ_dir)    with probability  1 - P_slip(s_t)
```

### 6.2 Slip

With probability `P_slip`, the rover fails to advance and remains at `(x, y)`:

```
P_slip(s_t) = sigmoid(k · (σ(x,y) - σ_soft))
```

where `σ(x,y)` is the true slope and `σ_soft = 15°` is the soft threshold.

### 6.3 Energy Consumption

Moving uphill costs more energy than flat or downhill:

```
Δe = e_base + e_slope · max(0, Δh / Δx)
```

`e_base` — flat-ground power draw per step
`e_slope` — additional cost per unit of uphill gradient

### 6.4 Belief Map Update

On observing `ô_terrain`, the belief of each observed cell is updated via a Kalman-style correction:

```
μ_ij^{t+1} = μ_ij^t + K · (ô_h(i,j) - μ_ij^t)
Σ_ij^{t+1} = (1 - K) · Σ_ij^t
K = Σ_ij^t / (Σ_ij^t + σ_sensor²)
```

Unobserved cells: `m^{t+1}(i,j) = m^t(i,j)` (belief unchanged).

---

## 7. Communication Delay Model

### 7.1 Delay Parameter

One-way delay `d` in timesteps. For a 1-second timestep and 10-minute delay:

```
d = 600 timesteps
```

### 7.2 What the Delay Means for Planning

At uplink time `t = 0`, ground has access to rover state `s_{-d}` (state from `d` steps ago).
Ground plans over a **belief** `b_0` that is already `d` steps stale.

The rover receives the action sequence at time `t = d` (after one-way delay) and begins executing.
Ground receives the rover's observations at time `t + d` — again `d` steps after the fact.

### 7.3 Autonomy Window

The rover must plan for **H timesteps** of fully autonomous operation:

```
H = T_window - 2d
```

where `T_window` is the total time between uplink and the next communication opportunity. For a typical Mars sol with two comm windows 8 hours apart and 10-min delay:

```
H ≈ 8 hours - 20 min = ~27,600 timesteps (at 1 s/step)
```

### 7.4 Delayed Observation POMDP

The planning problem from the ground's perspective is a **delayed-observation POMDP**:

```
π* = argmax_π  E[ Σ_{t=0}^{H} γ^t R(s_t, a_t) | b_{-d} ]
```

The rover's onboard policy uses its **current** belief `b_t` (updated in real time) to decide whether to continue executing the uplinked plan or trigger a replan.

---

## 8. Reward / Cost Function

We use a **cost minimisation** framing (negated reward):

```
C(s, a) = w_dist · c_dist(s, a)
         + w_energy · c_energy(s, a)
         + w_risk · c_risk(s, a)
         + w_time · c_time(s, a)
         + c_infeasible(s, a)
```

| Term | Description |
|------|-------------|
| `c_dist` | Distance remaining to goal waypoint |
| `c_energy` | Energy consumed this step |
| `c_risk` | Instantaneous slip / damage probability |
| `c_time` | Penalty for time steps used (encourages efficiency) |
| `c_infeasible` | Large penalty for entering impassable terrain or exhausting battery |

### 8.1 CVaR Objective (Phase 2 target)

Instead of minimising expected cost `E[C]`, we minimise **Conditional Value at Risk**:

```
CVaR_α[C] = E[ C | C ≥ VaR_α[C] ]
```

`CVaR_α` is the expected cost in the worst `α` fraction of outcomes (e.g., `α = 0.1` = worst 10%).

**Why CVaR over E[C]?**
A plan that almost always costs 100 but occasionally costs 10,000 (rover gets stuck and is lost) has the same expected cost as a plan that always costs ~200. CVaR penalises the tail — it forces the planner to trade some average efficiency for bounded catastrophic risk. This is the correct objective for an irreplaceable scientific asset.

The full Phase 2 objective:

```
π* = argmin_π  CVaR_α[ Σ_{t=0}^{H} γ^t C(s_t, a_t) ]
```

---

## 9. Belief State and Replanning Trigger

### 9.1 Belief Update (Bayes Filter)

At each step, the rover updates its belief:

```
b_{t+1}(s') = η · Z(o_t | s', a_t) · Σ_s T(s' | s, a_t) · b_t(s)
```

where `η` is a normalisation constant.

### 9.2 Replanning Trigger

The rover monitors the **divergence** between its current belief and the assumptions the uplinked plan was based on:

```
D_t = KL( b_t || b_plan_t )
```

If `D_t > δ_threshold`, the rover switches from plan-execution to **onboard replanning mode**, using a lightweight local planner (e.g., a truncated A* over its current belief map) to find a safe continuation or a safe stopping point.

This is the key mechanism that closes the loop between delayed ground planning and real-time autonomous execution.

---

## 10. Summary: Implementation Roadmap

| Component | Phase | Method |
|-----------|-------|--------|
| Discrete state/action space | Phase 2 | Grid world on HiRISE DEM |
| Terrain belief map | Phase 2 | Per-cell Gaussian, Kalman update |
| Transition model | Phase 2 | Slip probability from slope |
| Observation model | Phase 2 | Gaussian sensor noise |
| Cost function (expected) | Phase 2 | Weighted sum, A*/RL baselines |
| CVaR objective | Phase 2 | CVXPY integration in SCP solver |
| Communication delay model | Phase 3 | Delayed-obs POMDP wrapper |
| Replanning trigger | Phase 3 | KL divergence threshold |
| Full belief propagation | Phase 3 | Particle filter or POMDP-lite |

---

## 11. Notation Reference

| Symbol | Meaning |
|--------|---------|
| `s_t` | State at time t |
| `a_t` | Action at time t |
| `o_t` | Observation at time t |
| `b_t` | Belief (distribution over states) at time t |
| `d` | One-way comm delay in timesteps |
| `H` | Autonomy window length in timesteps |
| `α` | CVaR risk level (e.g., 0.1 = worst 10%) |
| `σ` | Terrain slope in degrees |
| `σ_soft` | Soft slope limit (15°) — cost starts rising |
| `σ_hard` | Hard slope limit (25°) — impassable |
| `r_s` | Sensor footprint radius |
| `e_t` | Energy/battery level at time t |
| `γ` | Discount factor |
| `δ` | Replanning trigger threshold |
