"""
A* path planner over a traversability cost map.

- 8-directional movement (cardinal + diagonal)
- Edge cost: average of the two cell costs × Euclidean step distance
- Heuristic: Euclidean distance (admissible, consistent)
- Returns path as list of (row, col) grid indices
"""

import heapq
import numpy as np
from dataclasses import dataclass

from backend.terrain.cost_map import CostMap


# 8-directional neighbours: (drow, dcol, step_distance)
_NEIGHBOURS = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, 1.4142135623730951),
    (-1,  1, 1.4142135623730951),
    ( 1, -1, 1.4142135623730951),
    ( 1,  1, 1.4142135623730951),
]


@dataclass
class PlanResult:
    path: list[tuple[int, int]]  # (row, col) from start to goal
    cost: float                  # total accumulated path cost
    nodes_expanded: int          # planner effort metric


class NoPathError(Exception):
    pass


def plan(
    cost_map: CostMap,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> PlanResult:
    """
    Run A* from start to goal on the given cost map.

    Args:
        cost_map: CostMap from build_cost_map().
        start: (row, col) grid index.
        goal:  (row, col) grid index.

    Returns:
        PlanResult with path, total cost, and nodes expanded.

    Raises:
        ValueError: if start or goal is out of bounds or impassable.
        NoPathError: if no path exists between start and goal.
    """
    grid = cost_map.cost
    rows, cols = grid.shape

    _validate_point(grid, start, "start")
    _validate_point(grid, goal, "goal")

    if start == goal:
        return PlanResult(path=[start], cost=0.0, nodes_expanded=0)

    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[start] = 0.0

    came_from = {}

    # heap entries: (f_score, counter, g_score, row, col)
    # Storing g in the heap enables correct lazy-deletion:
    # a stale entry has g_popped > g_score[r,c] (a better path was found later).
    counter = 0
    h_start = _heuristic(start[0], start[1], goal)
    heap = [(h_start, counter, 0.0, start[0], start[1])]

    nodes_expanded = 0

    while heap:
        f, _, g_popped, r, c = heapq.heappop(heap)

        if (r, c) == goal:
            return PlanResult(
                path=_reconstruct_path(came_from, goal),
                cost=float(g_score[goal]),
                nodes_expanded=nodes_expanded,
            )

        # Skip stale heap entries — a better path to (r,c) was found after push
        if g_popped > g_score[r, c] + 1e-9:
            continue

        nodes_expanded += 1

        for dr, dc, step_dist in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            neighbour_cost = grid[nr, nc]
            if not np.isfinite(neighbour_cost):
                continue

            # Edge cost: mean of the two cell costs × step distance
            edge_cost = 0.5 * (grid[r, c] + neighbour_cost) * step_dist
            tentative_g = g_score[r, c] + edge_cost

            if tentative_g < g_score[nr, nc]:
                g_score[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                h = _heuristic(nr, nc, goal)
                counter += 1
                heapq.heappush(heap, (tentative_g + h, counter, tentative_g, nr, nc))

    raise NoPathError(f"No traversable path from {start} to {goal}.")


def _heuristic(r: int, c: int, goal: tuple[int, int]) -> float:
    """Euclidean distance — admissible since minimum cell cost is 1.0."""
    return ((r - goal[0]) ** 2 + (c - goal[1]) ** 2) ** 0.5


def _validate_point(grid: np.ndarray, point: tuple[int, int], name: str) -> None:
    r, c = point
    rows, cols = grid.shape
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(f"{name} {point} is out of bounds ({rows}x{cols}).")
    if not np.isfinite(grid[r, c]):
        raise ValueError(f"{name} {point} is impassable (cost=inf).")


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    node = goal
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path
