import time
from dataclasses import dataclass

import numpy as np

from .model import Customer, Instance, Solution, Vehicle
from .search import local_search, perturbation


@dataclass
class IterationStats:
    distance: float
    improved: bool
    ls_attempts: int
    cross_improvements: int
    intra_relocate_improvements: int
    exchange_improvements: int
    two_opt_improvements: int
    cross_gain: float
    intra_relocate_gain: float
    exchange_gain: float
    two_opt_gain: float
    perturb_moves: int
    elapsed_s: float
    dist_before_ls: float
    ls_time_s: float
    perturb_time_s: float


ILSStats = list[IterationStats]


def run_vehicle(candidates: list[Customer], instance: Instance) -> Vehicle:
    def most_suitable(current, candidates):
        values = [instance.distances[current.cust_id][c.cust_id] * (c.ready_time + 1) * c.due_date
                  for c in candidates]
        return candidates[np.argmin(values)]

    v = Vehicle(instance.capacity, instance.depot, instance.distances)
    remaining = candidates[:]
    while remaining:
        feasible = [c for c in remaining if v.can_visit(c)]
        if not feasible:
            break
        candidate = most_suitable(v.route.customers[-1], feasible)
        remaining.remove(candidate)
        v.visit(candidate)
    v.visit(v.depot)
    return v


def get_greedy_solution(instance: Instance) -> Solution:
    vehicles = []
    candidates = instance.customers[1:]
    for _ in range(instance.n_vehicles):
        if not candidates:
            break
        v = run_vehicle(candidates, instance)
        vehicles.append(v)
        candidates = [c for c in candidates if c not in v.route.customers]
    return Solution(vehicles)


def ls_attempts_and_time_limit(n_vehicles: int, n_customers: int) -> tuple[int, int]:
    if n_vehicles > 25 or n_customers > 101:
        return 1_000_000, 1800
    return 250_000, 600


def ils(
    sol: Solution,
    max_ls_attempts: int,
    n_perturbation_moves: int,
    time_limit: int,
    verbose: bool = False,
) -> tuple[int, Solution, ILSStats]:
    best_sol = current_sol = sol
    best_dist = sol.distance
    made_iters = 0
    n_failed_iters = 0
    stats: ILSStats = []

    start = time.time()
    while time.time() - start < time_limit and n_failed_iters < 20:
        made_iters += 1
        t0 = time.time()
        p_changed, current_sol, actual_p_moves = perturbation(current_sol, n_moves=n_perturbation_moves)
        t1 = time.time()
        dist_before_ls = current_sol.distance
        ls_changed, current_sol, ls_stats = local_search(current_sol, max_attempts=max_ls_attempts, deadline=start + time_limit)
        t2 = time.time()

        if not (p_changed or ls_changed):
            break

        current_dist = current_sol.distance
        delta = best_dist - current_dist
        improved = delta > 1e-3
        if improved:
            best_sol = current_sol
            best_dist = current_dist
            n_failed_iters = 0
            if verbose:
                print(f"New best: {best_dist:.2f} ({delta:+.3f}), vehicles = {len(best_sol)}")
        else:
            n_failed_iters += 1

        stats.append(IterationStats(
            distance=round(best_dist, 2),
            improved=improved,
            ls_attempts=ls_stats.n_attempts,
            cross_improvements=ls_stats.cross_improvements,
            intra_relocate_improvements=ls_stats.intra_relocate_improvements,
            exchange_improvements=ls_stats.exchange_improvements,
            two_opt_improvements=ls_stats.two_opt_improvements,
            cross_gain=round(ls_stats.cross_gain, 4),
            intra_relocate_gain=round(ls_stats.intra_relocate_gain, 4),
            exchange_gain=round(ls_stats.exchange_gain, 4),
            two_opt_gain=round(ls_stats.two_opt_gain, 4),
            perturb_moves=actual_p_moves,
            elapsed_s=round(t2 - start, 3),
            dist_before_ls=round(dist_before_ls, 2),
            ls_time_s=round(t2 - t1, 3),
            perturb_time_s=round(t1 - t0, 3),
        ))

    return made_iters, best_sol, stats
