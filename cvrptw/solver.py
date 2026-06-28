import time

import numpy as np

from .model import Customer, Instance, Solution, Vehicle
from .search import local_search, perturbation


def run_vehicle(candidates: list[Customer], instance: Instance) -> Vehicle:
    def most_suitable(current, candidates):
        values = [instance.distances[current.cust_id][c.cust_id] * (c.ready_time + 1) * c.due_date
                  for c in candidates]
        return candidates[np.argmin(values)]

    v = Vehicle(instance.capacity, instance.depot, instance.distances)
    remaining = candidates[:]
    while remaining:
        if v.left_capacity < min(c.demand for c in remaining):
            break
        candidate = most_suitable(v.route.customers[-1], remaining)
        v.try_visit(candidate)
        remaining.remove(candidate)
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


def ils(sol: Solution, max_ls_attempts: int, n_perturbation_moves: int, time_limit: int) -> tuple[int, Solution]:
    best_sol = current_sol = sol
    best_dist = sol.distance
    made_iters = 0
    n_failed_iters = 0

    start = time.time()
    while time.time() - start < time_limit and n_failed_iters < 20:
        made_iters += 1
        p_changed, current_sol = perturbation(current_sol, n_moves=n_perturbation_moves)
        ls_changed, current_sol = local_search(current_sol, max_attempts=max_ls_attempts)

        if not (p_changed or ls_changed):
            break

        current_dist = current_sol.distance
        delta = best_dist - current_dist
        if delta > 1e-3:
            best_sol = current_sol
            best_dist = current_dist
            n_failed_iters = 0
            print(f"New best: {best_dist:.2f} ({delta:+.3f}), vehicles = {len(best_sol)}")
        else:
            n_failed_iters += 1

    return made_iters, best_sol
