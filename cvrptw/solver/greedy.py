import numpy as np

from ..model import Customer, Instance, Solution, Vehicle


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
