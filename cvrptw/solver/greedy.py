import numpy as np

from ..model import Customer, Instance, Solution, Vehicle


def run_vehicle(candidates: list[Customer], instance: Instance) -> Vehicle:
    scores = {c.cust_id: (c.ready_time + 1) * c.due_date for c in candidates}

    def most_suitable(current, candidates):
        values = [instance.distances[current.cust_id][c.cust_id] * scores[c.cust_id]
                  for c in candidates]
        return candidates[np.argmin(values)]

    v = Vehicle(instance.capacity, instance.depot, instance.distances)
    remaining = {c.cust_id: c for c in candidates}
    while remaining:
        feasible = [c for c in remaining.values() if v.can_visit(c)]
        if not feasible:
            break
        candidate = most_suitable(v.route.customers[-1], feasible)
        del remaining[candidate.cust_id]
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
        visited = {c.cust_id for c in v.route.customers}
        candidates = [c for c in candidates if c.cust_id not in visited]
    return Solution(vehicles)
