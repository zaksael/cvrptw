import numpy as np

from ..model import Customer, Instance, Solution, Vehicle
from ..operators import check_route_from


def run_vehicle(candidates: list[Customer], instance: Instance, seed: Customer | None = None) -> Vehicle:
    """Fill one vehicle greedily from candidates.

    seed, when given, is visited first (before greedy selection) — used by
    get_greedy_solution's retry pass to guarantee a slot for customers whose
    time windows are too tight to survive end-append construction. A seed
    that is infeasible even as the sole stop is skipped.
    """
    scores = {c.cust_id: (c.ready_time + 1) * c.due_date for c in candidates}

    def most_suitable(current, candidates):
        values = [instance.distances[current.cust_id][c.cust_id] * scores[c.cust_id]
                  for c in candidates]
        return candidates[np.argmin(values)]

    v = Vehicle(instance.capacity, instance.depot, instance.distances)
    remaining = {c.cust_id: c for c in candidates}
    if seed is not None and v.try_visit(seed):
        del remaining[seed.cust_id]
    while remaining:
        feasible = [c for c in remaining.values() if v.can_visit(c)]
        if not feasible:
            break
        candidate = most_suitable(v.route.customers[-1], feasible)
        del remaining[candidate.cust_id]
        v.visit(candidate)
    v.visit(v.depot)
    return v


def insert_missing(sol: Solution, instance: Instance) -> Solution:
    """Cheapest-insertion repair for customers construction left unassigned.

    run_vehicle only appends at route ends, so on tight instances (e.g. r101)
    it can exhaust the vehicle limit with customers left over; nothing in the
    search can re-add them. For each missing customer this tries every
    insertion position in every route, plus a fresh vehicle while the limit
    allows, and applies the feasible option with the smallest added distance.
    Customers with no feasible option anywhere stay missing (callers warn).
    """
    missing = sol.missing_customers(instance)
    if not missing:
        return sol

    by_id = {c.cust_id: c for c in instance.customers}
    depot = instance.depot
    vehicles = list(sol.vehicles)
    for cust_id in sorted(missing):
        c = by_id[cust_id]
        best_added, best_i, best_vehicle = float('inf'), None, None
        for i, v in enumerate(vehicles):
            route = v.route.customers
            for pos in range(1, len(route)):
                ok, new_v = check_route_from([c] + route[pos:], v, pos - 1)
                if ok and new_v.distance() - v.distance() < best_added:
                    best_added, best_i, best_vehicle = new_v.distance() - v.distance(), i, new_v
        if len(vehicles) < instance.n_vehicles:
            fresh = Vehicle(instance.capacity, depot, instance.distances)
            if (instance.distances[depot.cust_id][cust_id] * 2 < best_added
                    and fresh.try_visit(c)):
                fresh.visit(depot)
                best_i, best_vehicle = None, fresh
        if best_vehicle is not None:
            if best_i is None:
                vehicles.append(best_vehicle)
            else:
                vehicles[best_i] = best_vehicle
    return Solution(vehicles)


_MAX_SEED_RETRIES = 3


def _construct(instance: Instance, seeds: list[Customer] = ()) -> Solution:
    vehicles = []
    candidates = instance.customers[1:]
    seeds = list(seeds)
    while candidates and len(vehicles) < instance.n_vehicles:
        v = run_vehicle(candidates, instance, seed=seeds.pop(0) if seeds else None)
        vehicles.append(v)
        visited = {c.cust_id for c in v.route.customers}
        candidates = [c for c in candidates if c.cust_id not in visited]
    return Solution(vehicles)


def get_greedy_solution(instance: Instance) -> Solution:
    """Greedy construction, guaranteed-coverage best-effort.

    End-append construction can strand customers with tight time windows once
    the vehicle limit is hit (r101/r102 in the 2026-07-06 calibration).
    Two-stage repair: insert_missing tries cheapest insertion into the built
    routes; if customers still don't fit anywhere (r102's customer 38, window
    [83, 93]), construction is retried with the missing customers seeded as
    first stops of their own vehicles, accumulating seeds across retries.
    Returns the best-coverage solution found; run_instance warns if customers
    remain missing after all retries.
    """
    best = cand = insert_missing(_construct(instance), instance)
    best_missing = cand_missing = best.missing_customers(instance)
    seed_ids: set[int] = set()
    by_id = {c.cust_id: c for c in instance.customers}
    for _ in range(_MAX_SEED_RETRIES):
        if not cand_missing:
            break
        seed_ids |= cand_missing
        cand = insert_missing(
            _construct(instance, [by_id[i] for i in sorted(seed_ids)]), instance,
        )
        cand_missing = cand.missing_customers(instance)
        if len(cand_missing) < len(best_missing):
            best, best_missing = cand, cand_missing
    return best
