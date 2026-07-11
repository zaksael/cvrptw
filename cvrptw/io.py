from pathlib import Path

import numpy as np

from .model import Customer, Instance, Solution, Vehicle


def load_instance(file_path: str | Path) -> Instance:
    customers = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines(), start=1):
            if i in [1, 2, 3, 4, 6, 7, 8, 9]:
                continue
            line = line.strip()
            if not line:
                continue
            if i == 5:
                n_vehicles, capacity = map(int, line.split())
            else:
                cust_id, x, y, demand, ready_time, due_date, service_time = map(int, line.split())
                customers.append(Customer(cust_id, x, y, demand, ready_time, due_date, service_time))
    return Instance(n_vehicles, capacity, customers, distances=calculate_distances(customers))


def calculate_distances(customers: list[Customer]) -> list[list[float]]:
    coords = np.array([(c.x, c.y) for c in customers], dtype=float)
    diffs = coords[:, None, :] - coords[None, :, :]
    # .tolist(): scalar dm[a][b] lookups in the search hot path are several
    # times faster on nested lists than on an ndarray
    return np.hypot(diffs[..., 0], diffs[..., 1]).tolist()


def parse_sintef_routes(text: str) -> list[list[int]]:
    """Extract customer-id routes from a SINTEF detailed-solution file.

    Route lines look like 'Route  1 : 81 78 76' — customer ids only, depot
    implied at both ends. Header/metadata lines are ignored.
    """
    routes = []
    for line in text.splitlines():
        if not line.lstrip().lower().startswith('route'):
            continue
        _, _, ids = line.partition(':')
        routes.append([int(t) for t in ids.split()])
    return routes


def solution_from_routes(routes: list[list[int]], instance: Instance) -> Solution:
    """Build a Solution by replaying customer-id routes through Vehicle.visit.

    Each route is a list of customer ids WITHOUT the depot at either end;
    the depot stops (and all arrival times) are reconstructed here.
    """
    by_id = {c.cust_id: c for c in instance.customers}
    vehicles = []
    for route in routes:
        v = Vehicle(instance.capacity, instance.depot, instance.distances)
        for cust_id in route:
            v.visit(by_id[cust_id])
        v.visit(instance.depot)
        vehicles.append(v)
    return Solution(vehicles)


def load_solution(file_path: str | Path, instance: Instance) -> Solution:
    """Rebuild a Solution from a .sol file written by save_solution.

    Each line's customer ids are replayed through Vehicle.visit against the
    instance, so arrival times are recomputed rather than trusted from the
    file. Blank lines are ignored.
    """
    routes = []
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            # tokens[0:2] is the leading depot stop; ids at even positions,
            # the last of which is the closing depot stop
            routes.append([int(t) for t in tokens[2:-2:2]])
    return solution_from_routes(routes, instance)


def save_solution(file_path: str | Path, sol: Solution) -> None:
    lines = []
    for v in sol:
        parts = [f"{c.cust_id} {t:.4f}" for c, t in zip(v.route.customers, v.route.time_points)]
        lines.append(' '.join(parts))
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
