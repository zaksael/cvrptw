from pathlib import Path

import numpy as np

from .model import Customer, Instance, Solution


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


def save_solution(file_path: str | Path, sol: Solution) -> None:
    lines = []
    for v in sol:
        parts = [f"{c.cust_id} {t:.4f}" for c, t in zip(v.route.customers, v.route.time_points)]
        lines.append(' '.join(parts))
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
