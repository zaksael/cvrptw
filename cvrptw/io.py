from sklearn.metrics.pairwise import euclidean_distances

from .model import Customer


def read_instance_data(file_path):
    customers = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines(), start=1):
            if i in [1, 2, 3, 4, 6, 7, 8, 9]:
                continue
            line = line.strip()
            if i == 5:
                n_vehicles, capacity = map(int, line.split())
            else:
                cust_id, x, y, demand, ready_time, due_date, service_time = map(int, line.split())
                customers.append(Customer(cust_id, x, y, demand, ready_time, due_date, service_time))
    return n_vehicles, capacity, customers


def calculate_distances(customers):
    coords = [(c.x, c.y) for c in customers]
    return euclidean_distances(coords, coords)


def save_solution(file_path, sol):
    lines = []
    for v in sol:
        parts = [f"{c.cust_id} {t:.4f}" for c, t in zip(v.route, v.time_points)]
        lines.append(' '.join(parts))
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
