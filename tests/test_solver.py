from pathlib import Path

import pytest

from cvrptw.io import calculate_distances, read_instance_data
from cvrptw.operators import check_route
from cvrptw.solver import get_greedy_solution

C108 = Path(__file__).parent.parent / 'ils' / 'resources' / 'instances' / 'C108.txt'


@pytest.fixture
def c108():
    n_vehicles, capacity, customers = read_instance_data(C108)
    distances = calculate_distances(customers)
    return n_vehicles, capacity, customers, distances


def test_greedy_covers_all_customers(c108):
    n_vehicles, capacity, customers, distances = c108
    sol = get_greedy_solution(customers, distances, n_vehicles, capacity)
    depot = customers[0]
    visited = {c.cust_id for v in sol for c in v.route if c is not depot}
    expected = {c.cust_id for c in customers[1:]}
    assert visited == expected


def test_greedy_routes_are_feasible(c108):
    n_vehicles, capacity, customers, distances = c108
    sol = get_greedy_solution(customers, distances, n_vehicles, capacity)
    for v in sol:
        valid, _ = check_route(v.route, capacity, distances)
        assert valid, f"Infeasible route: {v}"


def test_greedy_respects_vehicle_limit(c108):
    n_vehicles, capacity, customers, distances = c108
    sol = get_greedy_solution(customers, distances, n_vehicles, capacity)
    assert len(sol) <= n_vehicles
