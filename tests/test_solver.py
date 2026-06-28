from pathlib import Path

import pytest

from cvrptw.io import load_instance
from cvrptw.operators import check_route
from cvrptw.solver import get_greedy_solution

C108 = Path(__file__).parent.parent / 'ils' / 'resources' / 'instances' / 'C108.txt'


@pytest.fixture
def c108():
    return load_instance(C108)


def test_greedy_covers_all_customers(c108):
    sol = get_greedy_solution(c108)
    visited = {c.cust_id for v in sol for c in v.route.customers if c is not c108.depot}
    expected = {c.cust_id for c in c108.customers[1:]}
    assert visited == expected


def test_greedy_routes_are_feasible(c108):
    sol = get_greedy_solution(c108)
    for v in sol:
        valid, _ = check_route(v.route.customers, c108.capacity, c108.distances)
        assert valid, f"Infeasible route: {v}"


def test_greedy_respects_vehicle_limit(c108):
    sol = get_greedy_solution(c108)
    assert len(sol) <= c108.n_vehicles
