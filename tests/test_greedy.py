from pathlib import Path

import pytest

from cvrptw.io import calculate_distances, load_instance
from cvrptw.model import Customer, Instance
from cvrptw.operators import check_route
from cvrptw.solver import get_greedy_solution

C108 = Path(__file__).parent.parent / 'data' / 'instances' / 'C108.txt'


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


def test_greedy_routes_return_before_depot_closing_despite_waiting():
    """Waiting at a late-ready customer must not let greedy break the depot
    due date: arrival 10, wait to 90, service to 95, return 105 > 100."""
    depot = Customer(0, 0, 0, 0, 0, 100, 0)
    c = Customer(1, 10, 0, 1, 90, 95, 5)
    customers = [depot, c]
    inst = Instance(
        n_vehicles=1,
        capacity=10,
        customers=customers,
        distances=calculate_distances(customers),
    )

    sol = get_greedy_solution(inst)
    for v in sol:
        assert v.route.time_points[-1] <= depot.due_date


def test_greedy_picks_nearest_when_ready_time_zero():
    """Greedy score was distance * ready_time * due_date.

    When ready_time==0 all scores collapse to 0 and argmin picks the first
    candidate regardless of distance. The fix uses (ready_time+1) so distance
    still drives selection. c_far is listed first so the old bug picks it;
    the fix picks c_near.
    """
    depot  = Customer(0,  0, 0, 0, 0, 1000, 0)
    c_far  = Customer(1, 20, 0, 1, 0, 1000, 0)   # ready_time=0, distance 20 from depot
    c_near = Customer(2,  5, 0, 1, 0, 1000, 0)   # ready_time=0, distance  5 from depot
    customers = [depot, c_far, c_near]
    inst = Instance(
        n_vehicles=2,
        capacity=10,
        customers=customers,
        distances=calculate_distances(customers),
    )

    sol = get_greedy_solution(inst)
    first_stop = sol.vehicles[0].route.customers[1]
    assert first_stop.cust_id == c_near.cust_id
