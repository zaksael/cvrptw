import random
from pathlib import Path

import numpy as np
import pytest

from cvrptw.io import calculate_distances, load_instance
from cvrptw.model import Customer, Instance
from cvrptw.operators import check_route
from cvrptw.solver import get_greedy_solution, ils

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


def test_ils_stats_structure():
    """ILS stats have one entry per iteration with consistent, non-negative values."""
    random.seed(42)
    depot = Customer(0,  0, 0,  0, 0, 1000, 0)
    c1    = Customer(1, 10, 0, 10, 0,  800, 5)
    c2    = Customer(2, 20, 0, 10, 0,  800, 5)
    c3    = Customer(3,  0,10, 10, 0,  800, 5)
    customers = [depot, c1, c2, c3]
    inst = Instance(
        n_vehicles=3,
        capacity=20,
        customers=customers,
        distances=calculate_distances(customers),
    )
    greedy = get_greedy_solution(inst)
    n_iters, _, stats = ils(greedy, max_ls_attempts=2_000, n_perturbation_moves=2, time_limit=2)

    assert len(stats) == n_iters

    prev_elapsed = -1.0
    prev_dist = greedy.distance
    for s in stats:
        # timing is non-negative and elapsed is non-decreasing
        assert s.elapsed_s >= 0
        assert s.ls_time_s >= 0
        assert s.perturb_time_s >= 0
        assert s.elapsed_s >= prev_elapsed
        # gains are non-negative
        assert s.cross_gain >= 0
        assert s.intra_relocate_gain >= 0
        assert s.exchange_gain >= 0
        assert s.intra_or_opt_gain >= 0
        assert s.or_opt_gain >= 0
        assert s.relocate_gain >= 0
        # improved flag is consistent with distance change
        if s.improved:
            assert s.distance < prev_dist - 1e-4
        prev_elapsed = s.elapsed_s
        prev_dist = s.distance


def test_ils_preserves_all_customers():
    """ILS must not silently drop any customers across perturbation and local search."""
    random.seed(42)
    depot = Customer(0,  0, 0,  0, 0, 1000, 0)
    c1    = Customer(1, 10, 0, 10, 0,  800, 5)
    c2    = Customer(2, 20, 0, 10, 0,  800, 5)
    c3    = Customer(3,  0,10, 10, 0,  800, 5)
    c4    = Customer(4,  0,20, 10, 0,  800, 5)
    customers = [depot, c1, c2, c3, c4]
    inst = Instance(
        n_vehicles=4,
        capacity=20,
        customers=customers,
        distances=calculate_distances(customers),
    )
    greedy = get_greedy_solution(inst)
    _, final, _ = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=2)
    assert not final.missing_customers(inst)


def test_greedy_picks_nearest_when_ready_time_zero():
    """Greedy score was distance * ready_time * due_date.

    When ready_time==0 all scores collapse to 0 and argmin picks the first
    candidate regardless of distance. The fix uses (ready_time+1) so distance
    still drives selection. c_far is listed first so the old bug picks it;
    the fix picks c_near.
    """
    from cvrptw.io import calculate_distances

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
    assert first_stop is c_near
