import random

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Instance, Solution
from cvrptw.operators import check_route
from cvrptw.search import try_eliminate_route
from cvrptw.solver import ils

from conftest import ids


def _solution(routes: list[list[Customer]], capacity: int, distances) -> Solution:
    vehicles = []
    for route in routes:
        ok, v = check_route(route, capacity, distances)
        assert ok
        vehicles.append(v)
    return Solution(vehicles)


def test_eliminate_moves_smallest_route_into_slack(tiny):
    """[c3] and [c1, c2] under capacity 30: c3 fits into the other route,
    so the single-customer route must be eliminated."""
    random.seed(42)
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    sol = _solution([[depot, c1, c2, depot], [depot, c3, depot]], capacity, distances)

    done, result = try_eliminate_route(sol)

    assert done
    assert len(result.vehicles) == 1
    assert sorted(ids(result.vehicles[0].route.customers)[1:-1]) == [1, 2, 3]
    ok, _ = check_route(result.vehicles[0].route.customers, capacity, distances)
    assert ok


def test_eliminate_fails_without_capacity_slack(tiny):
    """Capacity 20 leaves no room in either direction: the solution must be
    returned unchanged (same object)."""
    random.seed(42)
    customers, distances, _ = tiny
    depot, c1, c2, c3 = customers
    sol = _solution([[depot, c1, c2, depot], [depot, c3, depot]], 20, distances)

    done, result = try_eliminate_route(sol)

    assert not done
    assert result is sol


def test_ils_reduces_vehicles_under_hierarchical_objective():
    """Three single-customer routes, capacity for all three in one vehicle:
    the default (minimize_vehicles=True) ILS must shrink the fleet, record
    the vehicle drop as an improvement, and keep every customer."""
    random.seed(42)
    depot = Customer(0,   0,  0,  0, 0, 1000, 0)
    c1    = Customer(1,  10,  0, 10, 0,  800, 5)
    c2    = Customer(2,   0, 10, 10, 0,  800, 5)
    c3    = Customer(3, -10,  0, 10, 0,  800, 5)
    customers = [depot, c1, c2, c3]
    inst = Instance(
        n_vehicles=3,
        capacity=30,
        customers=customers,
        distances=calculate_distances(customers),
    )
    sol = _solution([[depot, c, depot] for c in (c1, c2, c3)], inst.capacity, inst.distances)

    _, best, stats = ils(sol, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=2)

    assert len(best) == 1
    assert not best.missing_customers(inst)
    assert any(s.improved for s in stats)
    # best-so-far fleet size never grows across iterations
    fleet = [s.n_vehicles for s in stats]
    assert fleet == sorted(fleet, reverse=True)
    assert fleet[-1] == 1


def test_ils_distance_only_flag_ignores_vehicle_count(tiny):
    """minimize_vehicles=False restores the old objective: improvements are
    strictly distance drops."""
    random.seed(42)
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    sol = _solution([[depot, c1, c2, depot], [depot, c3, depot]], capacity, distances)

    _, best, stats = ils(sol, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=2,
                         minimize_vehicles=False)

    prev = sol.distance
    for s in stats:
        if s.improved:
            assert s.distance < prev - 1e-4
        prev = s.distance
    assert best.distance <= sol.distance + 1e-9
