from pathlib import Path

import pytest

from cvrptw.io import calculate_distances, load_instance
from cvrptw.model import Customer, Instance, Solution
from cvrptw.operators import check_route
from cvrptw.solver import get_greedy_solution, insert_missing, run_vehicle

from conftest import ids

INSTANCES = Path(__file__).parent.parent / 'data' / 'instances' / 'solomon'
C108 = INSTANCES / 'c108.txt'


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


def _tiny_instance(n_vehicles: int, customers: list[Customer]) -> Instance:
    return Instance(
        n_vehicles=n_vehicles,
        capacity=30,
        customers=customers,
        distances=calculate_distances(customers),
    )


@pytest.fixture
def collinear():
    """Depot + 4 collinear customers on the x-axis, demand 10 each,
    capacity 30 — one vehicle holds at most three."""
    depot = Customer(0,  0, 0,  0, 0, 1000, 0)
    return [depot] + [Customer(i, 10 * i, 0, 10, 0, 500, 5) for i in range(1, 5)]


def test_insert_missing_picks_cheapest_position(collinear):
    """c2 (x=20) is missing from [0, 1, 3, 0]; inserting between c1 and c3
    is free (collinear), so it must beat both other positions and the
    fresh-vehicle round trip (40)."""
    depot, c1, c2, c3, _ = collinear
    inst = _tiny_instance(2, [depot, c1, c2, c3])
    _, v = check_route([depot, c1, c3, depot], inst.capacity, inst.distances)

    repaired = insert_missing(Solution([v]), inst)

    assert len(repaired.vehicles) == 1
    assert ids(repaired.vehicles[0].route.customers) == [0, 1, 2, 3, 0]
    assert repaired.missing_customers(inst) == set()


def test_insert_missing_opens_fresh_vehicle_when_no_route_fits(collinear):
    """The single route is capacity-full, but the vehicle limit allows one more."""
    depot, c1, c2, c3, c4 = collinear
    inst = _tiny_instance(2, collinear)
    _, v = check_route([depot, c1, c2, c3, depot], inst.capacity, inst.distances)

    repaired = insert_missing(Solution([v]), inst)

    assert len(repaired.vehicles) == 2
    assert ids(repaired.vehicles[1].route.customers) == [0, 4, 0]
    assert repaired.missing_customers(inst) == set()


def test_insert_missing_leaves_unplaceable_customer_missing(collinear):
    """Capacity-full route and no vehicle budget left: the customer stays
    missing rather than crashing or violating feasibility."""
    depot, c1, c2, c3, c4 = collinear
    inst = _tiny_instance(1, collinear)
    _, v = check_route([depot, c1, c2, c3, depot], inst.capacity, inst.distances)

    repaired = insert_missing(Solution([v]), inst)

    assert len(repaired.vehicles) == 1
    assert repaired.missing_customers(inst) == {4}


def test_run_vehicle_seed_is_visited_first(collinear):
    """Greedy would start at c1 (nearest); seeding c3 forces it first."""
    depot, c1, c2, c3, _ = collinear
    inst = _tiny_instance(2, [depot, c1, c2, c3])

    v = run_vehicle([c1, c2, c3], inst, seed=c3)

    assert ids(v.route.customers)[1] == 3


def test_run_vehicle_skips_infeasible_seed():
    """A seed unreachable within its own time window is skipped, not crashed on."""
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    ok = Customer(1, 10, 0, 10, 0, 500, 5)
    impossible = Customer(2, 100, 0, 10, 0, 50, 5)  # depot distance 100 > due 50
    inst = _tiny_instance(1, [depot, ok, impossible])

    v = run_vehicle([ok, impossible], inst, seed=impossible)

    assert 2 not in ids(v.route.customers)
    assert 1 in ids(v.route.customers)


def test_greedy_gives_up_gracefully_when_coverage_impossible(collinear):
    """4 customers x demand 10, one vehicle of capacity 30: one customer can
    never be served. Seeded retries must terminate and return the
    best-coverage solution instead of looping or crashing."""
    inst = _tiny_instance(1, collinear)
    sol = get_greedy_solution(inst)
    assert len(sol) == 1
    assert len(sol.missing_customers(inst)) == 1


@pytest.mark.parametrize('name', ['r101', 'r102'])
def test_greedy_covers_all_customers_on_tight_instances(name):
    """Regression: on r101/r102 run_vehicle exhausts all 25 vehicles with one
    customer left over (found by the 2026-07-06 calibration run); the
    insert_missing repair must place it."""
    inst = load_instance(INSTANCES / f'{name}.txt')
    sol = get_greedy_solution(inst)
    assert sol.missing_customers(inst) == set()
    assert len(sol) <= inst.n_vehicles
    for v in sol:
        valid, _ = check_route(v.route.customers, inst.capacity, inst.distances)
        assert valid


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
