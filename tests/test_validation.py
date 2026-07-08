import numpy as np

from conftest import ids
from cvrptw.model import Customer, Instance, Solution, Vehicle
from cvrptw.operators import check_route, check_route_from, verify_solution


def test_check_route_valid(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    valid, v = check_route([depot, c1, c2, depot], capacity, distances)
    assert valid
    assert v.length() == 4


def test_check_route_infeasible_capacity(tiny):
    customers, distances, _ = tiny
    depot, c1, c2, c3 = customers
    # All three customers exceed capacity=20
    valid, _ = check_route([depot, c1, c2, c3, depot], capacity=20, distances=distances)
    assert not valid


def test_check_route_from_matches_full_check(tiny):
    """check_route_from resuming mid-route gives the same result as full check_route."""
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    route = [depot, c1, c2, c3, depot]
    _, src = check_route([depot, c1, c2], capacity, distances)

    ok_full, v_full = check_route(route, capacity, distances)
    ok_fast, v_fast = check_route_from(route[3:], src, prefix_end=2)

    assert ok_fast == ok_full
    assert abs(v_fast.distance() - v_full.distance()) < 1e-9
    assert ids(v_fast.route.customers) == ids(v_full.route.customers)


def test_check_route_from_preserves_prefix_state(tiny):
    """Prefix state in the result matches src exactly up to prefix_end."""
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    _, src = check_route([depot, c1], capacity, distances)
    prefix_end = 1

    ok, v = check_route_from([c2, c3, depot], src, prefix_end)

    assert ok
    assert v.route.customers[:2] == src.route.customers[:2]
    assert v.route.time_points[:2] == src.route.time_points[:2]
    assert v.route.demand_used[:2] == src.route.demand_used[:2]
    assert v.route.dist_used[:2] == src.route.dist_used[:2]


def test_check_route_from_detects_capacity_infeasibility(tiny):
    """check_route_from returns False when the modified suffix exceeds capacity."""
    customers, distances, _ = tiny
    depot, c1, c2, c3 = customers
    # capacity=25: after prefix [depot, c1] (demand=10), left=15; c2+c3=20 > 15
    capacity = 25
    _, src = check_route([depot, c1], capacity, distances)
    ok, _ = check_route_from([c2, c3, depot], src, prefix_end=1)
    assert not ok


def test_check_route_from_empty_suffix(tiny):
    """Prefix_end at last real stop: suffix is just the closing depot, always valid."""
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    route = [depot, c1, c2, c3, depot]
    _, src = check_route(route, capacity, distances)

    # resume from just before the closing depot
    ok, v = check_route_from(route[5:], src, prefix_end=4)

    assert ok
    assert ids(v.route.customers) == ids(route)
    assert abs(v.distance() - src.distance()) < 1e-9


def test_check_route_from_detects_time_window_infeasibility(tiny):
    """check_route_from returns False when the suffix violates a time window."""
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    # Build a tight-window customer that can't be reached in time after c1
    n = len(customers)
    tight = Customer(cust_id=n, x=50, y=50, demand=1, ready_time=0, due_date=1, service_time=0)
    big_dist = np.pad(distances, ((0, 1), (0, 1)), constant_values=999.0)
    big_dist[n][n] = 0.0

    _, src = check_route([depot, c1], capacity, big_dist)
    ok, _ = check_route_from([tight, depot], src, prefix_end=1)
    assert not ok


def test_check_route_from_rejects_when_waiting_pushes_return_past_depot_closing():
    """The depot-return lookahead must use the ready_time-clamped arrival.

    Arrive at c at t=10, wait until ready_time=90, service to 95, return at
    105 > depot due_date 100 — infeasible even though the un-clamped check
    (10+5+10=25 <= 100) would pass. Same scenario as the Vehicle-level
    regression test, exercised through the hot-path validator.
    """
    depot = Customer(0, 0, 0, 0, 0, 100, 0)
    c = Customer(1, 10, 0, 5, 90, 95, 5)
    distances = np.array([[0., 10.], [10., 0.]])

    _, src = check_route([depot], capacity=100, distances=distances)
    ok, res = check_route_from([c, depot], src, prefix_end=0)
    assert not ok
    assert res is src                                   # failure returns src untouched


def _tiny_instance(tiny, n_vehicles=3):
    customers, distances, capacity = tiny
    return Instance(n_vehicles=n_vehicles, capacity=capacity,
                    customers=customers, distances=distances)


def _closed_vehicle(tiny, *stops):
    customers, distances, capacity = tiny
    v = Vehicle(capacity, customers[0], distances)
    for c in stops:
        v.visit(c)
    v.visit(customers[0])
    return v


def test_verify_solution_valid(tiny):
    customers, _, _ = tiny
    _, c1, c2, c3 = customers
    sol = Solution([_closed_vehicle(tiny, c1, c2), _closed_vehicle(tiny, c3)])
    assert verify_solution(sol, _tiny_instance(tiny)) == []


def test_verify_solution_flags_vehicle_limit(tiny):
    customers, _, _ = tiny
    _, c1, c2, c3 = customers
    sol = Solution([_closed_vehicle(tiny, c) for c in (c1, c2, c3)])
    problems = verify_solution(sol, _tiny_instance(tiny, n_vehicles=2))
    assert any('vehicle limit' in p for p in problems)


def test_verify_solution_flags_capacity_violation(tiny):
    """visit() is unconditional, so an over-capacity route can be built; the
    full-rebuild check must catch it."""
    customers, _, _ = tiny
    _, c1, c2, c3 = customers
    sol = Solution([_closed_vehicle(tiny, c1, c2, c3)])  # demand 30 > capacity replayed below
    inst = _tiny_instance(tiny)
    inst.capacity = 20
    problems = verify_solution(sol, inst)
    assert any('infeasible' in p for p in problems)


def test_verify_solution_flags_duplicate_and_missing(tiny):
    customers, _, _ = tiny
    _, c1, _, c3 = customers
    sol = Solution([_closed_vehicle(tiny, c1), _closed_vehicle(tiny, c1)])
    problems = verify_solution(sol, _tiny_instance(tiny))
    assert any('visited in routes' in p for p in problems)
    assert any('missing' in p for p in problems)


def test_verify_solution_flags_corrupted_cached_distance(tiny):
    customers, _, _ = tiny
    _, c1, c2, _ = customers
    v = _closed_vehicle(tiny, c1, c2)
    v.route._distance += 5.0
    sol = Solution([v, _closed_vehicle(tiny, customers[3])])
    problems = verify_solution(sol, _tiny_instance(tiny))
    assert any('cached distance' in p for p in problems)


def test_verify_solution_flags_route_not_anchored_at_depot(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)  # never closed to depot
    sol = Solution([v, _closed_vehicle(tiny, c2), _closed_vehicle(tiny, c3)])
    problems = verify_solution(sol, _tiny_instance(tiny))
    assert any('does not start and end at the depot' in p for p in problems)
