import numpy as np

from cvrptw.model import Customer
from cvrptw.operators import check_route, check_route_from


def ids(route):
    return [c.cust_id for c in route]


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
    ok_fast, v_fast = check_route_from(route, src, prefix_end=2)

    assert ok_fast == ok_full
    assert abs(v_fast.distance() - v_full.distance()) < 1e-9
    assert ids(v_fast.route.customers) == ids(v_full.route.customers)


def test_check_route_from_preserves_prefix_state(tiny):
    """Prefix state in the result matches src exactly up to prefix_end."""
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    _, src = check_route([depot, c1], capacity, distances)
    prefix_end = 1

    ok, v = check_route_from([depot, c1, c2, c3, depot], src, prefix_end)

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
    ok, _ = check_route_from([depot, c1, c2, c3, depot], src, prefix_end=1)
    assert not ok


def test_check_route_from_empty_suffix(tiny):
    """Prefix_end at last real stop: suffix is just the closing depot, always valid."""
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    route = [depot, c1, c2, c3, depot]
    _, src = check_route(route, capacity, distances)

    # resume from just before the closing depot
    ok, v = check_route_from(route, src, prefix_end=4)

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
    route = [depot, c1, tight, depot]
    ok, _ = check_route_from(route, src, prefix_end=1)
    assert not ok
