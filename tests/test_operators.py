from types import SimpleNamespace

import numpy as np

from cvrptw.io import calculate_distances
from cvrptw.model import Customer
from cvrptw.operators import check_route, check_route_from, cross, exchange, or_opt, relocate, segments_cross, two_opt


def ids(route):
    return [c.cust_id for c in route]


def make_v(customers):
    """Minimal stand-in for Vehicle: operators only access .route.customers."""
    return SimpleNamespace(route=SimpleNamespace(customers=customers))


def test_cross_swaps_suffixes(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    # v1: [depot, c1, c2, depot]  v2: [depot, c3, depot]
    # cross(v1, 2, v2, 1) → r1=[depot,c1]+[c3,depot]  r2=[depot]+[c2,depot]
    v1 = make_v([depot, c1, c2, depot])
    v2 = make_v([depot, c3, depot])
    r1, r2 = cross(v1, 2, v2, 1)
    assert ids(r1) == [0, 1, 3, 0]
    assert ids(r2) == [0, 2, 0]


def test_exchange_swaps_single_customers(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    # swap c1 (position 1 in v1) with c2 (position 1 in v2)
    v1 = make_v([depot, c1, depot])
    v2 = make_v([depot, c2, c3, depot])
    r1, r2 = exchange(v1, 1, v2, 1)
    assert ids(r1) == [0, 2, 0]
    assert ids(r2) == [0, 1, 3, 0]


def test_exchange_does_not_mutate_original(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    v1 = make_v([depot, c1, depot])
    v2 = make_v([depot, c2, depot])
    exchange(v1, 1, v2, 1)
    assert ids(v1.route.customers) == [0, 1, 0]         # original unchanged


def test_relocate_moves_customer_between_routes(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    # move c1 (position 1 in v1) into v2 at position 1
    v1 = make_v([depot, c1, c2, depot])
    v2 = make_v([depot, c3, depot])
    r1, r2 = relocate(v1, 1, v2, 1)
    assert ids(r1) == [0, 2, 0]
    assert ids(r2) == [0, 1, 3, 0]


def test_or_opt_moves_segment_between_routes(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    # move segment [c1, c2] (positions 1-2 in v1) into v2 at position 1
    v1 = make_v([depot, c1, c2, depot])
    v2 = make_v([depot, c3, depot])
    r1, r2 = or_opt(v1, 1, 2, v2, 1, reverse=False)
    assert ids(r1) == [0, 0]
    assert ids(r2) == [0, 1, 2, 3, 0]


def test_or_opt_reverses_segment_when_requested(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    v1 = make_v([depot, c1, c2, depot])
    v2 = make_v([depot, c3, depot])
    r1, r2 = or_opt(v1, 1, 2, v2, 1, reverse=True)
    assert ids(r1) == [0, 0]
    assert ids(r2) == [0, 2, 1, 3, 0]


def test_or_opt_does_not_mutate_original(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    v1 = make_v([depot, c1, c2, depot])
    v2 = make_v([depot, c3, depot])
    or_opt(v1, 1, 2, v2, 1, reverse=False)
    assert ids(v1.route.customers) == [0, 1, 2, 0]
    assert ids(v2.route.customers) == [0, 3, 0]


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
    from cvrptw.model import Customer
    import numpy as np

    n = len(customers)
    tight = Customer(cust_id=n, x=50, y=50, demand=1, ready_time=0, due_date=1, service_time=0)
    big_dist = np.pad(distances, ((0, 1), (0, 1)), constant_values=999.0)
    big_dist[n][n] = 0.0

    _, src = check_route([depot, c1], capacity, big_dist)
    route = [depot, c1, tight, depot]
    ok, _ = check_route_from(route, src, prefix_end=1)
    assert not ok


def _c(cust_id, x, y):
    return Customer(cust_id=cust_id, x=x, y=y, demand=0, ready_time=0, due_date=9999, service_time=0)


def test_segments_cross():
    # diagonals of a 10×10 square — proper crossing
    assert segments_cross(_c(0, 0, 10), _c(1, 10, 0), _c(2, 0, 0), _c(3, 10, 10))
    # parallel horizontal segments — no crossing
    assert not segments_cross(_c(0, 0, 0), _c(1, 1, 0), _c(2, 0, 1), _c(3, 1, 1))
    # T-intersection: one endpoint on the other segment, no cross-through
    assert not segments_cross(_c(0, 0, 0), _c(1, 2, 0), _c(2, 1, 0), _c(3, 1, 1))
    # shared endpoint
    assert not segments_cross(_c(0, 0, 0), _c(1, 1, 0), _c(2, 1, 0), _c(3, 2, 0))


def test_two_opt_reverses_segment(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    v = make_v([depot, c1, c2, c3, depot])
    result = two_opt(v, i=1, j=3)
    assert ids(result) == [0, 1, 3, 2, 0]


def test_two_opt_fixes_crossing():
    depot = Customer(cust_id=0, x=5,  y=5,  demand=0, ready_time=0, due_date=9999, service_time=0)
    c1    = Customer(cust_id=1, x=0,  y=10, demand=1, ready_time=0, due_date=9999, service_time=0)
    c2    = Customer(cust_id=2, x=10, y=0,  demand=1, ready_time=0, due_date=9999, service_time=0)
    c3    = Customer(cust_id=3, x=0,  y=0,  demand=1, ready_time=0, due_date=9999, service_time=0)
    c4    = Customer(cust_id=4, x=10, y=10, demand=1, ready_time=0, due_date=9999, service_time=0)
    all_customers = [depot, c1, c2, c3, c4]
    distances = calculate_distances(all_customers)
    # edge c1→c2 and edge c3→c4 cross geometrically
    assert segments_cross(c1, c2, c3, c4)
    route = [depot, c1, c2, c3, c4, depot]
    _, v = check_route(route, capacity=10, distances=distances)
    # two_opt(v, 1, 3) reverses custs[2:4] = [c2, c3] → [c3, c2]
    new_route = two_opt(v, i=1, j=3)
    assert ids(new_route) == [0, 1, 3, 2, 4, 0]
    ok, new_v = check_route_from(new_route, v, prefix_end=1)
    assert ok
    assert new_v.distance() < v.distance()
