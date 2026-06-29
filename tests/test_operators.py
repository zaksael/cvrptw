from types import SimpleNamespace

from cvrptw.operators import check_route, check_route_from, cross, exchange, relocate


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
