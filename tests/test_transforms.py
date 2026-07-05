from types import SimpleNamespace

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


def test_two_opt_reverses_segment(tiny):
    customers, _, _ = tiny
    depot, c1, c2, c3 = customers
    v = make_v([depot, c1, c2, c3, depot])
    result = two_opt(v, i=1, j=3)
    assert ids(result) == [0, 1, 3, 2, 0]


def test_two_opt_fixes_crossing():
    from cvrptw.io import calculate_distances
    from cvrptw.model import Customer

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
