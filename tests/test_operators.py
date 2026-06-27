from types import SimpleNamespace

from cvrptw.operators import check_route, cross, exchange, relocate


def ids(route):
    return [c.cust_id for c in route]


def make_v(route):
    """Minimal stand-in for Vehicle: operators only access .route."""
    return SimpleNamespace(route=route)


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
    assert ids(v1.route) == [0, 1, 0]                  # original unchanged


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
