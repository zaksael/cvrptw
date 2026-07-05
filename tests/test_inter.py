from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.search.budget import AttemptBudget
from cvrptw.search.inter import apply_operator, apply_or_opt, apply_relocate, cross_suffix, exchange_suffix


def _symmetric_two_vehicle_solution() -> Solution:
    """Two mirrored single-customer routes, one on each side of the depot --
    already distance-optimal, no cross-route move can improve either."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    p1 = Customer(1, 10, 0, 1, 0, 9999, 0)
    p2 = Customer(2, -10, 0, 1, 0, 9999, 0)
    customers = [depot, p1, p2]
    distances = calculate_distances(customers)

    v1 = Vehicle(10, depot, distances)
    v1.visit(p1)
    v1.visit(depot)
    v2 = Vehicle(10, depot, distances)
    v2.visit(p2)
    v2.visit(depot)
    return Solution([v1, v2])


def test_apply_relocate_finds_improving_move():
    """Same two-vehicle geometry as test_local_search.py::test_local_search_finds_relocate_move."""
    depot = Customer(0, 24, 29, 0, 0, 1000, 0)
    c1 = Customer(1, 38, 3, 1, 0, 1000, 0)
    c2 = Customer(2, 31, 4, 1, 0, 1000, 0)
    c3 = Customer(3, 3, 3, 1, 0, 1000, 0)
    c4 = Customer(4, 7, 12, 1, 0, 1000, 0)
    c5 = Customer(5, 26, 30, 1, 0, 1000, 0)
    c6 = Customer(6, 40, 24, 1, 0, 1000, 0)
    c7 = Customer(7, 30, 26, 1, 0, 1000, 0)
    customers = [depot, c1, c2, c3, c4, c5, c6, c7]
    distances = calculate_distances(customers)

    v1 = Vehicle(5, depot, distances)
    for c in [c1, c2, c3, c4]:
        v1.visit(c)
    v1.visit(depot)

    v2 = Vehicle(4, depot, distances)
    for c in [c5, c6, c7]:
        v2.visit(c)
    v2.visit(depot)

    sol = Solution([v1, v2])
    changed, result, gain = apply_relocate(sol, AttemptBudget(max_attempts=3000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_apply_relocate_no_improvement():
    sol = _symmetric_two_vehicle_solution()
    changed, result, gain = apply_relocate(sol, AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol


def test_apply_or_opt_finds_improving_chain_move():
    """A 2-customer chain [x, y] sits in v1 but geometrically belongs next to
    v2's customer b; relocating the chain together shortens both routes."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    a = Customer(1, 10, 0, 1, 0, 9999, 0)
    x = Customer(2, 50, 50, 1, 0, 9999, 0)
    y = Customer(3, 55, 50, 1, 0, 9999, 0)
    b = Customer(4, 50, 45, 1, 0, 9999, 0)
    customers = [depot, a, x, y, b]
    distances = calculate_distances(customers)

    v1 = Vehicle(10, depot, distances)
    for c in [a, x, y]:
        v1.visit(c)
    v1.visit(depot)
    v2 = Vehicle(10, depot, distances)
    v2.visit(b)
    v2.visit(depot)

    sol = Solution([v1, v2])
    changed, result, gain = apply_or_opt(sol, AttemptBudget(max_attempts=5000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_apply_or_opt_no_improvement():
    sol = _symmetric_two_vehicle_solution()
    changed, result, gain = apply_or_opt(sol, AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol


def test_apply_operator_cross_finds_improving_swap():
    """v1's and v2's route suffixes are spatially crossed; swapping them
    (the `cross` transform) shortens total distance."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    a1 = Customer(1, 10, 0, 1, 0, 9999, 0)
    a2 = Customer(2, 20, 10, 1, 0, 9999, 0)
    b1 = Customer(3, 0, 10, 1, 0, 9999, 0)
    b2 = Customer(4, 10, 20, 1, 0, 9999, 0)
    customers = [depot, a1, a2, b1, b2]
    distances = calculate_distances(customers)

    v1 = Vehicle(10, depot, distances)
    for c in [a1, a2]:
        v1.visit(c)
    v1.visit(depot)
    v2 = Vehicle(10, depot, distances)
    for c in [b1, b2]:
        v2.visit(c)
    v2.visit(depot)

    sol = Solution([v1, v2])
    changed, result, gain = apply_operator(sol, cross_suffix, with_last=True, budget=AttemptBudget(max_attempts=3000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_apply_operator_exchange_finds_improving_swap():
    """b (in v1) geometrically belongs near v2's cluster and d (in v2)
    belongs near v1's cluster; swapping the two single customers improves both."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    a = Customer(1, 10, 0, 1, 0, 9999, 0)
    b = Customer(2, 50, 50, 1, 0, 9999, 0)
    c = Customer(3, 50, 45, 1, 0, 9999, 0)
    d = Customer(4, 10, 5, 1, 0, 9999, 0)
    customers = [depot, a, b, c, d]
    distances = calculate_distances(customers)

    v1 = Vehicle(10, depot, distances)
    for x in [a, b]:
        v1.visit(x)
    v1.visit(depot)
    v2 = Vehicle(10, depot, distances)
    for x in [c, d]:
        v2.visit(x)
    v2.visit(depot)

    sol = Solution([v1, v2])
    changed, result, gain = apply_operator(sol, exchange_suffix, with_last=False, budget=AttemptBudget(max_attempts=3000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_apply_operator_cross_no_improvement():
    sol = _symmetric_two_vehicle_solution()
    changed, result, gain = apply_operator(sol, cross_suffix, with_last=True, budget=AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol


def test_apply_operator_exchange_no_improvement():
    sol = _symmetric_two_vehicle_solution()
    changed, result, gain = apply_operator(sol, exchange_suffix, with_last=False, budget=AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol
