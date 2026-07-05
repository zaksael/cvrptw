from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.search.budget import AttemptBudget
from cvrptw.search.intra import intra_or_opt, intra_relocate, intra_two_opt


def test_intra_relocate_finds_improving_move():
    """Visiting order [c3, c1, c2] backtracks past the origin twice; relocating
    c3 to the end (straight-line order) is the only improving move."""
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c1 = Customer(1, 10, 0, 1, 0, 1000, 0)
    c2 = Customer(2, 20, 0, 1, 0, 1000, 0)
    c3 = Customer(3, 30, 0, 1, 0, 1000, 0)
    customers = [depot, c1, c2, c3]
    distances = calculate_distances(customers)

    v = Vehicle(10, depot, distances)
    for c in [c3, c1, c2]:
        v.visit(c)
    v.visit(depot)
    sol = Solution([v])

    changed, result, gain = intra_relocate(sol, AttemptBudget(max_attempts=2000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_intra_relocate_no_improvement(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    v = Vehicle(capacity, depot, distances)
    for c in [c1, c2, c3]:
        v.visit(c)
    v.visit(depot)
    sol = Solution([v])

    changed, result, gain = intra_relocate(sol, AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol


def test_intra_two_opt_fixes_crossing_route():
    """Same crossing geometry as test_transforms.py::test_two_opt_fixes_crossing,
    but calls intra_two_opt directly instead of driving two_opt/check_route_from by hand."""
    depot = Customer(0, 5, 5, 0, 0, 9999, 0)
    c1 = Customer(1, 0, 10, 1, 0, 9999, 0)
    c2 = Customer(2, 10, 0, 1, 0, 9999, 0)
    c3 = Customer(3, 0, 0, 1, 0, 9999, 0)
    c4 = Customer(4, 10, 10, 1, 0, 9999, 0)
    customers = [depot, c1, c2, c3, c4]
    distances = calculate_distances(customers)

    v = Vehicle(10, depot, distances)
    for c in [c1, c2, c3, c4]:
        v.visit(c)
    v.visit(depot)
    sol = Solution([v])

    changed, result, gain = intra_two_opt(sol, AttemptBudget(max_attempts=2000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_intra_two_opt_no_improvement(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    v = Vehicle(capacity, depot, distances)
    for c in [c1, c2, c3]:
        v.visit(c)
    v.visit(depot)
    sol = Solution([v])

    changed, result, gain = intra_two_opt(sol, AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol


def test_intra_or_opt_finds_improving_chain_move():
    """Same geometry as test_local_search.py::test_local_search_finds_or_opt_move,
    but calls intra_or_opt directly -- no random.seed needed since we aren't
    racing other cascade operators for which one fires first."""
    depot = Customer(0, 10, 23, 0, 0, 1000, 0)
    c1 = Customer(1, 2, 27, 1, 0, 1000, 0)
    c2 = Customer(2, 12, 19, 1, 0, 1000, 0)
    c3 = Customer(3, 27, 5, 1, 0, 1000, 0)
    c4 = Customer(4, 20, 28, 1, 0, 1000, 0)
    c5 = Customer(5, 11, 23, 1, 0, 1000, 0)
    customers = [depot, c1, c2, c3, c4, c5]
    distances = calculate_distances(customers)

    v = Vehicle(50, depot, distances)
    for c in [c1, c2, c3, c4, c5]:
        v.visit(c)
    v.visit(depot)
    sol = Solution([v])

    changed, result, gain = intra_or_opt(sol, AttemptBudget(max_attempts=5000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_intra_or_opt_no_improvement(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2, c3 = customers
    v = Vehicle(capacity, depot, distances)
    for c in [c1, c2, c3]:
        v.visit(c)
    v.visit(depot)
    sol = Solution([v])

    changed, result, gain = intra_or_opt(sol, AttemptBudget(max_attempts=2000))
    assert not changed
    assert gain == 0.0
    assert result is sol
