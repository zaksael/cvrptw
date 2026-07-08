from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.search._util import build_neighbor_sets
from cvrptw.search.budget import AttemptBudget
from cvrptw.search.inter import (
    apply_operator, apply_or_opt, apply_relocate,
    cross_gate, cross_suffix, exchange_gate, exchange_suffix,
)


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


def _build_solution(customers, route1, route2, cap1=10, cap2=10):
    distances = calculate_distances(customers)
    depot = customers[0]
    v1 = Vehicle(cap1, depot, distances)
    for c in route1:
        v1.visit(c)
    v1.visit(depot)
    v2 = Vehicle(cap2, depot, distances)
    for c in route2:
        v2.visit(c)
    v2.visit(depot)
    return Solution([v1, v2]), distances


def _relocate_improving_solution():
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
    return _build_solution(customers, [c1, c2, c3, c4], [c5, c6, c7], cap1=5, cap2=4)


def _or_opt_improving_solution():
    """A 2-customer chain [x, y] sits in v1 but geometrically belongs next to
    v2's customer b; relocating the chain together shortens both routes."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    a = Customer(1, 10, 0, 1, 0, 9999, 0)
    x = Customer(2, 50, 50, 1, 0, 9999, 0)
    y = Customer(3, 55, 50, 1, 0, 9999, 0)
    b = Customer(4, 50, 45, 1, 0, 9999, 0)
    customers = [depot, a, x, y, b]
    return _build_solution(customers, [a, x, y], [b])


def _cross_improving_solution():
    """v1's and v2's route suffixes are spatially crossed; swapping them
    (the `cross` transform) shortens total distance."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    a1 = Customer(1, 10, 0, 1, 0, 9999, 0)
    a2 = Customer(2, 20, 10, 1, 0, 9999, 0)
    b1 = Customer(3, 0, 10, 1, 0, 9999, 0)
    b2 = Customer(4, 10, 20, 1, 0, 9999, 0)
    customers = [depot, a1, a2, b1, b2]
    return _build_solution(customers, [a1, a2], [b1, b2])


def _exchange_improving_solution():
    """b (in v1) geometrically belongs near v2's cluster and d (in v2)
    belongs near v1's cluster; swapping the two single customers improves both."""
    depot = Customer(0, 0, 0, 0, 0, 9999, 0)
    a = Customer(1, 10, 0, 1, 0, 9999, 0)
    b = Customer(2, 50, 50, 1, 0, 9999, 0)
    c = Customer(3, 50, 45, 1, 0, 9999, 0)
    d = Customer(4, 10, 5, 1, 0, 9999, 0)
    customers = [depot, a, b, c, d]
    return _build_solution(customers, [a, b], [c, d])


def test_apply_relocate_finds_improving_move():
    sol, _ = _relocate_improving_solution()
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
    sol, _ = _or_opt_improving_solution()
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
    sol, _ = _cross_improving_solution()
    changed, result, gain = apply_operator(sol, cross_suffix, with_last=True, budget=AttemptBudget(max_attempts=3000))
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_apply_operator_exchange_finds_improving_swap():
    sol, _ = _exchange_improving_solution()
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


# --- granular-neighborhood gating -------------------------------------------
# Empty neighbor sets gate every insertion/swap out (the improving move is no
# longer even evaluated); sets containing every node gate nothing and the
# operator finds the same improving move as the ungated run.


def _assert_gate_blocks_and_permits(apply_fn, sol, n_nodes, **kw):
    blocked = [set() for _ in range(n_nodes)]
    changed, result, gain = apply_fn(sol, budget=AttemptBudget(max_attempts=5000), neighbors=blocked, **kw)
    assert not changed
    assert result is sol

    permissive = [set(range(n_nodes)) - {a} for a in range(n_nodes)]
    changed, result, gain = apply_fn(sol, budget=AttemptBudget(max_attempts=5000), neighbors=permissive, **kw)
    assert changed
    assert gain > 0
    assert result.distance < sol.distance


def test_apply_relocate_neighbor_gate():
    sol, distances = _relocate_improving_solution()
    _assert_gate_blocks_and_permits(apply_relocate, sol, len(distances))


def test_apply_or_opt_neighbor_gate():
    sol, distances = _or_opt_improving_solution()
    _assert_gate_blocks_and_permits(apply_or_opt, sol, len(distances))


def test_apply_operator_cross_neighbor_gate():
    sol, distances = _cross_improving_solution()
    _assert_gate_blocks_and_permits(apply_operator, sol, len(distances),
                                    operator=cross_suffix, gate=cross_gate, with_last=True)


def test_apply_operator_exchange_neighbor_gate():
    sol, distances = _exchange_improving_solution()
    _assert_gate_blocks_and_permits(apply_operator, sol, len(distances),
                                    operator=exchange_suffix, gate=exchange_gate, with_last=False)


def test_apply_relocate_tight_real_neighbors_still_find_the_move():
    """k=2 built from the real distance matrix keeps the improving relocate
    reachable — the relocated customer's nearest nodes flank its new spot."""
    sol, distances = _or_opt_improving_solution()  # single-customer relocate also improves here
    nbrs = build_neighbor_sets(distances, k=2)
    changed, result, gain = apply_relocate(sol, AttemptBudget(max_attempts=5000), neighbors=nbrs)
    assert changed
    assert gain > 0
