import random
import time

import numpy as np

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.search import local_search


def test_local_search_finds_or_opt_move():
    """A 2-customer chain must move together (reversed) to realize the only
    improving move in this configuration.

    This layout and random.seed were found by empirical search: local_search
    converges via a single intra_or_opt move (segment [c2, c3] relocated,
    reversed, next to c4) with intra_relocate/two_opt/cross/exchange all
    finding zero improvements throughout -- proof the move is only reachable
    through the chain operator, not a coincidence of move ordering.
    """
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
    v.visit(depot)                     # route = [depot, c1, c2, c3, c4, c5, depot]

    sol = Solution([v])
    random.seed(1614)
    changed, result, stats = local_search(sol, max_attempts=5_000)

    assert changed
    assert result.distance < sol.distance - 1e-3
    assert stats.improvements['intra_or_opt'] > 0
    assert stats.improvements['intra_relocate'] == 0
    assert stats.improvements['two_opt'] == 0
    assert stats.improvements['cross'] == 0
    assert stats.improvements['exchange'] == 0


def test_local_search_finds_relocate_move():
    """A single customer must move from v1 to v2 to realize the only improving move.

    Both vehicles are capacitated with exactly 1 unit of slack over their own
    demand, so any 2+ customer move (cross's tail swap, or_opt's chain) is
    infeasible -- only a single-customer relocate fits. This layout and seed
    were found by empirical search: local_search converges via one
    apply_relocate move with cross/exchange/intra_relocate/two_opt/or_opt all
    finding zero improvements, proving relocate() is reachable from the
    cascade and not dead code.
    """
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

    v1 = Vehicle(5, depot, distances)          # 4 customers, demand 4, slack 1
    for c in [c1, c2, c3, c4]:
        v1.visit(c)
    v1.visit(depot)

    v2 = Vehicle(4, depot, distances)          # 3 customers, demand 3, slack 1
    for c in [c5, c6, c7]:
        v2.visit(c)
    v2.visit(depot)

    sol = Solution([v1, v2])
    random.seed(4491)
    changed, result, stats = local_search(sol, max_attempts=3_000)

    assert changed
    assert result.distance < sol.distance - 1e-3
    assert stats.improvements['relocate'] == 1
    assert stats.improvements['cross'] == 0
    assert stats.improvements['exchange'] == 0
    assert stats.improvements['intra_relocate'] == 0
    assert stats.improvements['two_opt'] == 0
    assert stats.improvements['intra_or_opt'] == 0
    assert stats.improvements['or_opt'] == 0


def test_local_search_stops_at_deadline():
    """local_search exits via LimitReached when deadline is already past."""
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c1 = Customer(1, 10, 0, 1, 0, 1000, 0)
    c2 = Customer(2, 20, 0, 1, 0, 1000, 0)
    distances = np.array([[0., 10., 20.], [10., 0., 10.], [20., 10., 0.]])

    v = Vehicle(10, depot, distances)
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)

    sol = Solution([v])
    expired = time.time() - 1.0
    _, _, stats = local_search(sol, max_attempts=1_000_000, deadline=expired)
    assert stats.n_attempts == 1
