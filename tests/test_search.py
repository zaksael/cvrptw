import random
import time

import numpy as np

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.search import local_search, perturbation


def test_perturbation_relocates_into_two_stop_route():
    """inter_relocate must try j up to v2.length()-1 (inclusive).

    When the target vehicle has only [depot, depot] (length=2), the only valid
    insertion position is j=1. The old bug used range(1, v2.length()-1) which
    collapsed to range(1,1)=[] and found no moves. The fix uses range(1, v2.length()).
    """
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c1 = Customer(1, 10, 0, 1, 0, 1000, 0)
    distances = np.array([[0., 10.], [10., 0.]])

    v_source = Vehicle(10, depot, distances)
    v_source.visit(c1)
    v_source.visit(depot)                      # route=[depot, c1, depot], length=3

    v_target = Vehicle(10, depot, distances)
    v_target.visit(depot)                      # route=[depot, depot], length=2

    random.seed(42)
    sol = Solution([v_source, v_target])
    changed, _, _ = perturbation(sol, n_moves=1)
    assert changed


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
    assert stats.intra_or_opt_improvements > 0
    assert stats.intra_relocate_improvements == 0
    assert stats.two_opt_improvements == 0
    assert stats.cross_improvements == 0
    assert stats.exchange_improvements == 0


def test_local_search_stops_at_deadline():
    """local_search exits via _LimitReached when deadline is already past."""
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
