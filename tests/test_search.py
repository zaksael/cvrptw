import numpy as np

from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.search import perturbation


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

    sol = Solution([v_source, v_target])
    changed, _ = perturbation(sol, n_moves=1)
    assert changed
