import random

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Instance
from cvrptw.search import OPERATOR_NAMES
from cvrptw.solver import IterationStats, get_greedy_solution, ils, summarize_operator_stats


def test_ils_stats_structure():
    """ILS stats have one entry per iteration with consistent, non-negative values."""
    random.seed(42)
    depot = Customer(0,  0, 0,  0, 0, 1000, 0)
    c1    = Customer(1, 10, 0, 10, 0,  800, 5)
    c2    = Customer(2, 20, 0, 10, 0,  800, 5)
    c3    = Customer(3,  0,10, 10, 0,  800, 5)
    customers = [depot, c1, c2, c3]
    inst = Instance(
        n_vehicles=3,
        capacity=20,
        customers=customers,
        distances=calculate_distances(customers),
    )
    greedy = get_greedy_solution(inst)
    n_iters, _, stats = ils(greedy, max_ls_attempts=2_000, n_perturbation_moves=2, time_limit=2)

    assert len(stats) == n_iters

    prev_elapsed = -1.0
    prev_dist = greedy.distance
    for s in stats:
        # timing is non-negative and elapsed is non-decreasing
        assert s.elapsed_s >= 0
        assert s.ls_time_s >= 0
        assert s.perturb_time_s >= 0
        assert s.elapsed_s >= prev_elapsed
        # gains are non-negative
        assert all(g >= 0 for g in s.gains.values())
        # improved flag is consistent with distance change
        if s.improved:
            assert s.distance < prev_dist - 1e-4
        prev_elapsed = s.elapsed_s
        prev_dist = s.distance


def test_ils_preserves_all_customers():
    """ILS must not silently drop any customers across perturbation and local search."""
    random.seed(42)
    depot = Customer(0,  0, 0,  0, 0, 1000, 0)
    c1    = Customer(1, 10, 0, 10, 0,  800, 5)
    c2    = Customer(2, 20, 0, 10, 0,  800, 5)
    c3    = Customer(3,  0,10, 10, 0,  800, 5)
    c4    = Customer(4,  0,20, 10, 0,  800, 5)
    customers = [depot, c1, c2, c3, c4]
    inst = Instance(
        n_vehicles=4,
        capacity=20,
        customers=customers,
        distances=calculate_distances(customers),
    )
    greedy = get_greedy_solution(inst)
    _, final, _ = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=2)
    assert not final.missing_customers(inst)


def _iteration_stats(improvements: dict[str, int] | None = None, gains: dict[str, float] | None = None) -> IterationStats:
    return IterationStats(
        distance=0.0, improved=False, ls_attempts=0,
        improvements=dict.fromkeys(OPERATOR_NAMES, 0) | (improvements or {}),
        gains=dict.fromkeys(OPERATOR_NAMES, 0.0) | (gains or {}),
        perturb_moves=0, elapsed_s=0.0, dist_before_ls=0.0, ls_time_s=0.0, perturb_time_s=0.0,
    )


def test_summarize_operator_stats_sums_across_iterations():
    stats = [
        _iteration_stats(improvements={'cross': 2, 'relocate': 1}, gains={'cross': 5.0, 'relocate': 1.5}),
        _iteration_stats(improvements={'cross': 1, 'or_opt': 1}, gains={'cross': 3.0, 'or_opt': 2.0}),
    ]
    totals = summarize_operator_stats(stats)
    assert totals['cross_improvements'] == 3
    assert totals['cross_gain'] == 8.0
    assert totals['relocate_improvements'] == 1
    assert totals['relocate_gain'] == 1.5
    assert totals['or_opt_improvements'] == 1
    assert totals['or_opt_gain'] == 2.0
    assert totals['exchange_improvements'] == 0
    assert totals['exchange_gain'] == 0.0


def test_summarize_operator_stats_empty():
    totals = summarize_operator_stats([])
    assert totals and all(v == 0 for v in totals.values())
