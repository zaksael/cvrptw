import random

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Instance
from cvrptw.search import OPERATOR_NAMES
from cvrptw.solver import (
    IterationStats, get_greedy_solution, ils, ls_attempts_and_time_limit, summarize_operator_stats,
)


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


def test_ils_improves_on_suboptimal_greedy():
    """ILS must actually find and record improvements when greedy is suboptimal.

    Nearest-neighbor trap: from A(10,0) greedy detours to C(10.5,5) (distance
    ~5.02) before B(20,0), giving depot-A-C-B-depot ~= 45.76, while
    depot-A-B-C-depot ~= 42.37 is better. n_vehicles=1 disables perturbation
    (inter-route relocate needs two routes), so the gain must come from local
    search and be recorded as an improved iteration. verbose=True exercises
    the progress-bar/new-best reporting path.
    """
    random.seed(42)
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    a = Customer(1, 10, 0, 1, 0, 1000, 0)
    b = Customer(2, 20, 0, 1, 0, 1000, 0)
    c = Customer(3, 10.5, 5, 1, 0, 1000, 0)
    customers = [depot, a, b, c]
    inst = Instance(
        n_vehicles=1,
        capacity=10,
        customers=customers,
        distances=calculate_distances(customers),
    )
    greedy = get_greedy_solution(inst)
    assert [cust.cust_id for cust in greedy.vehicles[0].route.customers] == [0, 1, 3, 2, 0]

    _, best, stats = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=2,
                         verbose=True)

    assert best.distance < greedy.distance - 1e-3
    assert any(s.improved for s in stats)
    assert min(s.distance for s in stats) < greedy.distance - 1e-3


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


def test_ils_restart_from_best_survives_failed_iterations():
    """restart_from_best must reset current_sol on every non-improving iteration.

    Greedy is already optimal here ({c1,c2} + {c3} = 60; any other split is
    worse), and c3's vehicle has 10 capacity slack, so perturbation always
    finds a (worsening) relocate and local search at best restores 60 —
    every iteration is a failed one that exercises the restart path, until
    the 20-failed-iterations stop fires.
    """
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
    _, final, stats = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=5,
                          restart_from_best=True)

    assert sum(1 for s in stats if not s.improved) >= 1
    assert final.distance <= greedy.distance + 1e-9
    assert not final.missing_customers(inst)


def test_ils_adaptive_perturbation_escalates_on_failures(monkeypatch):
    """adaptive_perturbation must grow the kick with consecutive failed iterations.

    Same forced-non-improving setup as the restart_from_best test above:
    greedy is optimal and c3's vehicle has capacity slack, so every iteration
    fails and n_failed_iters climbs to the stop threshold. The recorded
    per-iteration strengths must start at the base, follow
    min(base + n_failed_iters, 3 * base), and stay constant with the flag off.
    """
    import cvrptw.solver.loop as loop_mod

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

    real_perturbation = loop_mod.perturbation
    seen_moves: list[int] = []

    def recording_perturbation(sol, n_moves):
        seen_moves.append(n_moves)
        return real_perturbation(sol, n_moves=n_moves)

    monkeypatch.setattr(loop_mod, 'perturbation', recording_perturbation)

    base = 2
    random.seed(42)
    greedy = get_greedy_solution(inst)
    ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=base, time_limit=5,
        adaptive_perturbation=True)

    assert seen_moves[0] == base
    # every iteration fails, so strength must follow the capped escalation exactly
    assert seen_moves == [min(base + failed, 3 * base) for failed in range(len(seen_moves))]
    assert seen_moves[-1] == 3 * base

    # control: with the flag off, the strength never moves
    seen_moves.clear()
    random.seed(42)
    greedy = get_greedy_solution(inst)
    ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=base, time_limit=5)
    assert set(seen_moves) == {base}


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


def test_ls_attempts_and_time_limit_scales_with_instance_size():
    assert ls_attempts_and_time_limit(25, 101) == (250_000, 600)
    assert ls_attempts_and_time_limit(26, 101) == (1_000_000, 1800)   # vehicles over threshold
    assert ls_attempts_and_time_limit(25, 102) == (1_000_000, 1800)   # customers over threshold
