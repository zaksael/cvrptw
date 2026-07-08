import random

import pytest

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Instance
from cvrptw.operators import verify_solution
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
    prev_veh = len(greedy)
    for s in stats:
        # timing is non-negative and elapsed is non-decreasing
        assert s.elapsed_s >= 0
        assert s.ls_time_s >= 0
        assert s.perturb_time_s >= 0
        assert s.elapsed_s >= prev_elapsed
        # gains are non-negative
        assert all(g >= 0 for g in s.gains.values())
        # improved flag is consistent with the hierarchical objective:
        # fewer vehicles, or same vehicles and smaller distance
        if s.improved:
            assert s.n_vehicles < prev_veh or (
                s.n_vehicles == prev_veh and s.distance < prev_dist - 1e-4
            )
        prev_elapsed = s.elapsed_s
        prev_dist = s.distance
        prev_veh = s.n_vehicles


def test_ils_records_stats_for_final_no_change_iteration():
    """The iteration that triggers the nothing-changed break is still recorded.

    Full vehicles (no capacity slack) block perturbation, elimination, and
    every LS move, so ils exits on iteration 1 via the no-change break.
    """
    depot = Customer(0,  0,  0,  0, 0, 1000, 0)
    c1    = Customer(1, 10,  0, 10, 0,  500, 5)
    c2    = Customer(2,  0, 10, 10, 0,  500, 5)
    customers = [depot, c1, c2]
    inst = Instance(n_vehicles=2, capacity=10, customers=customers,
                    distances=calculate_distances(customers))
    greedy = get_greedy_solution(inst)
    n_iters, _, stats = ils(greedy, max_ls_attempts=2_000, n_perturbation_moves=2,
                            time_limit=2, rng=random.Random(0))
    assert n_iters == 1
    assert len(stats) == n_iters
    assert not stats[0].improved


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
    """ILS must not silently drop any customers across perturbation and local
    search, and the final solution must survive the independent full-rebuild
    check (verify_solution trusts none of the search's incremental state)."""
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
    assert verify_solution(final, inst) == []


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


def test_ils_max_failed_iters_bounds_the_run():
    """max_failed_iters caps consecutive non-improving iterations.

    Same forced-non-improving setup as the restart_from_best test above:
    greedy is optimal and c3's vehicle has capacity slack, so every
    iteration fails and the run stops exactly at the cap.
    """
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
    n_iters, _, stats = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2,
                            time_limit=5, max_failed_iters=3, rng=random.Random(42))
    assert n_iters == 3
    assert all(not s.improved for s in stats)


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

    def recording_perturbation(sol, n_moves, rng=random):
        seen_moves.append(n_moves)
        return real_perturbation(sol, n_moves=n_moves, rng=rng)

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
    ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=base, time_limit=5,
        adaptive_perturbation=False)
    assert set(seen_moves) == {base}


def _expected_elim_attempts(n_iters: int, limit: int) -> int:
    """Attempt count when every elimination fails: the counter n climbs one
    per iteration, and an attempt fires iff n < limit or n % limit == 0."""
    return sum(1 for n in range(n_iters) if n < limit or n % limit == 0)


def test_ils_throttles_elimination_after_consecutive_failures(monkeypatch):
    """After max_elim_failures consecutive non-eliminating iterations,
    try_eliminate_route must back off to every max_elim_failures-th
    iteration; a success must reset the back-off; None must never throttle.

    Same forced-non-improving setup as the adaptive_perturbation test above:
    {c1,c2} is capacity-full and {c1,c2,c3} would exceed capacity, so
    elimination always fails, while c3's slack keeps perturbation alive until
    the 20-failed-iterations stop — plenty of iterations past the cap.
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

    real_eliminate = loop_mod.try_eliminate_route
    calls: list[bool] = []

    def recording_eliminate(sol, rng=random):
        ok, out = real_eliminate(sol, rng)
        calls.append(ok)
        return ok, out

    monkeypatch.setattr(loop_mod, 'try_eliminate_route', recording_eliminate)

    limit = 3
    random.seed(42)
    greedy = get_greedy_solution(inst)
    made_iters, _, _ = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=5,
                           max_elim_failures=limit)
    assert made_iters > 2 * limit  # long enough that the back-off actually skips
    assert calls == [False] * _expected_elim_attempts(made_iters, limit)
    assert len(calls) < made_iters

    # None never throttles: one call per iteration
    calls.clear()
    random.seed(42)
    greedy = get_greedy_solution(inst)
    made_iters, _, _ = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=5,
                           max_elim_failures=None)
    assert len(calls) == made_iters > limit

    # a success resets the back-off: fake one on the first call, then delegate
    # to the real (always-failing) function — the pattern restarts after it
    calls.clear()

    def succeed_once_eliminate(sol, rng=random):
        if not calls:
            calls.append(True)
            return True, sol
        ok, out = real_eliminate(sol, rng)
        calls.append(ok)
        return ok, out

    monkeypatch.setattr(loop_mod, 'try_eliminate_route', succeed_once_eliminate)
    random.seed(42)
    greedy = get_greedy_solution(inst)
    made_iters, _, _ = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2, time_limit=5,
                           max_elim_failures=limit)
    assert made_iters > 1 + 2 * limit
    assert calls == [True] + [False] * _expected_elim_attempts(made_iters - 1, limit)


def test_ils_explicit_rng_is_reproducible_and_isolated():
    """ils(rng=random.Random(seed)) must give identical trajectories regardless
    of global random state, and must leave the global state untouched.

    Same capacity-slack setup as the tests above so perturbation stays alive —
    the run actually consumes randomness every iteration.
    """
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

    def run(global_seed):
        random.seed(global_seed)
        greedy = get_greedy_solution(inst)
        state_before = random.getstate()
        n_iters, best, stats = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2,
                                   time_limit=5, rng=random.Random(7))
        assert random.getstate() == state_before  # global stream untouched
        return n_iters, best.distance, [(s.distance, s.perturb_moves) for s in stats]

    assert run(global_seed=1) == run(global_seed=2)


def _iteration_stats(improvements: dict[str, int] | None = None, gains: dict[str, float] | None = None) -> IterationStats:
    return IterationStats(
        distance=0.0, n_vehicles=0, improved=False, ls_attempts=0,
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


def test_ils_seeded_regression():
    """Tripwire: exact best distance for a fixed seed on a fixed instance.

    The run exercises the full stack — greedy, perturbation, elimination
    (3 -> 2 vehicles), the whole LS cascade — over 21 iterations and is
    bit-reproducible. Any change to move discovery order, rng consumption,
    or acceptance logic shifts this value. That can be deliberate (a new
    operator, reordered candidate loops) — update the expected values then —
    but a shift from a supposedly behavior-neutral refactor means the
    trajectory silently drifted.
    """
    depot = Customer(0,  0,  0,  0, 0, 1000, 0)
    cs = [
        Customer(1, 10,  0, 10, 0,  800, 5),
        Customer(2, 20,  5, 10, 0,  800, 5),
        Customer(3, 15, 15, 10, 50, 800, 5),
        Customer(4,  0, 20, 10, 0,  800, 5),
        Customer(5, -10, 10, 10, 0, 800, 5),
        Customer(6, -5, -10, 10, 0, 800, 5),
        Customer(7,  5, -15, 10, 0, 800, 5),
        Customer(8, 18, -5, 10, 0,  800, 5),
    ]
    customers = [depot] + cs
    inst = Instance(n_vehicles=3, capacity=50, customers=customers,
                    distances=calculate_distances(customers))
    greedy = get_greedy_solution(inst)
    assert greedy.distance == pytest.approx(158.11295121898218)

    n_iters, best, _ = ils(greedy, max_ls_attempts=50_000, n_perturbation_moves=2,
                           time_limit=60, rng=random.Random(42))

    assert n_iters == 21
    assert len(best) == 2
    assert best.distance == pytest.approx(132.32764823109753)
    assert verify_solution(best, inst) == []


def test_ils_granular_neighborhood_produces_valid_solution():
    """n_neighbors gates inter-route LS moves; the run must still improve on
    suboptimal greedy and survive the independent full-rebuild check."""
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    a = Customer(1, 10, 0, 1, 0, 1000, 0)
    b = Customer(2, 20, 0, 1, 0, 1000, 0)
    c = Customer(3, 10.5, 5, 1, 0, 1000, 0)
    customers = [depot, a, b, c]
    inst = Instance(n_vehicles=1, capacity=10, customers=customers,
                    distances=calculate_distances(customers))
    greedy = get_greedy_solution(inst)
    _, best, _ = ils(greedy, max_ls_attempts=5_000, n_perturbation_moves=2,
                     time_limit=2, n_neighbors=2, rng=random.Random(42))
    assert best.distance < greedy.distance - 1e-3
    assert verify_solution(best, inst) == []
