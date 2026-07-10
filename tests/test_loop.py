import itertools
import random
from types import SimpleNamespace

import pytest

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Instance
from cvrptw.operators import verify_solution
from cvrptw.search import OPERATOR_NAMES, local_search
from cvrptw.solver import (
    IterationStats, get_greedy_solution, ils, ls_attempts_and_time_limit,
    stop_after_from_stats, summarize_operator_stats,
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


def _regression_instance() -> Instance:
    """The fixed 8-customer instance of the seeded regression tripwire;
    greedy is suboptimal on it (158.11 -> 132.33 over 21 seeded iterations)."""
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
    return Instance(n_vehicles=3, capacity=50, customers=customers,
                    distances=calculate_distances(customers))


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
    inst = _regression_instance()
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


def test_ils_ungates_ls_after_successful_elimination(monkeypatch):
    """The local-search pass right after a successful route elimination must
    run exhaustively (neighbors=None) even when n_neighbors is set — the
    feasibility-only reinsertions need repair moves the gate would filter
    out. All other passes must receive the built neighbor sets.

    Same forced-non-improving setup as the throttle test above; elimination
    successes are faked (returning the solution unchanged) so which
    iterations "succeed" is fully scripted.
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

    success_iters = {2, 5}  # 1-indexed elimination calls that fake a success
    elim_outcomes: list[bool] = []

    def scripted_eliminate(sol, rng=random):
        ok = len(elim_outcomes) + 1 in success_iters
        elim_outcomes.append(ok)
        return ok, sol

    real_ls = loop_mod.local_search
    neighbors_seen = []

    def recording_ls(sol, **kwargs):
        neighbors_seen.append(kwargs.get('neighbors'))
        return real_ls(sol, **kwargs)

    monkeypatch.setattr(loop_mod, 'try_eliminate_route', scripted_eliminate)
    monkeypatch.setattr(loop_mod, 'local_search', recording_ls)

    greedy = get_greedy_solution(inst)
    made_iters, _, _ = ils(greedy, max_ls_attempts=2_000, n_perturbation_moves=2,
                           time_limit=5, max_elim_failures=None, n_neighbors=2,
                           rng=random.Random(42))

    # max_elim_failures=None → one elimination call per iteration, so the
    # i-th LS pass pairs with the i-th elimination outcome
    assert made_iters > max(success_iters)
    assert len(neighbors_seen) == len(elim_outcomes) == made_iters
    for eliminated, neighbors in zip(elim_outcomes, neighbors_seen):
        if eliminated:
            assert neighbors is None
        else:
            assert neighbors is not None


def _run_and_replay(max_ls_attempts: int):
    """Run seeded ils on the regression instance, then replay it via
    stop_after_from_stats with a fresh copy of the same seed."""
    inst = _regression_instance()
    greedy = get_greedy_solution(inst)
    args = dict(max_ls_attempts=max_ls_attempts, n_perturbation_moves=2, time_limit=60)
    n1, best1, stats1 = ils(greedy, **args, rng=random.Random(42))
    n2, best2, stats2 = ils(greedy, **args, rng=random.Random(42),
                            stop_after=stop_after_from_stats(stats1))
    return inst, (n1, best1, stats1), (n2, best2, stats2)


def _assert_identical_runs(inst, run1, run2):
    n1, best1, stats1 = run1
    n2, best2, stats2 = run2
    assert n2 == n1
    assert len(best2) == len(best1)
    assert best2.distance == pytest.approx(best1.distance)
    assert [[c.cust_id for c in v.route.customers] for v in best2] \
        == [[c.cust_id for c in v.route.customers] for v in best1]
    assert [s.ls_attempts for s in stats2] == [s.ls_attempts for s in stats1]
    assert [s.distance for s in stats2] == [s.distance for s in stats1]
    assert verify_solution(best2, inst) == []


def test_stop_after_replays_streak_ended_run():
    """Replaying a run that ended on the failure streak reproduces it
    bit-for-bit, including the final iteration's attempt-capped LS call."""
    inst, run1, run2 = _run_and_replay(max_ls_attempts=50_000)
    _assert_identical_runs(inst, run1, run2)


def test_stop_after_replays_attempt_cut_run():
    """Same when every iteration's LS is cut by LimitReached (tiny budget)."""
    inst, run1, run2 = _run_and_replay(max_ls_attempts=200)
    _assert_identical_runs(inst, run1, run2)


def test_stop_after_stops_at_exact_iteration():
    inst = _regression_instance()
    greedy = get_greedy_solution(inst)
    n_iters, _, stats = ils(greedy, max_ls_attempts=50_000, n_perturbation_moves=2,
                            time_limit=60, rng=random.Random(42), stop_after=(3, 10**9))
    assert n_iters == 3
    assert len(stats) == 3


def test_deadline_cut_is_identical_to_attempt_cut(monkeypatch):
    """The clock-free replay contract: a local_search pass cut by the
    deadline stops at the same candidate — with the same result — as one cut
    by max_attempts at the recorded tick. The deadline is checked when
    n_attempts & 63 == 1, so a fake clock crossing it on its second reading
    fires at tick 65."""
    inst = _regression_instance()
    sol = get_greedy_solution(inst)

    calls = itertools.count(1)
    monkeypatch.setattr('cvrptw.search.budget.time',
                        SimpleNamespace(perf_counter=lambda: next(calls)))
    _, res_deadline, stats_deadline = local_search(
        sol, max_attempts=10_000, deadline=1.5, rng=random.Random(7))
    assert stats_deadline.n_attempts == 65

    _, res_attempts, stats_attempts = local_search(
        sol, max_attempts=65, deadline=None, rng=random.Random(7))
    assert stats_attempts.n_attempts == 65
    assert res_attempts.distance == res_deadline.distance
    assert [[c.cust_id for c in v.route.customers] for v in res_attempts] \
        == [[c.cust_id for c in v.route.customers] for v in res_deadline]
    assert stats_attempts.improvements == stats_deadline.improvements
