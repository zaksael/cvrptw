import time
from dataclasses import dataclass

from tqdm.auto import tqdm

from ..model import Solution
from ..search import OPERATOR_NAMES, local_search, perturbation, try_eliminate_route


@dataclass
class IterationStats:
    distance: float
    n_vehicles: int
    improved: bool
    ls_attempts: int
    improvements: dict[str, int]
    gains: dict[str, float]
    perturb_moves: int
    elapsed_s: float
    dist_before_ls: float
    ls_time_s: float
    perturb_time_s: float


ILSStats = list[IterationStats]


def summarize_operator_stats(stats: ILSStats) -> dict[str, float]:
    """Sum per-operator improvements/gains across all iterations, keyed
    `{operator}_improvements` / `{operator}_gain`."""
    totals: dict[str, float] = {}
    for name in OPERATOR_NAMES:
        totals[f'{name}_improvements'] = sum(s.improvements[name] for s in stats)
        totals[f'{name}_gain'] = sum(s.gains[name] for s in stats)
    return totals


def ls_attempts_and_time_limit(n_vehicles: int, n_customers: int) -> tuple[int, int]:
    if n_vehicles > 25 or n_customers > 101:
        return 1_000_000, 1800
    return 250_000, 600


_ILS_BAR_FORMAT = '{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}]{postfix}'


def ils(
    sol: Solution,
    max_ls_attempts: int,
    n_perturbation_moves: int,
    time_limit: int,
    verbose: bool = False,
    desc: str = 'ILS',
    restart_from_best: bool = False,
    adaptive_perturbation: bool = True,
    minimize_vehicles: bool = True,
    max_elim_failures: int | None = 5,
) -> tuple[int, Solution, ILSStats]:
    """Iterated local search.

    minimize_vehicles=True (default) pursues the hierarchical Solomon
    objective — fewer vehicles first, then distance: each iteration attempts
    an all-or-nothing route elimination between perturbation and local
    search, and best-so-far is compared lexicographically on
    (vehicles, distance). False restores the plain distance-only objective.
    After max_elim_failures consecutive non-eliminating iterations the step
    backs off to every max_elim_failures-th iteration (on fleet-tight
    instances it mostly burns budget and churns the RNG, but late successes
    after long failure streaks do happen, so it is never fully disabled);
    a success resets the back-off. None never throttles.
    """
    best_sol = current_sol = sol
    best_dist = sol.distance
    made_iters = 0
    n_failed_iters = 0
    n_elim_failures = 0
    stats: ILSStats = []

    start = time.time()
    pbar = tqdm(total=time_limit, desc=desc, unit='s', leave=False, bar_format=_ILS_BAR_FORMAT) if verbose else None
    if verbose:
        tqdm.write(f'Initial : distance = {best_dist:.2f}, vehicles = {len(best_sol)}')
    try:
        while time.time() - start < time_limit and n_failed_iters < 20:
            made_iters += 1
            moves = n_perturbation_moves
            if adaptive_perturbation:
                moves = min(n_perturbation_moves + n_failed_iters, 3 * n_perturbation_moves)
            t0 = time.time()
            p_changed, current_sol, actual_p_moves = perturbation(current_sol, n_moves=moves)
            e_changed = False
            if minimize_vehicles:
                if (max_elim_failures is None or n_elim_failures < max_elim_failures
                        or n_elim_failures % max_elim_failures == 0):
                    e_changed, current_sol = try_eliminate_route(current_sol)
                n_elim_failures = 0 if e_changed else n_elim_failures + 1
            t1 = time.time()
            dist_before_ls = current_sol.distance
            ls_changed, current_sol, ls_stats = local_search(current_sol, max_attempts=max_ls_attempts, deadline=start + time_limit)
            t2 = time.time()

            if pbar is not None:
                pbar.update(min(t2, start + time_limit) - start - pbar.n)
                pbar.set_postfix(best=f'{best_dist:.2f}', iter=made_iters, failed=n_failed_iters)

            if not (p_changed or e_changed or ls_changed):
                break

            current_dist = current_sol.distance
            delta = best_dist - current_dist
            improved = delta > 1e-3
            if minimize_vehicles and len(current_sol) != len(best_sol):
                improved = len(current_sol) < len(best_sol)
            if improved:
                best_sol = current_sol
                best_dist = current_dist
                n_failed_iters = 0
                if verbose:
                    tqdm.write(f"New best: distance = {best_dist:.2f} ({-delta:.2f}), vehicles = {len(best_sol)}")
            else:
                n_failed_iters += 1
                if restart_from_best:
                    current_sol = best_sol

            stats.append(IterationStats(
                distance=round(best_dist, 2),
                n_vehicles=len(best_sol),
                improved=improved,
                ls_attempts=ls_stats.n_attempts,
                improvements=ls_stats.improvements,
                gains={k: round(v, 4) for k, v in ls_stats.gains.items()},
                perturb_moves=actual_p_moves,
                elapsed_s=round(t2 - start, 2),
                dist_before_ls=round(dist_before_ls, 2),
                ls_time_s=round(t2 - t1, 2),
                perturb_time_s=round(t1 - t0, 2),
            ))
    finally:
        if pbar is not None:
            pbar.close()

    return made_iters, best_sol, stats
