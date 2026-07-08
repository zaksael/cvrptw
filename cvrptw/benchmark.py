import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from .io import load_instance, save_solution
from .model import Solution
from .solver import ILSStats, get_greedy_solution, ils, ls_attempts_and_time_limit, summarize_operator_stats
from .viz import draw_solution

_REPO_ROOT = Path(__file__).parent.parent


@dataclass
class BenchmarkResult:
    name: str
    distance: float
    n_vehicles: int
    n_iters: int
    elapsed: float
    solution: Solution
    stats: ILSStats
    init_distance: float
    init_n_vehicles: int
    improvement_pct: float
    total_ls_time_s: float
    total_perturb_time_s: float
    operator_totals: dict[str, float]


def run_instance(
    path: Path | str,
    results_dir: Path | str | None = None,
    perturbation_moves: int = 5,
    verbose: bool = True,
    **ils_kwargs,
) -> BenchmarkResult:
    """Extra keyword arguments are forwarded to ils() — e.g. time_limit,
    max_ls_attempts (defaulted from ls_attempts_and_time_limit when absent),
    minimize_vehicles, max_elim_failures."""
    path = Path(path)

    inst = load_instance(path)
    default_attempts, default_time = ls_attempts_and_time_limit(inst.n_vehicles, len(inst.customers))
    ils_kwargs.setdefault('max_ls_attempts', default_attempts)
    ils_kwargs.setdefault('time_limit', default_time)
    time_limit = ils_kwargs['time_limit']

    start = time.perf_counter()
    init_sol = get_greedy_solution(inst)
    missing = init_sol.missing_customers(inst)
    if missing:
        warnings.warn(
            f'{path.name}: greedy construction left {len(missing)} customers '
            f'unassigned ({sorted(missing)}); reported distance is for an incomplete solution',
            stacklevel=2,
        )
    init_distance = init_sol.distance
    init_n_vehicles = len(init_sol)

    n_iters, sol, stats = ils(init_sol, n_perturbation_moves=perturbation_moves,
                              verbose=verbose, desc=path.name, **ils_kwargs)
    elapsed = time.perf_counter() - start

    improvement_pct = round((init_distance - sol.distance) / init_distance * 100, 2) if init_distance else 0.0
    tqdm.write(
        f'{path.name}: best dist={sol.distance:.2f}, vehicles={len(sol)}, '
        f'{-improvement_pct:.2f}% vs initial, {elapsed:.2f}/{time_limit:.2f} sec.'
    )

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        save_solution(results_dir / path.with_suffix('.sol').name, sol)
        draw_solution(
            sol,
            title=f'{path.name}: dist={sol.distance:.2f}, vehicles={len(sol)}',
            save_path=results_dir / path.with_suffix('.png').name,
        )

    return BenchmarkResult(
        name=path.name,
        distance=round(sol.distance, 2),
        n_vehicles=len(sol),
        n_iters=n_iters,
        elapsed=round(elapsed, 2),
        solution=sol,
        stats=stats,
        init_distance=round(init_distance, 2),
        init_n_vehicles=init_n_vehicles,
        improvement_pct=improvement_pct,
        total_ls_time_s=round(sum(s.ls_time_s for s in stats), 2),
        total_perturb_time_s=round(sum(s.perturb_time_s for s in stats), 2),
        operator_totals=summarize_operator_stats(stats),
    )


def run_benchmark(
    instances_dir: Path | str = _REPO_ROOT / 'data' / 'instances' / 'solomon',
    results_dir: Path | str | None = _REPO_ROOT / 'results',
    perturbation_moves: int = 5,
    **ils_kwargs,
) -> list[BenchmarkResult]:
    """Extra keyword arguments are forwarded to ils() via run_instance."""
    paths = sorted(
        p for p in Path(instances_dir).iterdir()
        if p.is_file() and p.suffix.lower() == '.txt'
    )
    results = []
    with tqdm(paths, desc='ILS', unit='instance') as pbar:
        for p in pbar:
            results.append(run_instance(p, results_dir, perturbation_moves, **ils_kwargs))
    return results
