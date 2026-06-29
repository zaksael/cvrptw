import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

from .io import load_instance, save_solution
from .model import Solution
from .solver import ILSStats, get_greedy_solution, ils, ls_attempts_and_time_limit


@dataclass
class BenchmarkResult:
    name: str
    distance: float
    n_vehicles: int
    n_iters: int
    elapsed: float
    solution: Solution
    stats: ILSStats


def run_instance(
    path: Path | str,
    results_dir: Path | str | None = None,
    perturbation_moves: int = 5,
    verbose: bool = True,
) -> BenchmarkResult:
    path = Path(path)
    print()
    print('*' * 75)
    print(path.name)

    inst = load_instance(path)
    ls_max_moves, time_limit = ls_attempts_and_time_limit(inst.n_vehicles, len(inst.customers))

    start = time.time()
    init_sol = get_greedy_solution(inst)
    n_iters, sol, stats = ils(init_sol, ls_max_moves, perturbation_moves, time_limit, verbose=verbose)
    elapsed = time.time() - start

    print(f'Best distance = {sol.distance:.2f}')
    print(f'{elapsed:.2f}/{time_limit:.2f} sec.')

    if results_dir is not None:
        save_solution(Path(results_dir) / path.with_suffix('.sol').name, sol)

    return BenchmarkResult(
        name=path.name,
        distance=round(sol.distance, 2),
        n_vehicles=len(sol),
        n_iters=n_iters,
        elapsed=round(elapsed, 2),
        solution=sol,
        stats=stats,
    )


def run_benchmark(
    instances_dir: Path | str = _REPO_ROOT / 'data' / 'instances',
    results_dir: Path | str | None = _REPO_ROOT / 'results',
    perturbation_moves: int = 5,
) -> list[BenchmarkResult]:
    paths = sorted(
        p for p in Path(instances_dir).iterdir()
        if p.is_file() and p.suffix.lower() == '.txt'
    )
    return [run_instance(p, results_dir, perturbation_moves) for p in paths]
