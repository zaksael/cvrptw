import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import pytest

from cvrptw.benchmark import BenchmarkResult, run_benchmark, run_instance


def _write_instance(path: Path, n_vehicles: int, capacity: int, customers: list[tuple[int, ...]]) -> None:
    """Write a minimal Solomon-format instance file matching what
    cvrptw.io.load_instance parses: lines 1-4 and 6-9 are skipped, line 5 is
    'n_vehicles capacity', remaining lines are 'id x y demand ready due service'.
    """
    lines = ['header'] * 4
    lines.append(f'{n_vehicles} {capacity}')
    lines += ['header'] * 4
    for c in customers:
        lines.append(' '.join(str(x) for x in c))
    path.write_text('\n'.join(lines) + '\n')


_TINY_CUSTOMERS = [
    (0, 0, 0, 0, 0, 1000, 0),
    (1, 10, 0, 1, 0, 1000, 0),
    (2, 0, 10, 1, 0, 1000, 0),
    (3, 10, 10, 1, 0, 1000, 0),
]


def test_run_instance_writes_results(tmp_path):
    path = tmp_path / 'inst.txt'
    _write_instance(path, n_vehicles=2, capacity=10, customers=_TINY_CUSTOMERS)

    result = run_instance(path, results_dir=tmp_path, verbose=False)

    assert isinstance(result, BenchmarkResult)
    assert result.distance >= 0
    assert result.n_vehicles >= 1
    assert result.n_iters >= 1
    visited = {c.cust_id for v in result.solution for c in v.route.customers}
    assert visited == {0, 1, 2, 3}

    sol_path = tmp_path / 'inst.sol'
    png_path = tmp_path / 'inst.png'
    assert sol_path.exists() and sol_path.stat().st_size > 0
    assert png_path.exists() and png_path.stat().st_size > 0


def test_run_instance_creates_missing_results_dir(tmp_path):
    path = tmp_path / 'inst.txt'
    _write_instance(path, n_vehicles=2, capacity=10, customers=_TINY_CUSTOMERS)
    results_dir = tmp_path / 'out' / 'nested'

    run_instance(path, results_dir=results_dir, verbose=False)

    assert (results_dir / 'inst.sol').exists()
    assert (results_dir / 'inst.png').exists()


def test_run_instance_without_results_dir(tmp_path):
    path = tmp_path / 'inst.txt'
    _write_instance(path, n_vehicles=2, capacity=10, customers=_TINY_CUSTOMERS)

    result = run_instance(path, results_dir=None, verbose=False)

    assert isinstance(result, BenchmarkResult)
    assert result.distance >= 0
    assert result.n_vehicles >= 1

    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {'inst.txt'}


def test_run_instance_warns_when_customers_left_unassigned(tmp_path):
    path = tmp_path / 'inst.txt'
    # one vehicle of capacity 1 can serve only one of the three unit-demand customers
    _write_instance(path, n_vehicles=1, capacity=1, customers=_TINY_CUSTOMERS)

    with pytest.warns(UserWarning, match='unassigned'):
        run_instance(path, results_dir=None, verbose=False)


def test_run_benchmark_iterates_directory_in_sorted_order(tmp_path):
    _write_instance(tmp_path / 'inst_a.txt', n_vehicles=2, capacity=10, customers=_TINY_CUSTOMERS)
    _write_instance(tmp_path / 'inst_b.txt', n_vehicles=2, capacity=10, customers=_TINY_CUSTOMERS)
    (tmp_path / 'notes.md').write_text('not an instance file')

    results = run_benchmark(instances_dir=tmp_path, results_dir=tmp_path, perturbation_moves=2)

    assert len(results) == 2
    names = [r.name for r in results]
    assert names == sorted(names)
    assert set(names) == {'inst_a.txt', 'inst_b.txt'}
