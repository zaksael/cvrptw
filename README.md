# CVRPTW · Iterated Local Search

A heuristic solver for the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**, written in pure Python (+ NumPy for the distance matrix). Solutions are constructed greedily and then improved with **Iterated Local Search (ILS)** over a cascade of seven route operators.

<p align="center">
  <img src="docs/c108_solution.png" alt="Solved Solomon C108 instance: distance 828.94 with 10 vehicles" width="560">
</p>

<p align="center"><em>Solomon instance C108 solved to distance 828.94 with 10 vehicles — matching the best-known value.</em></p>

## The problem

Given a depot, a fleet of identical vehicles with limited capacity, and a set of customers — each with a demand, a service time, and a `[ready_time, due_date]` time window — find routes that visit every customer exactly once, respect capacity and all time windows (arriving early means waiting; every route must also make it back to the depot before its closing time), and minimize total travelled distance. Instances follow the classic [Solomon benchmark](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/) format; ten of them ship in [`data/instances/`](data/instances/).

## The algorithm

1. **Greedy construction** — routes are built one vehicle at a time; each step picks the feasible candidate minimizing `distance × (ready_time + 1) × due_date`, a score that favors close, early, urgent customers.

2. **Iterated Local Search** — repeat until the time budget runs out or 20 consecutive iterations fail to improve:
   - **Perturbation**: kick the current solution with random feasible inter-route relocations. The kick strength is *adaptive*: it grows with each non-improving iteration (capped at 3× the base) and resets on improvement.
   - **Local search**: a first-improvement cascade over seven operators — inter-route *cross* (tail swap), intra-route *relocate*, *2-opt* (geometrically gated: only tried where route segments actually cross), and *or-opt* (2–3 customer chains, both orientations), then inter-route *exchange*, *relocate*, and *or-opt*. Any accepted move restarts the cascade from the top.
   - **Acceptance**: random walk — the search continues from the perturbed solution even when it is worse, while the best solution found is tracked separately. (Restart-from-best acceptance is available via `ils(restart_from_best=True)`, but lost an A/B test against the random walk.)

Feasibility checking is the hot path, so it is engineered accordingly: routes carry prefix sums of cumulative demand, distance, and arrival times, and candidate moves are validated with `check_route_from`, which resumes from the unchanged prefix's cached state in O(1) and only replays the modified suffix. Accepted moves build a new solution from shallow copies — nothing is ever deep-copied and vehicles are never mutated after insertion.

## Quick start

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install dependencies
uv run pytest -q # run the test suite
```

Solve a single instance:

```python
from cvrptw.benchmark import run_instance

result = run_instance('data/instances/C108.txt', results_dir='results')
print(result.distance, result.n_vehicles, result.improvement_pct)
```

Or drive the pieces yourself:

```python
from cvrptw.io import load_instance, save_solution
from cvrptw.solver import get_greedy_solution, ils, ls_attempts_and_time_limit

inst = load_instance('data/instances/C108.txt')
greedy = get_greedy_solution(inst)

max_attempts, time_limit = ls_attempts_and_time_limit(inst.n_vehicles, len(inst.customers))
n_iters, best, stats = ils(greedy, max_attempts, n_perturbation_moves=5,
                           time_limit=time_limit, verbose=True)

save_solution('results/C108.sol', best)
```

The full benchmark — every instance in `data/instances/`, with progress bars, per-instance `.sol` files and route plots saved to `results/` — runs from the driver notebook:

```bash
uv run jupyter notebook notebooks/ILS.ipynb
```

## Package layout

```
cvrptw/
  model/        Customer, Instance, Route, Vehicle, Solution
  io.py         load_instance, save_solution, distance matrix
  operators/    route transforms (cross/exchange/relocate/or_opt/two_opt),
                feasibility validation, segment-crossing geometry
  search/       local-search cascade, perturbation, attempt/deadline budget
  solver/       greedy construction + the ILS loop
  viz.py        route plots and per-operator search statistics
  benchmark.py  run one instance or the whole directory
tests/          one test file per source submodule
```

## File formats

**Instances** (`data/instances/*.txt`) — Solomon format: line 5 holds `n_vehicles capacity`; each remaining line is `id x y demand ready_time due_date service_time`. Customer 0 is the depot.

**Solutions** (`results/*.sol`) — one route per line as space-separated `customer_id arrival_time` pairs, starting and ending at the depot.
