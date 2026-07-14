# CVRPTW · Iterated Local Search

A heuristic solver for the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**, written in pure Python (NumPy is used only to compute the distance matrix). Solutions are constructed greedily and then improved with **Iterated Local Search (ILS)** over a cascade of seven route operators.

<p align="center">
  <img src="docs/c108_solution.png" alt="Solved Solomon C108 instance: distance 828.94 with 10 vehicles" width="560">
</p>

<p align="center"><em>Solomon instance C108 solved to distance 828.94 with 10 vehicles — matching the best-known value.</em></p>

## The problem

Given a depot, a fleet of identical vehicles with limited capacity, and a set of customers — each with a demand, a service time, and a `[ready_time, due_date]` time window — find routes that visit every customer exactly once, respect capacity and all time windows (arriving early means waiting; every route must also make it back to the depot before its closing time), and optimize the hierarchical Solomon objective: fewest vehicles first, then minimum total travelled distance (`ils(minimize_vehicles=False)` switches to distance-only). Instances follow the classic [Solomon benchmark](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/) format; the full 56-instance 100-customer set ships in [`data/instances/solomon/`](data/instances/solomon/).

## The algorithm

1. **Greedy construction** — routes are built one vehicle at a time; each step picks the feasible candidate minimizing `distance × (ready_time + 1) × due_date`, a score that favors close, early, urgent customers.

2. **Iterated Local Search** — repeat until the time budget runs out or `max_failed_iters` (default 20) consecutive iterations fail to improve:
   - **Perturbation**: kick the current solution with random feasible inter-route relocations. The kick strength is *adaptive*: it grows with each non-improving iteration (capped at 3× the base) and resets on improvement.
   - **Route elimination**: try to empty the smallest route by relocating all of its customers into the others (all-or-nothing, feasibility-only) — the move that actually shrinks the fleet; the local search that follows repairs the distance damage.
   - **Local search**: a first-improvement cascade over seven operators — inter-route *cross* (tail swap), intra-route *relocate*, *2-opt* (geometrically gated: only tried where route segments actually cross), and *or-opt* (2–3 customer chains, both orientations), then inter-route *exchange*, *relocate*, and *or-opt*. Any accepted move restarts the cascade from the top. `ils(n_neighbors=k)` optionally switches the inter-route operators to a *granular neighborhood*: only moves creating an arc to one of the moved customer's k nearest nodes are evaluated — a large win on distance-dominated instances (it has hit the exact best-known 824.78 on C104 in a 60 s run), off by default because exhaustive scanning keeps a small fleet-size edge on vehicle-contested instances.
   - **Acceptance**: random walk — the search continues from the perturbed solution even when it is worse, while the best solution found is tracked separately, compared lexicographically on (vehicles, distance). (Restart-from-best acceptance is available via `ils(restart_from_best=True)`, but lost an A/B test against the random walk.)

Feasibility checking is the hot path, so it is engineered accordingly: routes carry prefix sums of cumulative demand, distance, and arrival times, and candidate moves are validated with `check_route_from`, which resumes from the unchanged prefix's cached state in O(1) and only replays the modified suffix. The distance matrix is stored as plain nested lists — scalar lookups dominate, and list indexing beats NumPy's per-access overhead by ~1.7× in local-search throughput. Accepted moves build a new solution from shallow copies — nothing is ever deep-copied and vehicles are never mutated after insertion.

Runs are reproducible: every randomized component threads a single `rng`, so `ils(..., rng=random.Random(seed))` gives a deterministic trajectory isolated from global random state. That covers even runs cut off by the time limit — the seed determines everything except where the clock stopped the run, and `stop_after=stop_after_from_stats(stats)` replays a finished run's exact stopping point (down to the local-search candidate the deadline interrupted) without a clock. An independent full-rebuild checker, `verify_solution(sol, instance)`, re-validates a solution from scratch and returns any violations — the end-to-end tests assert it on every solver result.

## Quick start

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install dependencies
uv run pytest -q # run the test suite
```

Solve a single instance:

```python
from cvrptw.benchmark import run_instance

result = run_instance('data/instances/solomon/c108.txt', results_dir='data/solutions/solomon')
print(result.distance, result.n_vehicles, result.improvement_pct)
```

Or drive the pieces yourself (the headline API is re-exported at the top level):

```python
import random

from cvrptw import load_instance, save_solution, get_greedy_solution, ils, ls_attempts_and_time_limit

inst = load_instance('data/instances/solomon/c108.txt')
greedy = get_greedy_solution(inst)

max_attempts, time_limit = ls_attempts_and_time_limit(inst.n_vehicles, len(inst.customers))
n_iters, best, stats = ils(greedy, max_attempts, n_perturbation_moves=5,
                           time_limit=time_limit, verbose=True,
                           rng=random.Random(42))  # optional: reproducible run

save_solution('data/solutions/solomon/c108.sol', best)
```

Compare results against the SINTEF best-known solutions:

```python
from cvrptw import compare_to_bks, format_bks_table

print(format_bks_table(compare_to_bks(results)))  # results from run_benchmark
```

The full benchmark — every instance in `data/instances/solomon/`, with progress bars and per-instance `.sol` files saved to `data/solutions/solomon/` — runs from the driver notebook:

```bash
uv run jupyter notebook notebooks/ILS.ipynb
```

## Package layout

```
cvrptw/
  model/        Customer, Instance, Route, Vehicle, Solution
  io.py         load_instance, save_solution, distance matrix
  operators/    route transforms (cross/exchange/relocate/or_opt/two_opt),
                feasibility validation + verify_solution, crossing geometry
  search/       local-search cascade, perturbation, route elimination,
                attempt/deadline budget, k-nearest-neighbor sets
  solver/       greedy construction + the ILS loop
  viz.py        route plots and per-operator search statistics
  benchmark.py  run one instance or the whole directory
  bks.py        Solomon best-known solutions + comparison table
tests/          one test file per source submodule
```

## File formats

**Instances** (`data/instances/solomon/*.txt`) — Solomon format: line 5 holds `n_vehicles capacity`; each remaining line is `id x y demand ready_time due_date service_time`. Customer 0 is the depot.

**Solutions** (`data/solutions/solomon/*.sol`) — one route per line as space-separated `customer_id arrival_time` pairs, starting and ending at the depot.

## Results

All 56 Solomon 100-customer instances, 120 s budget per instance with the stagnation cutoff raised to 50 non-improving iterations (`max_failed_iters=50`), seeded run (`rng=random.Random(42)`), default settings (hierarchical objective), 2026-07-14. Total wall time 58.2 min — most runs still converge and stop on the failure streak before the 120 s cap (7/56 hit it). Mean distance gap to the [SINTEF best-known solutions](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/) is **+0.81%**, with the best-known vehicle count matched on **27/56** instances (total fleet 442 vs 405) and 13 instances solved to the exact best-known solution. A negative gap means less distance than the BKS using extra vehicles — not a better solution under the hierarchical objective.

| class | instances | mean gap | fleet (BKS) | at BKS fleet | time |
|-------|--:|--:|--:|--:|--:|
| C1 | 9 | +1.32% | 90 (90) | 9/9 | 4.2 min |
| C2 | 8 | +0.73% | 24 (24) | 8/8 | 7.1 min |
| R1 | 12 | -0.48% | 158 (143) | 0/12 | 10.2 min |
| R2 | 11 | +0.68% | 36 (30) | 5/11 | 17.4 min |
| RC1 | 8 | +0.99% | 105 (92) | 0/8 | 6.0 min |
| RC2 | 8 | +2.25% | 29 (26) | 5/8 | 13.3 min |
| **all** | **56** | **+0.81%** | **442 (405)** | **27/56** | **58.2 min** |

One solved instance per Solomon class from this run — the light-grey routes underneath are the SINTEF best-known solution (fetched with `uv run python scripts/fetch_bks_solutions.py` into [`data/solutions/solomon-bks/`](data/solutions/solomon-bks/)). The `.sol` files live in [`data/solutions/solomon/`](data/solutions/solomon/), and the images are rendered from them with `uv run python scripts/render_readme_images.py`:

<table>
  <tr>
    <td><img src="docs/solutions/c101.png" alt="c101 solution: distance 828.94, 10 vehicles (BKS 828.94, 10 vehicles)" width="400"></td>
    <td><img src="docs/solutions/c201.png" alt="c201 solution: distance 591.56, 3 vehicles (BKS 591.56, 3 vehicles)" width="400"></td>
  </tr>
  <tr>
    <td><img src="docs/solutions/r101.png" alt="r101 solution: distance 1644.82, 20 vehicles (BKS 1650.80, 19 vehicles)" width="400"></td>
    <td><img src="docs/solutions/r201.png" alt="r201 solution: distance 1263.38, 4 vehicles (BKS 1252.37, 4 vehicles)" width="400"></td>
  </tr>
  <tr>
    <td><img src="docs/solutions/rc101.png" alt="rc101 solution: distance 1670.12, 16 vehicles (BKS 1696.95, 14 vehicles)" width="400"></td>
    <td><img src="docs/solutions/rc201.png" alt="rc201 solution: distance 1462.75, 4 vehicles (BKS 1406.94, 4 vehicles)" width="400"></td>
  </tr>
</table>

<details>
<summary>Per-instance results</summary>

**C1**

| instance | distance | BKS distance | gap | vehicles | BKS vehicles | time (s) |
|----------|--:|--:|--:|--:|--:|--:|
| c101 | 828.94 | 828.94 | +0.00% | 10 | 10 | 9.8 |
| c102 | 828.94 | 828.94 | +0.00% | 10 | 10 | 35.9 |
| c103 | 878.36 | 828.06 | +6.07% | 10 | 10 | 48.8 |
| c104 | 872.78 | 824.78 | +5.82% | 10 | 10 | 46.5 |
| c105 | 828.94 | 828.94 | +0.00% | 10 | 10 | 11.7 |
| c106 | 828.94 | 828.94 | +0.00% | 10 | 10 | 17.5 |
| c107 | 828.94 | 828.94 | +0.00% | 10 | 10 | 18.5 |
| c108 | 828.94 | 828.94 | +0.00% | 10 | 10 | 29.9 |
| c109 | 828.94 | 828.94 | +0.00% | 10 | 10 | 34.2 |

**C2**

| instance | distance | BKS distance | gap | vehicles | BKS vehicles | time (s) |
|----------|--:|--:|--:|--:|--:|--:|
| c201 | 591.56 | 591.56 | +0.00% | 3 | 3 | 9.8 |
| c202 | 591.56 | 591.56 | +0.00% | 3 | 3 | 68.5 |
| c203 | 600.21 | 591.17 | +1.53% | 3 | 3 | 73.0 |
| c204 | 616.19 | 590.60 | +4.33% | 3 | 3 | 120.0 |
| c205 | 588.88 | 588.88 | +0.00% | 3 | 3 | 24.8 |
| c206 | 588.49 | 588.49 | +0.00% | 3 | 3 | 35.5 |
| c207 | 588.29 | 588.29 | +0.00% | 3 | 3 | 46.4 |
| c208 | 588.32 | 588.32 | +0.00% | 3 | 3 | 48.5 |

**R1**

| instance | distance | BKS distance | gap | vehicles | BKS vehicles | time (s) |
|----------|--:|--:|--:|--:|--:|--:|
| r101 | 1644.82 | 1650.80 | -0.36% | 20 | 19 | 31.8 |
| r102 | 1473.65 | 1486.12 | -0.84% | 18 | 17 | 40.8 |
| r103 | 1226.31 | 1292.68 | -5.13% | 14 | 13 | 35.8 |
| r104 | 1044.09 | 1007.31 | +3.65% | 11 | 9 | 44.4 |
| r105 | 1381.10 | 1377.11 | +0.29% | 15 | 14 | 37.5 |
| r106 | 1271.81 | 1252.03 | +1.58% | 13 | 12 | 33.9 |
| r107 | 1088.83 | 1104.66 | -1.43% | 11 | 10 | 51.1 |
| r108 | 973.19 | 960.88 | +1.28% | 10 | 9 | 69.8 |
| r109 | 1169.18 | 1194.73 | -2.14% | 13 | 11 | 47.4 |
| r110 | 1110.30 | 1118.84 | -0.76% | 12 | 10 | 47.5 |
| r111 | 1077.53 | 1096.73 | -1.75% | 11 | 10 | 105.5 |
| r112 | 980.48 | 982.14 | -0.17% | 10 | 9 | 64.9 |

**R2**

| instance | distance | BKS distance | gap | vehicles | BKS vehicles | time (s) |
|----------|--:|--:|--:|--:|--:|--:|
| r201 | 1263.38 | 1252.37 | +0.88% | 4 | 4 | 68.7 |
| r202 | 1119.04 | 1191.70 | -6.10% | 4 | 3 | 57.4 |
| r203 | 1009.34 | 939.50 | +7.43% | 3 | 3 | 120.0 |
| r204 | 803.12 | 825.52 | -2.71% | 3 | 2 | 76.2 |
| r205 | 1015.88 | 994.43 | +2.16% | 4 | 3 | 120.0 |
| r206 | 925.98 | 906.14 | +2.19% | 3 | 3 | 74.0 |
| r207 | 887.64 | 890.61 | -0.33% | 3 | 2 | 88.9 |
| r208 | 757.75 | 726.82 | +4.26% | 3 | 2 | 120.0 |
| r209 | 927.85 | 909.16 | +2.06% | 3 | 3 | 120.0 |
| r210 | 985.69 | 939.37 | +4.93% | 3 | 3 | 105.8 |
| r211 | 821.11 | 885.71 | -7.29% | 3 | 2 | 92.1 |

**RC1**

| instance | distance | BKS distance | gap | vehicles | BKS vehicles | time (s) |
|----------|--:|--:|--:|--:|--:|--:|
| rc101 | 1670.12 | 1696.95 | -1.58% | 16 | 14 | 25.7 |
| rc102 | 1520.20 | 1554.75 | -2.22% | 14 | 12 | 32.4 |
| rc103 | 1368.35 | 1261.67 | +8.46% | 13 | 11 | 38.0 |
| rc104 | 1165.15 | 1135.48 | +2.61% | 11 | 10 | 66.1 |
| rc105 | 1578.39 | 1629.44 | -3.13% | 15 | 13 | 38.4 |
| rc106 | 1397.48 | 1424.73 | -1.91% | 13 | 11 | 51.3 |
| rc107 | 1289.07 | 1230.48 | +4.76% | 12 | 11 | 35.6 |
| rc108 | 1150.11 | 1139.82 | +0.90% | 11 | 10 | 69.8 |

**RC2**

| instance | distance | BKS distance | gap | vehicles | BKS vehicles | time (s) |
|----------|--:|--:|--:|--:|--:|--:|
| rc201 | 1462.75 | 1406.94 | +3.97% | 4 | 4 | 70.4 |
| rc202 | 1200.77 | 1365.65 | -12.07% | 4 | 3 | 80.8 |
| rc203 | 1163.68 | 1049.62 | +10.87% | 3 | 3 | 120.0 |
| rc204 | 902.39 | 798.46 | +13.02% | 3 | 3 | 98.5 |
| rc205 | 1249.66 | 1297.65 | -3.70% | 5 | 4 | 92.8 |
| rc206 | 1117.35 | 1146.32 | -2.53% | 4 | 3 | 100.1 |
| rc207 | 1101.24 | 1061.14 | +3.78% | 3 | 3 | 117.5 |
| rc208 | 866.98 | 828.14 | +4.69% | 3 | 3 | 120.0 |

</details>
