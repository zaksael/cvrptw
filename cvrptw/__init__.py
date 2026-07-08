"""Heuristic CVRPTW solver: greedy construction + iterated local search.

Headline API is re-exported here; viz and benchmark are deliberately not
imported (they pull in matplotlib) — use `from cvrptw.benchmark import
run_benchmark` / `from cvrptw.viz import draw_solution` directly.
"""

from .bks import SOLOMON_100_BKS, compare_to_bks, format_bks_table
from .io import calculate_distances, load_instance, save_solution
from .model import Customer, Instance, Route, Solution, Vehicle
from .operators import verify_solution
from .solver import get_greedy_solution, ils, ls_attempts_and_time_limit, summarize_operator_stats

__all__ = [
    "SOLOMON_100_BKS", "compare_to_bks", "format_bks_table",
    "calculate_distances", "load_instance", "save_solution",
    "Customer", "Instance", "Route", "Solution", "Vehicle", "verify_solution",
    "get_greedy_solution", "ils", "ls_attempts_and_time_limit", "summarize_operator_stats",
]
