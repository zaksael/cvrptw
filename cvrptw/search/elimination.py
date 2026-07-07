import random

from ..model import Solution
from ..operators import check_route_from
from ._util import shuffled_vehicle_indices


def try_eliminate_route(sol: Solution, rng: random.Random = random) -> tuple[bool, Solution]:
    """All-or-nothing route elimination for the hierarchical objective.

    Tries source routes smallest-first; every customer of the source must be
    relocated into some other route (first feasible insertion, shuffled
    targets) or the source is skipped. Feasibility-only — no gain gating; the
    local-search pass that follows repairs the distance damage. Returns the
    solution with one route fewer on success, the original untouched on
    failure.
    """
    sources = sorted(range(len(sol.vehicles)), key=lambda i: sol.vehicles[i].length())
    for src_idx in sources:
        vehicles = sol.vehicles[:]
        placed_all = True
        for c in sol.vehicles[src_idx].route.customers[1:-1]:
            placed = False
            for t_idx in shuffled_vehicle_indices(sol, rng):
                if t_idx == src_idx:
                    continue
                v2 = vehicles[t_idx]
                for j in range(1, v2.length()):
                    ok, nv2 = check_route_from([c] + v2.route.customers[j:], v2, j - 1)
                    if ok:
                        vehicles[t_idx] = nv2
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                placed_all = False
                break
        if placed_all:
            del vehicles[src_idx]
            return True, Solution(vehicles)
    return False, sol
