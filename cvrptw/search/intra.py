from ..model import Solution
from ..operators import check_route_from, segments_cross
from ._util import shuffled_vehicle_indices
from .budget import AttemptBudget


def intra_relocate(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    for v_i in shuffled_vehicle_indices(sol):
        v = sol.vehicles[v_i]
        for i in range(1, v.length() - 1):
            for j in range(1, v.length() - 1):
                if i == j:
                    continue
                budget.tick()
                lo = min(i, j)
                suffix = v.route.customers[lo:][:]
                li, lj = i - lo, j - lo
                c = suffix[li]
                del suffix[li]
                suffix.insert(lj, c)
                valid, new_v = check_route_from(suffix, v, lo - 1)
                gain = v.distance() - new_v.distance()
                if valid and gain > 1e-3:
                    new_vehicles = sol.vehicles[:]
                    new_vehicles[v_i] = new_v
                    return True, Solution(new_vehicles), gain
    return False, sol, 0.0


def intra_two_opt(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    for v_i in shuffled_vehicle_indices(sol):
        v = sol.vehicles[v_i]
        custs = v.route.customers
        n = len(custs)
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if not segments_cross(custs[i], custs[i + 1], custs[j], custs[j + 1]):
                    continue
                budget.tick()
                suffix = custs[i + 1:j + 1][::-1] + custs[j + 1:]
                valid, new_v = check_route_from(suffix, v, i)
                gain = v.distance() - new_v.distance()
                if valid and gain > 1e-3:
                    new_vehicles = sol.vehicles[:]
                    new_vehicles[v_i] = new_v
                    return True, Solution(new_vehicles), gain
    return False, sol, 0.0


def intra_or_opt(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    for v_i in shuffled_vehicle_indices(sol):
        v = sol.vehicles[v_i]
        n = v.length()
        for seg_len in (2, 3):
            for i in range(1, n - seg_len):
                seg = v.route.customers[i:i + seg_len]
                for j in range(1, n - seg_len):
                    if j == i:
                        continue
                    for seg_variant in (seg, seg[::-1]):
                        budget.tick()
                        lo = min(i, j)
                        suffix = v.route.customers[lo:][:]
                        li, lj = i - lo, j - lo
                        del suffix[li:li + seg_len]
                        suffix[lj:lj] = seg_variant
                        valid, new_v = check_route_from(suffix, v, lo - 1)
                        gain = v.distance() - new_v.distance()
                        if valid and gain > 1e-3:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[v_i] = new_v
                            return True, Solution(new_vehicles), gain
    return False, sol, 0.0
