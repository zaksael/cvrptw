import random

from ..model import Solution
from ..operators import check_route_from, segments_cross, two_opt
from .budget import AttemptBudget


def intra_relocate(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    for v_i in indices:
        v = sol.vehicles[v_i]
        for i in range(1, v.length() - 1):
            for j in range(1, v.length() - 1):
                if i == j:
                    continue
                budget.tick()
                new_route = v.route.customers[:]
                c = new_route[i]
                del new_route[i]
                new_route.insert(j, c)
                valid, new_v = check_route_from(new_route, v, min(i, j) - 1)
                gain = v.distance() - new_v.distance()
                if valid and gain > 1e-3:
                    new_vehicles = sol.vehicles[:]
                    new_vehicles[v_i] = new_v
                    return True, Solution(new_vehicles), gain
    return False, sol, 0.0


def intra_two_opt(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    for v_i in indices:
        v = sol.vehicles[v_i]
        custs = v.route.customers
        n = len(custs)
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if not segments_cross(custs[i], custs[i + 1], custs[j], custs[j + 1]):
                    continue
                budget.tick()
                new_route = two_opt(v, i, j)
                valid, new_v = check_route_from(new_route, v, i)
                gain = v.distance() - new_v.distance()
                if valid and gain > 1e-3:
                    new_vehicles = sol.vehicles[:]
                    new_vehicles[v_i] = new_v
                    return True, Solution(new_vehicles), gain
    return False, sol, 0.0


def intra_or_opt(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    for v_i in indices:
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
                        new_route = v.route.customers[:i] + v.route.customers[i + seg_len:]
                        new_route[j:j] = seg_variant
                        valid, new_v = check_route_from(new_route, v, min(i, j) - 1)
                        gain = v.distance() - new_v.distance()
                        if valid and gain > 1e-3:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[v_i] = new_v
                            return True, Solution(new_vehicles), gain
    return False, sol, 0.0
