import random

from ..model import Solution
from ..operators import check_route_from, customer_indices, or_opt, relocate
from .budget import AttemptBudget


def apply_or_opt(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        n1 = v1.length()
        for idx2 in indices:
            if idx1 == idx2:
                continue
            v2 = sol.vehicles[idx2]
            for seg_len in (2, 3):
                for i in range(1, n1 - seg_len):
                    for j in customer_indices(v2, with_last=True):
                        for reverse in (False, True):
                            budget.tick()
                            r1, r2 = or_opt(v1, i, seg_len, v2, j, reverse)
                            ok1, nv1 = check_route_from(r1, v1, i - 1)
                            if not ok1:
                                continue
                            ok2, nv2 = check_route_from(r2, v2, j - 1)
                            if ok2:
                                gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                                if gain > 1e-3:
                                    new_vehicles = sol.vehicles[:]
                                    new_vehicles[idx1] = nv1
                                    new_vehicles[idx2] = nv2
                                    return True, Solution(new_vehicles).without_empty_routes(), gain
    return False, sol, 0.0


def apply_relocate(sol: Solution, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        for idx2 in indices:
            if idx1 == idx2:
                continue
            v2 = sol.vehicles[idx2]
            for i in range(1, v1.length() - 1):
                for j in range(1, v2.length()):
                    budget.tick()
                    r1, r2 = relocate(v1, i, v2, j)
                    ok1, nv1 = check_route_from(r1, v1, i - 1)
                    if not ok1:
                        continue
                    ok2, nv2 = check_route_from(r2, v2, j - 1)
                    if ok2:
                        gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                        if gain > 1e-3:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[idx1] = nv1
                            new_vehicles[idx2] = nv2
                            return True, Solution(new_vehicles).without_empty_routes(), gain
    return False, sol, 0.0


def apply_operator(sol: Solution, operator, with_last: bool, budget: AttemptBudget) -> tuple[bool, Solution, float]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        for idx2 in indices:
            if idx1 == idx2:
                continue
            v2 = sol.vehicles[idx2]
            for i in customer_indices(v1, with_last):
                for j in customer_indices(v2, with_last):
                    budget.tick()
                    r1, r2 = operator(v1, i, v2, j)
                    ok1, nv1 = check_route_from(r1, v1, i - 1)
                    if not ok1:
                        continue
                    ok2, nv2 = check_route_from(r2, v2, j - 1)
                    if ok2:
                        gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                        if gain > 1e-3:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[idx1] = nv1
                            new_vehicles[idx2] = nv2
                            return True, Solution(new_vehicles).without_empty_routes(), gain
    return False, sol, 0.0
