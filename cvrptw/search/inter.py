import random

from ..model import Solution
from ..operators import check_route_from, customer_indices
from ._util import shuffled_vehicle_indices
from .budget import AttemptBudget


def apply_or_opt(sol: Solution, budget: AttemptBudget, rng: random.Random = random) -> tuple[bool, Solution, float]:
    indices = shuffled_vehicle_indices(sol, rng)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        n1 = v1.length()
        for seg_len in (2, 3):
            for i in range(1, n1 - seg_len):
                budget.tick()
                seg = v1.route.customers[i:i + seg_len]
                r1_suffix = v1.route.customers[i + seg_len:]
                ok1, nv1 = check_route_from(r1_suffix, v1, i - 1)
                if not ok1:
                    continue
                seg_reversed = seg[::-1]
                for idx2 in indices:
                    if idx1 == idx2:
                        continue
                    v2 = sol.vehicles[idx2]
                    for j in customer_indices(v2, with_last=True):
                        for seg_variant in (seg, seg_reversed):
                            budget.tick()
                            r2_suffix = seg_variant + v2.route.customers[j:]
                            ok2, nv2 = check_route_from(r2_suffix, v2, j - 1)
                            if ok2:
                                gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                                if gain > 1e-3:
                                    new_vehicles = sol.vehicles[:]
                                    new_vehicles[idx1] = nv1
                                    new_vehicles[idx2] = nv2
                                    return True, Solution(new_vehicles).without_empty_routes(), gain
    return False, sol, 0.0


def apply_relocate(sol: Solution, budget: AttemptBudget, rng: random.Random = random) -> tuple[bool, Solution, float]:
    indices = shuffled_vehicle_indices(sol, rng)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        for i in range(1, v1.length() - 1):
            budget.tick()
            c = v1.route.customers[i]
            r1_suffix = v1.route.customers[i + 1:]
            ok1, nv1 = check_route_from(r1_suffix, v1, i - 1)
            if not ok1:
                continue
            for idx2 in indices:
                if idx1 == idx2:
                    continue
                v2 = sol.vehicles[idx2]
                for j in range(1, v2.length()):
                    budget.tick()
                    r2_suffix = [c] + v2.route.customers[j:]
                    ok2, nv2 = check_route_from(r2_suffix, v2, j - 1)
                    if ok2:
                        gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                        if gain > 1e-3:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[idx1] = nv1
                            new_vehicles[idx2] = nv2
                            return True, Solution(new_vehicles).without_empty_routes(), gain
    return False, sol, 0.0


def cross_suffix(v1, i, v2, j):
    return v2.route.customers[j:], v1.route.customers[i:]


def exchange_suffix(v1, i, v2, j):
    return ([v2.route.customers[j]] + v1.route.customers[i + 1:],
            [v1.route.customers[i]] + v2.route.customers[j + 1:])


def apply_operator(sol: Solution, operator, with_last: bool, budget: AttemptBudget, rng: random.Random = random) -> tuple[bool, Solution, float]:
    indices = shuffled_vehicle_indices(sol, rng)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        for idx2 in indices:
            if idx1 == idx2:
                continue
            v2 = sol.vehicles[idx2]
            for i in customer_indices(v1, with_last):
                for j in customer_indices(v2, with_last):
                    budget.tick()
                    r1_suffix, r2_suffix = operator(v1, i, v2, j)
                    ok1, nv1 = check_route_from(r1_suffix, v1, i - 1)
                    if not ok1:
                        continue
                    ok2, nv2 = check_route_from(r2_suffix, v2, j - 1)
                    if ok2:
                        gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                        if gain > 1e-3:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[idx1] = nv1
                            new_vehicles[idx2] = nv2
                            return True, Solution(new_vehicles).without_empty_routes(), gain
    return False, sol, 0.0
