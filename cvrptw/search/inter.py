import random

from ..model import Customer, Solution, Vehicle
from ..operators import check_route_from, customer_indices
from ._util import shuffled_vehicle_indices
from .budget import AttemptBudget


def apply_or_opt(sol: Solution, budget: AttemptBudget, rng: random.Random = random,
                 neighbors: list[set[int]] | None = None) -> tuple[bool, Solution, float]:
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
                    r2 = v2.route.customers
                    for j in customer_indices(v2, with_last=True):
                        for seg_variant in (seg, seg_reversed):
                            if neighbors is not None and not (
                                    r2[j - 1].cust_id in neighbors[seg_variant[0].cust_id]
                                    or r2[j].cust_id in neighbors[seg_variant[-1].cust_id]):
                                continue
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


def apply_relocate(sol: Solution, budget: AttemptBudget, rng: random.Random = random,
                   neighbors: list[set[int]] | None = None) -> tuple[bool, Solution, float]:
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
            nbr_c = None if neighbors is None else neighbors[c.cust_id]
            for idx2 in indices:
                if idx1 == idx2:
                    continue
                v2 = sol.vehicles[idx2]
                r2 = v2.route.customers
                for j in range(1, v2.length()):
                    if nbr_c is not None and not (
                            r2[j - 1].cust_id in nbr_c or r2[j].cust_id in nbr_c):
                        continue
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


def cross_suffix(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    return v2.route.customers[j:], v1.route.customers[i:]


def cross_gate(v1: Vehicle, i: int, v2: Vehicle, j: int, neighbors: list[set[int]]) -> bool:
    """True when a created arc — (r1[i-1], r2[j]) or (r2[j-1], r1[i]) — is short."""
    r1, r2 = v1.route.customers, v2.route.customers
    return (r2[j].cust_id in neighbors[r1[i - 1].cust_id]
            or r1[i].cust_id in neighbors[r2[j - 1].cust_id])


def exchange_suffix(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    return ([v2.route.customers[j]] + v1.route.customers[i + 1:],
            [v1.route.customers[i]] + v2.route.customers[j + 1:])


def exchange_gate(v1: Vehicle, i: int, v2: Vehicle, j: int, neighbors: list[set[int]]) -> bool:
    """True when a created arc around either swapped customer is short."""
    r1, r2 = v1.route.customers, v2.route.customers
    c1, c2 = r1[i].cust_id, r2[j].cust_id
    return (c2 in neighbors[r1[i - 1].cust_id] or c2 in neighbors[r1[i + 1].cust_id]
            or c1 in neighbors[r2[j - 1].cust_id] or c1 in neighbors[r2[j + 1].cust_id])


def apply_operator(sol: Solution, operator, with_last: bool, budget: AttemptBudget,
                   gate=None, rng: random.Random = random,
                   neighbors: list[set[int]] | None = None) -> tuple[bool, Solution, float]:
    indices = shuffled_vehicle_indices(sol, rng)
    for idx1 in indices:
        v1 = sol.vehicles[idx1]
        for idx2 in indices:
            if idx1 == idx2:
                continue
            v2 = sol.vehicles[idx2]
            for i in customer_indices(v1, with_last):
                for j in customer_indices(v2, with_last):
                    if neighbors is not None and not gate(v1, i, v2, j, neighbors):
                        continue
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
