import random

from .model import Solution
from .operators import check_route_from, cross, customer_indices, exchange


def local_search(sol: Solution, max_attempts: int = 200_000) -> tuple[bool, Solution]:
    class _LimitReached(Exception):
        pass

    n_attempts = 0

    def _count():
        nonlocal n_attempts
        n_attempts += 1
        if n_attempts == max_attempts:
            raise _LimitReached

    def intra_relocate(sol: Solution) -> tuple[bool, Solution]:
        indices = list(range(len(sol.vehicles)))
        random.shuffle(indices)
        for v_i in indices:
            v = sol.vehicles[v_i]
            for i in range(1, v.length() - 1):
                for j in range(1, v.length() - 1):
                    if i == j:
                        continue
                    _count()
                    new_route = v.route.customers[:]
                    c = new_route[i]
                    del new_route[i]
                    new_route.insert(j, c)
                    valid, new_v = check_route_from(new_route, v, min(i, j) - 1)
                    if valid and v.distance() - new_v.distance() > 1e-3:
                        new_vehicles = sol.vehicles[:]
                        new_vehicles[v_i] = new_v
                        return True, Solution(new_vehicles)
        return False, sol

    def apply_operator(sol: Solution, operator, with_last: bool) -> tuple[bool, Solution]:
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
                        _count()
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
                                return True, Solution(new_vehicles).without_empty_routes()
        return False, sol

    result = sol
    changes_made = False
    can_move = True
    while can_move:
        try:
            done, result = apply_operator(result, cross, with_last=True)
            if done:
                changes_made = True
            else:
                done, result = intra_relocate(result)
                if done:
                    changes_made = True
                else:
                    done, result = apply_operator(result, exchange, with_last=False)
                    if done:
                        changes_made = True
                    else:
                        can_move = False
        except _LimitReached:
            break
    return changes_made, result


def perturbation(solution: Solution, n_moves: int = 5) -> tuple[bool, Solution]:
    moved_ids: set[int] = set()

    def inter_relocate(sol: Solution) -> tuple[bool, Solution]:
        indices = list(range(len(sol.vehicles)))
        random.shuffle(indices)
        for v1_idx in indices:
            v1 = sol.vehicles[v1_idx]
            for v2_idx in indices:
                if v1_idx == v2_idx:
                    continue
                v2 = sol.vehicles[v2_idx]
                for i in range(1, v1.length() - 1):
                    c_id = v1.route.customers[i].cust_id
                    if c_id in moved_ids:
                        continue
                    r1 = v1.route.customers[:i] + v1.route.customers[i + 1:]
                    ok1, nv1 = check_route_from(r1, v1, i - 1)
                    if not ok1:
                        continue
                    c = v1.route.customers[i]
                    for j in range(1, v2.length()):
                        r2 = v2.route.customers[:j] + [c] + v2.route.customers[j:]
                        ok2, nv2 = check_route_from(r2, v2, j - 1)
                        if ok2:
                            new_vehicles = sol.vehicles[:]
                            new_vehicles[v1_idx] = nv1
                            new_vehicles[v2_idx] = nv2
                            moved_ids.add(c_id)
                            return True, Solution(new_vehicles).without_empty_routes()
        return False, sol

    result = solution
    changes_made = False
    for _ in range(n_moves):
        done, result = inter_relocate(result)
        if not done:
            break
        changes_made = True
    return changes_made, result
