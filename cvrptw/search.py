import copy
import random

from .model import remove_empty_routes
from .operators import check_route, cross, exchange, relocate, rng


def local_search(sol, max_attempts=200_000):
    class _LimitReached(Exception):
        pass

    n_attempts = 0

    def _count():
        nonlocal n_attempts
        n_attempts += 1
        if n_attempts == max_attempts:
            raise _LimitReached

    def intra_relocate(sol):
        result = copy.deepcopy(sol)
        random.shuffle(result)
        for v_i, v in enumerate(result):
            best_v = v
            for i in range(1, v.length() - 1):
                for j in range(1, v.length() - 1):
                    if i == j:
                        continue
                    _count()
                    new_route = copy.deepcopy(v.route)
                    c = new_route[i]
                    del new_route[i]
                    new_route.insert(j, c)
                    valid, new_v = check_route(new_route, v.initial_capacity, v.d)
                    if valid and best_v.distance() - new_v.distance() > 1e-3:
                        result[v_i] = new_v
                        return True, result
        return False, result

    def apply_operator(sol, operator, with_last):
        result = copy.deepcopy(sol)
        random.shuffle(result)
        for idx1, v1 in enumerate(result):
            for idx2, v2 in enumerate(result):
                if idx1 == idx2:
                    continue
                for i in rng(v1, with_last):
                    for j in rng(v2, with_last):
                        _count()
                        r1, r2 = operator(v1, i, v2, j)
                        ok1, nv1 = check_route(r1, v1.initial_capacity, v1.d)
                        ok2, nv2 = check_route(r2, v2.initial_capacity, v2.d)
                        if ok1 and ok2:
                            gain = v1.distance() + v2.distance() - nv1.distance() - nv2.distance()
                            if gain > 1e-3:
                                result[idx1] = nv1
                                result[idx2] = nv2
                                return True, remove_empty_routes(result)
        return False, result

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


def perturbation(solution, n_moves=5):
    moved_ids: set = set()

    def inter_relocate(sol):
        result = copy.deepcopy(sol)
        random.shuffle(result)
        for v1_idx, v1 in enumerate(result):
            for v2_idx, v2 in enumerate(result):
                if v1_idx == v2_idx:
                    continue
                for i in range(1, v1.length() - 1):
                    for j in range(1, v2.length() - 1):
                        r1, r2 = relocate(v1, i, v2, j)
                        ok1, nv1 = check_route(r1, v1.initial_capacity, v1.d)
                        ok2, nv2 = check_route(r2, v2.initial_capacity, v2.d)
                        if ok1 and ok2:
                            c_id = v1.route[i].cust_id
                            if c_id in moved_ids:
                                continue
                            result[v1_idx] = nv1
                            result[v2_idx] = nv2
                            moved_ids.add(c_id)
                            return True, remove_empty_routes(result)
        return False, result

    result = solution
    changes_made = False
    for _ in range(n_moves):
        done, result = inter_relocate(result)
        if not done:
            break
        changes_made = True
    return changes_made, result
