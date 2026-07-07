from ..model import Solution
from ..operators import check_route_from
from ._util import shuffled_vehicle_indices


def inter_relocate(sol: Solution, moved_ids: set[int]) -> tuple[bool, Solution]:
    indices = shuffled_vehicle_indices(sol)
    for v1_idx in indices:
        v1 = sol.vehicles[v1_idx]
        for i in range(1, v1.length() - 1):
            c = v1.route.customers[i]
            if c.cust_id in moved_ids:
                continue
            r1_suffix = v1.route.customers[i + 1:]
            ok1, nv1 = check_route_from(r1_suffix, v1, i - 1)
            if not ok1:
                continue
            for v2_idx in indices:
                if v1_idx == v2_idx:
                    continue
                v2 = sol.vehicles[v2_idx]
                for j in range(1, v2.length()):
                    r2_suffix = [c] + v2.route.customers[j:]
                    ok2, nv2 = check_route_from(r2_suffix, v2, j - 1)
                    if ok2:
                        new_vehicles = sol.vehicles[:]
                        new_vehicles[v1_idx] = nv1
                        new_vehicles[v2_idx] = nv2
                        moved_ids.add(c.cust_id)
                        return True, Solution(new_vehicles).without_empty_routes()
    return False, sol


def perturbation(solution: Solution, n_moves: int = 5) -> tuple[bool, Solution, int]:
    moved_ids: set[int] = set()
    result = solution
    changes_made = False
    actual_moves = 0
    for _ in range(n_moves):
        done, result = inter_relocate(result, moved_ids)
        if not done:
            break
        changes_made = True
        actual_moves += 1
    return changes_made, result, actual_moves
