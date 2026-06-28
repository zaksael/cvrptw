import numpy as np

from .model import Customer, Route, Vehicle


def customer_indices(v: Vehicle, with_last: bool) -> range:
    return range(1, v.length()) if with_last else range(1, v.length() - 1)


def check_route(route: list[Customer], capacity: int, distances: np.ndarray) -> tuple[bool, Vehicle]:
    v = Vehicle(capacity, route[0], distances)
    for c in route[1:]:
        if not v.try_visit(c):
            return False, v
    return True, v


def check_route_from(route: list[Customer], src: Vehicle, prefix_end: int) -> tuple[bool, Vehicle]:
    """Validate route[prefix_end+1:], resuming from src's state at position prefix_end.

    route[:prefix_end+1] must equal src.route.customers[:prefix_end+1].
    Skips replaying the already-validated prefix, cutting try_visit calls
    proportionally to how deep into the route the modification starts.
    """
    src_route = src.route
    src_customers = src_route.customers
    src_tp = src_route.time_points
    leg_prefix = src_route.leg_distances[:prefix_end + 1]

    v = object.__new__(Vehicle)
    v.initial_capacity = src.initial_capacity
    v.dist_matrix = src.dist_matrix
    v._depot = src._depot
    v.left_capacity = src.initial_capacity - src_route.demand_used[prefix_end]
    v._departure_time = src_tp[prefix_end] + src_customers[prefix_end].service_time
    v.route = Route(
        customers=src_customers[:prefix_end + 1],
        time_points=src_tp[:prefix_end + 1],
        leg_distances=leg_prefix,
        demand_used=src_route.demand_used[:prefix_end + 1],
    )
    v.route._distance = sum(leg_prefix)

    for c in route[prefix_end + 1:]:
        if not v.try_visit(c):
            return False, v
    return True, v


def cross(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route.customers, v2.route.customers
    return r1[:i] + r2[j:], r2[:j] + r1[i:]


def exchange(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route.customers[:], v2.route.customers[:]
    r1[i], r2[j] = r2[j], r1[i]
    return r1, r2


def relocate(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route.customers[:], v2.route.customers[:]
    c = r1[i]
    del r1[i]
    r2.insert(j, c)
    return r1, r2
