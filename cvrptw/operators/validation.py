import numpy as np

from ..model import Customer, Route, Vehicle


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
    Prefix lists are sliced only on success — infeasible routes incur no copy cost.
    """
    src_route = src.route
    dm = src.dist_matrix
    depot = src._depot
    pe = prefix_end + 1

    left_cap = src.initial_capacity - src_route.demand_used[prefix_end]
    dep_time = src_route.time_points[prefix_end] + src_route.customers[prefix_end].service_time
    last_id = src_route.customers[prefix_end].cust_id

    suffix_legs: list[float] = []
    suffix_arrivals: list[float] = []

    for i in range(pe, len(route)):
        c = route[i]
        if left_cap < c.demand:
            return False, src
        d = dm[last_id][c.cust_id]
        t = dep_time + d
        if t > c.due_date or t + c.service_time + dm[c.cust_id][depot.cust_id] > depot.due_date:
            return False, src
        arrival = t if t >= c.ready_time else float(c.ready_time)
        left_cap -= c.demand
        dep_time = arrival + c.service_time
        last_id = c.cust_id
        suffix_legs.append(d)
        suffix_arrivals.append(arrival)

    # Route is valid — slice prefix once and build full lists
    suffix_customers = route[pe:]
    cum_d = src_route.demand_used[prefix_end]
    cum_dist = src_route.dist_used[prefix_end]
    demand_used = src_route.demand_used[:pe]
    dist_used = src_route.dist_used[:pe]
    for c, d in zip(suffix_customers, suffix_legs):
        cum_d += c.demand
        cum_dist += d
        demand_used.append(cum_d)
        dist_used.append(cum_dist)

    v = object.__new__(Vehicle)
    v.initial_capacity = src.initial_capacity
    v.dist_matrix = dm
    v._depot = depot
    v.left_capacity = left_cap
    v._departure_time = dep_time
    v.route = Route(
        customers=src_route.customers[:pe] + suffix_customers,
        time_points=src_route.time_points[:pe] + suffix_arrivals,
        leg_distances=src_route.leg_distances[:pe] + suffix_legs,
        demand_used=demand_used,
        dist_used=dist_used,
    )
    v.route._distance = cum_dist
    return True, v
