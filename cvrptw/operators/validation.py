from ..model import Customer, Instance, Route, Solution, Vehicle


def verify_solution(sol: Solution, instance: Instance) -> list[str]:
    """Independent full-rebuild check of a Solution; returns violations (empty = valid).

    Deliberately trusts none of the incremental state the search maintains:
    every route is replayed from scratch through check_route. Meant as a
    guard in end-to-end tests against incremental-state bugs — not a hot path.
    """
    problems = []
    if len(sol.vehicles) > instance.n_vehicles:
        problems.append(f'{len(sol.vehicles)} routes exceed the {instance.n_vehicles}-vehicle limit')
    depot_id = instance.depot.cust_id
    seen: dict[int, int] = {}
    for r_i, v in enumerate(sol.vehicles):
        route = v.route.customers
        if len(route) < 2 or route[0].cust_id != depot_id or route[-1].cust_id != depot_id:
            problems.append(f'route {r_i} does not start and end at the depot')
            continue
        ok, rebuilt = check_route(route, instance.capacity, instance.distances)
        if not ok:
            problems.append(f'route {r_i} is infeasible (capacity or time windows)')
        elif abs(rebuilt.distance() - v.distance()) > 1e-6:
            problems.append(
                f'route {r_i} cached distance {v.distance():.6f} != rebuilt {rebuilt.distance():.6f}')
        for c in route[1:-1]:
            if c.cust_id == depot_id:
                problems.append(f'route {r_i} visits the depot mid-route')
            elif c.cust_id in seen:
                problems.append(f'customer {c.cust_id} visited in routes {seen[c.cust_id]} and {r_i}')
            else:
                seen[c.cust_id] = r_i
    missing = sol.missing_customers(instance)
    if missing:
        problems.append(f'customers missing: {sorted(missing)}')
    return problems


def check_route(route: list[Customer], capacity: int, distances: list[list[float]]) -> tuple[bool, Vehicle]:
    v = Vehicle(capacity, route[0], distances)
    for c in route[1:]:
        if not v.try_visit(c):
            return False, v
    return True, v


def check_route_from(suffix: list[Customer], src: Vehicle, prefix_end: int) -> tuple[bool, Vehicle]:
    """Validate suffix, resuming from src's state at position prefix_end.

    suffix must be exactly what src.route.customers[prefix_end+1:] should
    become. Skips replaying the already-validated prefix, cutting try_visit
    calls proportionally to how deep into the route the modification starts.
    Prefix lists are sliced only on success — infeasible routes incur no copy cost.
    """
    src_route = src.route
    dm = src.dist_matrix
    depot = src._depot

    left_cap = src.initial_capacity - src_route.demand_used[prefix_end]
    dep_time = src_route.time_points[prefix_end] + src_route.customers[prefix_end].service_time
    last_id = src_route.customers[prefix_end].cust_id

    suffix_legs: list[float] = []
    suffix_arrivals: list[float] = []

    for c in suffix:
        if left_cap < c.demand:
            return False, src
        d = dm[last_id][c.cust_id]
        t = dep_time + d
        if t > c.due_date:
            return False, src
        arrival = t if t >= c.ready_time else float(c.ready_time)
        if arrival + c.service_time + dm[c.cust_id][depot.cust_id] > depot.due_date:
            return False, src
        left_cap -= c.demand
        dep_time = arrival + c.service_time
        last_id = c.cust_id
        suffix_legs.append(d)
        suffix_arrivals.append(arrival)

    # Route is valid — slice prefix once and build full lists
    pe = prefix_end + 1
    cum_d = src_route.demand_used[prefix_end]
    cum_dist = src_route.dist_used[prefix_end]
    demand_used = src_route.demand_used[:pe]
    dist_used = src_route.dist_used[:pe]
    for c, d in zip(suffix, suffix_legs):
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
        customers=src_route.customers[:pe] + suffix,
        time_points=src_route.time_points[:pe] + suffix_arrivals,
        leg_distances=src_route.leg_distances[:pe] + suffix_legs,
        demand_used=demand_used,
        dist_used=dist_used,
    )
    v.route._distance = cum_dist
    return True, v
