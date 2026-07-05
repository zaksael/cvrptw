import numpy as np

from .customer import Customer
from .route import Route


class Vehicle:
    def __init__(self, capacity: int, depot: Customer, distances: np.ndarray) -> None:
        self.initial_capacity = capacity
        self.left_capacity = capacity
        self.dist_matrix = distances
        self._depot = depot
        self.route = Route(
            customers=[depot],
            time_points=[float(depot.ready_time)],
            leg_distances=[0.0],
            demand_used=[0],
            dist_used=[0.0],
        )
        self._departure_time: float = 0.0

    @property
    def depot(self) -> Customer:
        return self._depot

    @property
    def total_time(self) -> float:
        return self._departure_time

    def can_visit(self, c: Customer) -> bool:
        if self.left_capacity < c.demand:
            return False
        travel_time = self.dist_matrix[self.route.customers[-1].cust_id][c.cust_id]
        if self._departure_time + travel_time > c.due_date:
            return False
        depot = self._depot
        time_to_depot = self.dist_matrix[c.cust_id][depot.cust_id]
        return self._departure_time + travel_time + c.service_time + time_to_depot <= depot.due_date

    def try_visit(self, c: Customer) -> bool:
        if self.left_capacity < c.demand:
            return False
        route = self.route
        dm = self.dist_matrix
        d = dm[route.customers[-1].cust_id][c.cust_id]
        t = self._departure_time + d
        if t > c.due_date:
            return False
        depot = self._depot
        if t + c.service_time + dm[c.cust_id][depot.cust_id] > depot.due_date:
            return False
        arrival = t if t >= c.ready_time else float(c.ready_time)
        self.left_capacity -= c.demand
        route.leg_distances.append(d)
        route.time_points.append(arrival)
        route.customers.append(c)
        route.demand_used.append(route.demand_used[-1] + c.demand)
        route.dist_used.append(route._distance + d)
        route._distance += d
        self._departure_time = arrival + c.service_time
        return True

    def visit(self, c: Customer) -> None:
        self.left_capacity -= c.demand
        d = self.dist_matrix[self.route.customers[-1].cust_id][c.cust_id]
        arrival = max(self._departure_time + d, float(c.ready_time))
        self.route.leg_distances.append(d)
        self.route.time_points.append(arrival)
        self.route.customers.append(c)
        self.route.demand_used.append(self.route.demand_used[-1] + c.demand)
        self.route.dist_used.append(self.route._distance + d)
        self.route._distance += d
        self._departure_time = arrival + c.service_time

    def length(self) -> int:
        return self.route.length()

    def distance(self) -> float:
        return self.route.distance
