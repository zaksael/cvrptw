from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, repr=False)
class Customer:
    cust_id: int
    x: int
    y: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int

    def __repr__(self) -> str:
        return (f"Customer: <{self.cust_id:3}, {self.x:2}, {self.y:2}, {self.demand:2}, "
                f"{self.ready_time:3}, {self.due_date:4}, {self.service_time:2}>")


@dataclass(eq=False)
class Instance:
    n_vehicles: int
    capacity: int
    customers: list[Customer]
    distances: np.ndarray

    @property
    def depot(self) -> Customer:
        return self.customers[0]

    def __repr__(self) -> str:
        return (f"Instance(n_vehicles={self.n_vehicles}, capacity={self.capacity}, "
                f"customers={len(self.customers)}, distances={self.distances.shape})")


@dataclass(eq=False)
class Route:
    customers: list[Customer]
    time_points: list[float]
    leg_distances: list[float]
    demand_used: list[int] = field(default_factory=lambda: [0])
    _distance: float = field(default=0.0, init=False, repr=False)

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def total_time(self) -> float:
        return self.time_points[-1]

    def length(self) -> int:
        return len(self.customers)

    @property
    def depot(self) -> Customer:
        return self.customers[0]

    def __repr__(self) -> str:
        return str([c.cust_id for c in self.customers])


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
        self.route._distance += d
        self._departure_time = arrival + c.service_time

    def length(self) -> int:
        return self.route.length()

    def distance(self) -> float:
        return self.route.distance

    def print_info(self) -> None:
        for c, time, dist in zip(self.route.customers, self.route.time_points, self.route.leg_distances):
            print(f"{c.cust_id:3} demand={c.demand:2} dist={dist:6.3f} "
                  f"time={time:7.2f} tw=({c.ready_time:3},{c.due_date:4}) "
                  f"service_time={c.service_time:2} ->")
        print(f"Final time = {self.total_time:5.3f}, distance = {self.distance():6.2f}, "
              f"length={self.length():2}, left_capacity = {self.left_capacity}")

    def __repr__(self) -> str:
        return str([c.cust_id for c in self.route.customers])


@dataclass(eq=False)
class Solution:
    vehicles: list[Vehicle]

    @property
    def distance(self) -> float:
        return sum(v.distance() for v in self.vehicles)

    @property
    def time(self) -> float:
        return sum(v.total_time for v in self.vehicles)

    def without_empty_routes(self) -> Solution:
        return Solution([v for v in self.vehicles if v.length() > 2])

    def __len__(self) -> int:
        return len(self.vehicles)

    def __iter__(self):
        return iter(self.vehicles)

    def __repr__(self) -> str:
        return f"Solution(vehicles={len(self)}, distance={self.distance:.2f})"

    def print_info(self, verbose: bool = False) -> None:
        print(f"Total time = {self.time:.2f}, distance = {self.distance:.2f}, vehicles = {len(self)}:")
        for i, v in enumerate(sorted(self.vehicles, key=lambda v: v.distance())):
            if verbose:
                print('-' * 75)
                v.print_info()
            else:
                print(f"{i+1:2}) Time={v.total_time:7.2f}, distance={v.distance():6.2f}, "
                      f"length={v.length():2}, left capacity={v.left_capacity:3}: {v}")
