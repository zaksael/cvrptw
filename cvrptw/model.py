from dataclasses import dataclass


@dataclass(frozen=True, repr=False)
class Customer:
    cust_id: int
    x: int
    y: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int

    def __repr__(self):
        return (f"Customer: <{self.cust_id:3}, {self.x:2}, {self.y:2}, {self.demand:2}, "
                f"{self.ready_time:3}, {self.due_date:4}, {self.service_time:2}>")


class Vehicle:
    def __init__(self, capacity, depot, distances):
        self.depot = depot
        self.route = [depot]
        self.time_points = [depot.ready_time]
        self.initial_capacity = capacity
        self.left_capacity = capacity
        self.d = distances
        self.total_time = 0
        self.distances = [0]
        self._distance = 0.0

    def can_visit(self, c):
        if self.left_capacity < c.demand:
            return False
        travel_time = self.d[self.route[-1].cust_id][c.cust_id]
        if self.total_time + travel_time >= c.due_date:
            return False
        time_to_depot = self.d[c.cust_id][self.depot.cust_id]
        return self.total_time + travel_time + c.service_time + time_to_depot < self.depot.due_date

    def visit(self, c):
        self.left_capacity -= c.demand
        distance_to_customer = self.d[self.route[-1].cust_id][c.cust_id]
        self.distances.append(distance_to_customer)
        self._distance += distance_to_customer
        self.total_time += distance_to_customer
        if self.total_time < c.ready_time:
            self.total_time = c.ready_time
        self.time_points.append(self.total_time)
        self.total_time += c.service_time
        self.route.append(c)

    def length(self):
        return len(self.route)

    def distance(self):
        return self._distance

    def print_info(self):
        for c, time, dist in zip(self.route, self.time_points, self.distances):
            print(f"{c.cust_id:3} demand={c.demand:2} dist={dist:6.3f} "
                  f"time={time:7.2f} tw=({c.ready_time:3},{c.due_date:4}) "
                  f"service_time={c.service_time:2} ->")
        print(f"Final time = {self.total_time:5.3f}, distance = {self.distance():6.2f}, "
              f"length={self.length():2}, left_capacity = {self.left_capacity}")

    def __repr__(self):
        return str([c.cust_id for c in self.route])


def get_time(solution):
    return sum(v.total_time for v in solution)


def get_distance(solution):
    return sum(v.distance() for v in solution)


def remove_empty_routes(sol):
    return [v for v in sol if v.length() > 2]


def print_solution_info(solution, verbose=False):
    print(f"Total solution time = {get_time(solution):.2f}, "
          f"distance = {get_distance(solution):.2f}, vehicles = {len(solution)}:")
    for i, v in enumerate(sorted(solution, key=lambda v: v.distance())):
        if verbose:
            print('-' * 75)
            v.print_info()
        else:
            print(f"{i+1:2}) Time={v.total_time:7.2f}, distance={v.distance():6.2f}, "
                  f"length={v.length():2}, left capacity={v.left_capacity:3}: {v}")
