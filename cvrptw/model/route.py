from dataclasses import dataclass, field

from .customer import Customer


@dataclass(eq=False)
class Route:
    customers: list[Customer]
    time_points: list[float]
    leg_distances: list[float]
    demand_used: list[int] = field(default_factory=lambda: [0])
    dist_used: list[float] = field(default_factory=lambda: [0.0])
    _distance: float = field(default=0.0, init=False, repr=False)

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def total_time(self) -> float:
        return self.time_points[-1]

    def length(self) -> int:
        return len(self.customers)

    def __repr__(self) -> str:
        return str([c.cust_id for c in self.customers])
