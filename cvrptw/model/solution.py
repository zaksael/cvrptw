from dataclasses import dataclass

from .instance import Instance
from .vehicle import Vehicle


@dataclass(eq=False)
class Solution:
    vehicles: list[Vehicle]

    @property
    def distance(self) -> float:
        return sum(v.distance() for v in self.vehicles)

    def without_empty_routes(self) -> Solution:
        return Solution([v for v in self.vehicles if v.length() > 2])

    def missing_customers(self, instance: Instance) -> set[int]:
        visited = {c.cust_id for v in self.vehicles for c in v.route.customers}
        return {c.cust_id for c in instance.customers[1:]} - visited

    def __len__(self) -> int:
        return len(self.vehicles)

    def __iter__(self):
        return iter(self.vehicles)

    def __repr__(self) -> str:
        return f"Solution(vehicles={len(self)}, distance={self.distance:.2f})"
