from dataclasses import dataclass

from .instance import Instance
from .vehicle import Vehicle


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

    def missing_customers(self, instance: Instance) -> set[int]:
        visited = {c.cust_id for v in self.vehicles for c in v.route.customers}
        return {c.cust_id for c in instance.customers[1:]} - visited

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
