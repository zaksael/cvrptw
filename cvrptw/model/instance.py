from dataclasses import dataclass

from .customer import Customer


@dataclass(eq=False)
class Instance:
    n_vehicles: int
    capacity: int
    customers: list[Customer]
    # nested lists, not an ndarray: scalar lookups dm[a][b] dominate the hot
    # path and plain-list indexing is several times faster than numpy's
    distances: list[list[float]]

    @property
    def depot(self) -> Customer:
        return self.customers[0]
