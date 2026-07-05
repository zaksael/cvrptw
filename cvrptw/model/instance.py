from dataclasses import dataclass

import numpy as np

from .customer import Customer


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
