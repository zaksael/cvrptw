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
