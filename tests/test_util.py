import random

from cvrptw.model import Solution, Vehicle
from cvrptw.search._util import shuffled_vehicle_indices


def test_shuffled_vehicle_indices_explicit_rng(tiny):
    """A seeded rng gives a reproducible permutation of all vehicle indices."""
    customers, distances, capacity = tiny
    sol = Solution([Vehicle(capacity, customers[0], distances) for _ in range(5)])

    a = shuffled_vehicle_indices(sol, random.Random(3))
    b = shuffled_vehicle_indices(sol, random.Random(3))

    assert a == b
    assert sorted(a) == list(range(5))
