import random

from cvrptw.model import Solution, Vehicle
from cvrptw.search._util import build_neighbor_sets, shuffled_vehicle_indices


def test_shuffled_vehicle_indices_explicit_rng(tiny):
    """A seeded rng gives a reproducible permutation of all vehicle indices."""
    customers, distances, capacity = tiny
    sol = Solution([Vehicle(capacity, customers[0], distances) for _ in range(5)])

    a = shuffled_vehicle_indices(sol, random.Random(3))
    b = shuffled_vehicle_indices(sol, random.Random(3))

    assert a == b
    assert sorted(a) == list(range(5))


def test_build_neighbor_sets_k_nearest_excluding_self(tiny):
    """Collinear tiny layout (x = 0, 10, 20, 30): each node's k=2 set is its
    two nearest by Euclidean distance, never itself; depot is included as a
    node like any other."""
    _, distances, _ = tiny
    nbrs = build_neighbor_sets(distances, k=2)
    assert nbrs[0] == {1, 2}
    assert nbrs[1] == {0, 2}
    assert nbrs[2] == {1, 3}
    assert nbrs[3] == {2, 1}


def test_build_neighbor_sets_large_k_includes_everyone(tiny):
    _, distances, _ = tiny
    nbrs = build_neighbor_sets(distances, k=99)
    assert all(s == set(range(4)) - {a} for a, s in enumerate(nbrs))
