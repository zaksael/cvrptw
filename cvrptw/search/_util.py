import random

from ..model import Solution


def shuffled_vehicle_indices(sol: Solution, rng: random.Random = random) -> list[int]:
    """rng defaults to the global random module, which satisfies the Random
    interface; pass a seeded random.Random for runs isolated from global state."""
    indices = list(range(len(sol.vehicles)))
    rng.shuffle(indices)
    return indices


def build_neighbor_sets(distances: list[list[float]], k: int) -> list[set[int]]:
    """k-nearest-neighbor sets per node (depot included), indexed by cust_id.

    Granular-neighborhood support: inter-route operators only evaluate moves
    that create at least one arc ending in a neighbor of the moved customer,
    pruning the O(V²L²) candidate space. k >= n-1 gates nothing out.
    """
    n = len(distances)
    return [
        set(sorted((b for b in range(n) if b != a), key=distances[a].__getitem__)[:k])
        for a in range(n)
    ]
