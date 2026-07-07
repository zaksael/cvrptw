import random

from ..model import Solution


def shuffled_vehicle_indices(sol: Solution, rng: random.Random = random) -> list[int]:
    """rng defaults to the global random module, which satisfies the Random
    interface; pass a seeded random.Random for runs isolated from global state."""
    indices = list(range(len(sol.vehicles)))
    rng.shuffle(indices)
    return indices
