import random

from ..model import Solution


def shuffled_vehicle_indices(sol: Solution) -> list[int]:
    indices = list(range(len(sol.vehicles)))
    random.shuffle(indices)
    return indices
