import numpy as np

from .model import Customer, Vehicle


def customer_indices(v: Vehicle, with_last: bool) -> range:
    return range(1, v.length()) if with_last else range(1, v.length() - 1)


def check_route(route: list[Customer], capacity: int, distances: np.ndarray) -> tuple[bool, Vehicle]:
    v = Vehicle(capacity, route[0], distances)
    for c in route[1:]:
        if v.can_visit(c):
            v.visit(c)
        else:
            return False, v
    return True, v


def cross(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    return v1.route[:i] + v2.route[j:], v2.route[:j] + v1.route[i:]


def exchange(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route[:], v2.route[:]
    r1[i], r2[j] = r2[j], r1[i]
    return r1, r2


def relocate(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route[:], v2.route[:]
    c = r1[i]
    del r1[i]
    r2.insert(j, c)
    return r1, r2
