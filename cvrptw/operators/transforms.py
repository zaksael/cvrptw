from ..model import Customer, Vehicle


def customer_indices(v: Vehicle, with_last: bool) -> range:
    return range(1, v.length()) if with_last else range(1, v.length() - 1)


def two_opt(v: Vehicle, i: int, j: int) -> list[Customer]:
    c = v.route.customers
    return c[:i + 1] + c[i + 1:j + 1][::-1] + c[j + 1:]


def cross(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route.customers, v2.route.customers
    return r1[:i] + r2[j:], r2[:j] + r1[i:]


def exchange(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route.customers[:], v2.route.customers[:]
    r1[i], r2[j] = r2[j], r1[i]
    return r1, r2


def relocate(v1: Vehicle, i: int, v2: Vehicle, j: int) -> tuple[list[Customer], list[Customer]]:
    r1, r2 = v1.route.customers[:], v2.route.customers[:]
    c = r1[i]
    del r1[i]
    r2.insert(j, c)
    return r1, r2


def or_opt(v1: Vehicle, i: int, seg_len: int, v2: Vehicle, j: int, reverse: bool) -> tuple[list[Customer], list[Customer]]:
    r1 = v1.route.customers[:]
    seg = r1[i:i + seg_len]
    del r1[i:i + seg_len]
    if reverse:
        seg = seg[::-1]
    r2 = v2.route.customers[:]
    r2[j:j] = seg
    return r1, r2
