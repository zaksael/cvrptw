from .model import Vehicle


def rng(v, with_last):
    return range(1, v.length()) if with_last else range(1, v.length() - 1)


def check_route(route, capacity, distances):
    v = Vehicle(capacity, route[0], distances)
    for c in route[1:]:
        if v.can_visit(c):
            v.visit(c)
        else:
            return False, v
    return True, v


def cross(v1, i, v2, j):
    return v1.route[:i] + v2.route[j:], v2.route[:j] + v1.route[i:]


def exchange(v1, i, v2, j):
    r1, r2 = v1.route[:], v2.route[:]
    r1[i], r2[j] = r2[j], r1[i]
    return r1, r2


def relocate(v1, i, v2, j):
    r1, r2 = v1.route[:], v2.route[:]
    c = r1[i]
    del r1[i]
    r2.insert(j, c)
    return r1, r2
