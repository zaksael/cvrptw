from ..model import Customer


def _cross2d(o: Customer, a: Customer, b: Customer) -> int:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def segments_cross(a: Customer, b: Customer, c: Customer, d: Customer) -> bool:
    d1 = _cross2d(c, d, a)
    d2 = _cross2d(c, d, b)
    d3 = _cross2d(a, b, c)
    d4 = _cross2d(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0
