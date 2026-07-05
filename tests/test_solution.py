import pytest

from cvrptw.model import Solution, Vehicle


def test_without_empty_routes(tiny):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]

    non_empty = Vehicle(capacity, depot, distances)
    non_empty.visit(c1)
    non_empty.visit(depot)                              # length = 3 → kept

    empty = Vehicle(capacity, depot, distances)
    empty.visit(depot)                                  # length = 2 → removed

    result = Solution([non_empty, empty]).without_empty_routes()
    assert result.vehicles == [non_empty]


def test_solution_distance(tiny):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(depot)                                      # depot → c1 (+10), c1 → depot (+10)
    assert Solution([v]).distance == pytest.approx(20.0)
