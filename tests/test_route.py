import pytest

from cvrptw.model import Vehicle


def test_route_depot_and_length(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    v = Vehicle(capacity, depot, distances)
    assert v.route.customers[0] is depot
    assert v.route.length() == 1
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)
    assert v.route.length() == 4


def test_route_distance_caching(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)
    # Cached _distance must equal the explicit sum of leg_distances
    assert v.route.distance == pytest.approx(sum(v.route.leg_distances))
    assert v.route.distance == pytest.approx(40.0)          # 10 + 10 + 20


def test_route_total_time(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)
    assert v.route.total_time == pytest.approx(v.route.time_points[-1])
    assert v.route.total_time == pytest.approx(50.0)        # 10+5 + 10+5 + 20
