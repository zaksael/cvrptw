import numpy as np
import pytest

from cvrptw.model import Customer, Vehicle, get_distance, remove_empty_routes


def test_visit_updates_capacity_and_time(tiny):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    assert v.left_capacity == capacity - c1.demand      # 30 - 10 = 20
    assert v.total_time == pytest.approx(15.0)          # travel 10 + service 5
    assert v.route == [depot, c1]


def test_visit_waits_for_ready_time():
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c = Customer(1, 1, 0, 5, 100, 500, 0)  # only 1 unit away but ready_time=100
    distances = np.array([[0., 1.], [1., 0.]])
    v = Vehicle(10, depot, distances)
    v.visit(c)
    assert v.total_time == pytest.approx(100.0)         # waited until ready_time


def test_cannot_visit_over_capacity(tiny):
    customers, distances, _ = tiny
    depot, c1, c2, c3 = customers
    v = Vehicle(15, depot, distances)                   # room for one customer only
    v.visit(c1)
    assert not v.can_visit(c2)


def test_cannot_visit_past_due_date():
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c = Customer(1, 10, 0, 5, 0, 5, 0)                 # due_date=5, travel time=10
    distances = np.array([[0., 10.], [10., 0.]])
    v = Vehicle(100, depot, distances)
    assert not v.can_visit(c)


def test_cannot_visit_when_no_time_to_return():
    depot = Customer(0, 0, 0, 0, 0, 25, 0)             # depot closes at t=25
    c = Customer(1, 10, 0, 5, 0, 15, 5)                # travel=10, service=5, return=10 → 25 total
    distances = np.array([[0., 10.], [10., 0.]])
    v = Vehicle(100, depot, distances)
    assert not v.can_visit(c)                           # 10+5+10=25, not < 25


def test_remove_empty_routes(tiny):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]

    non_empty = Vehicle(capacity, depot, distances)
    non_empty.visit(c1)
    non_empty.visit(depot)                              # length = 3 → kept

    empty = Vehicle(capacity, depot, distances)
    empty.visit(depot)                                  # length = 2 → removed

    result = remove_empty_routes([non_empty, empty])
    assert result == [non_empty]


def test_get_distance(tiny):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(depot)                                      # depot → c1 (+10), c1 → depot (+10)
    assert get_distance([v]) == pytest.approx(20.0)
