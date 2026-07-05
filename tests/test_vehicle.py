import numpy as np
import pytest

from cvrptw.model import Customer, Vehicle


def test_visit_updates_capacity_and_time(tiny):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    assert v.left_capacity == capacity - c1.demand      # 30 - 10 = 20
    assert v.total_time == pytest.approx(15.0)          # travel 10 + service 5
    assert v.route.customers == [depot, c1]


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


def test_can_visit_exactly_at_customer_due_date():
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c = Customer(1, 10, 0, 5, 0, 10, 0)                # due_date=10, travel=10 → arrive exactly at 10
    distances = np.array([[0., 10.], [10., 0.]])
    v = Vehicle(100, depot, distances)
    assert v.can_visit(c)                               # arrival == due_date is valid


def test_cannot_visit_past_customer_due_date():
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c = Customer(1, 10, 0, 5, 0, 9, 0)                 # due_date=9, travel=10 → arrive at 10 > 9
    distances = np.array([[0., 10.], [10., 0.]])
    v = Vehicle(100, depot, distances)
    assert not v.can_visit(c)


def test_can_visit_returns_exactly_at_depot_closing():
    depot = Customer(0, 0, 0, 0, 0, 25, 0)             # depot closes at t=25
    c = Customer(1, 10, 0, 5, 0, 15, 5)                # travel=10, service=5, return=10 → 25 total
    distances = np.array([[0., 10.], [10., 0.]])
    v = Vehicle(100, depot, distances)
    assert v.can_visit(c)                               # 10+5+10=25 == depot.due_date is valid


def test_cannot_visit_when_return_arrives_late():
    depot = Customer(0, 0, 0, 0, 0, 24, 0)             # depot closes at t=24
    c = Customer(1, 10, 0, 5, 0, 15, 5)                # return at t=25 > 24
    distances = np.array([[0., 10.], [10., 0.]])
    v = Vehicle(100, depot, distances)
    assert not v.can_visit(c)
