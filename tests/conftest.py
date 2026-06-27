import pytest

from cvrptw.io import calculate_distances
from cvrptw.model import Customer


@pytest.fixture
def tiny():
    """Depot + 3 collinear customers on the x-axis at x=10, 20, 30.
    Distances are Euclidean integers: d(0,1)=10, d(1,2)=10, d(2,3)=10, d(0,3)=30.
    """
    depot = Customer(0,  0, 0,  0, 0, 1000, 0)
    c1    = Customer(1, 10, 0, 10, 0,  500, 5)
    c2    = Customer(2, 20, 0, 10, 0,  500, 5)
    c3    = Customer(3, 30, 0, 10, 0,  500, 5)
    customers = [depot, c1, c2, c3]
    distances = calculate_distances(customers)
    capacity = 30
    return customers, distances, capacity
