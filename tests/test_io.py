import os
import tempfile
from pathlib import Path

from cvrptw.io import load_instance, save_solution
from cvrptw.model import Vehicle

C108 = Path(__file__).parent.parent / 'ils' / 'resources' / 'instances' / 'C108.txt'


def test_read_vehicle_and_capacity():
    inst = load_instance(C108)
    assert inst.n_vehicles == 25
    assert inst.capacity == 200


def test_read_customer_count_and_depot():
    inst = load_instance(C108)
    assert len(inst.customers) == 101                   # depot + 100 customers
    assert inst.depot.cust_id == 0
    assert inst.depot.demand == 0
    assert inst.depot.x == 40
    assert inst.depot.y == 50


def test_save_solution_format(tiny):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)

    with tempfile.NamedTemporaryFile(suffix='.sol', delete=False) as f:
        path = f.name
    try:
        save_solution(path, [v])
        tokens = Path(path).read_text().split()
        # Each stop is a (cust_id, time) pair: "0 0.0000 1 10.0000 2 25.0000 0 50.0000"
        cust_ids = [int(tokens[i]) for i in range(0, len(tokens), 2)]
        assert cust_ids == [0, 1, 2, 0]
    finally:
        os.unlink(path)
