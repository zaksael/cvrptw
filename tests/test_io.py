import os
import tempfile
from pathlib import Path

from cvrptw.io import (
    load_instance,
    load_solution,
    parse_sintef_routes,
    save_solution,
    solution_from_routes,
)
from cvrptw.model import Instance, Solution, Vehicle
from cvrptw.operators import verify_solution
from cvrptw.solver import get_greedy_solution

C108 = Path(__file__).parent.parent / 'data' / 'instances' / 'solomon' / 'c108.txt'


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


def test_load_instance_ignores_blank_lines(tmp_path):
    text = C108.read_text()
    padded = tmp_path / 'c108_padded.txt'
    padded.write_text(text + '\n\n   \n')
    inst = load_instance(padded)
    assert inst.n_vehicles == 25
    assert len(inst.customers) == 101


def test_load_solution_roundtrip(tiny, tmp_path):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)
    original = Solution(vehicles=[v])

    inst = Instance(2, capacity, customers, distances)
    path = tmp_path / 'tiny.sol'
    save_solution(path, original)
    loaded = load_solution(path, inst)

    assert len(loaded) == 1
    assert [c.cust_id for c in loaded.vehicles[0].route.customers] == [0, 1, 2, 0]
    assert loaded.vehicles[0].route.time_points == v.route.time_points
    assert loaded.distance == original.distance


def test_load_solution_roundtrip_real_instance(tmp_path):
    inst = load_instance(C108)
    original = get_greedy_solution(inst)
    path = tmp_path / 'c108.sol'
    save_solution(path, original)

    loaded = load_solution(path, inst)

    assert len(loaded) == len(original)
    assert loaded.distance == original.distance
    assert loaded.missing_customers(inst) == set()
    assert verify_solution(loaded, inst) == []


def test_load_solution_ignores_blank_lines(tiny, tmp_path):
    customers, distances, capacity = tiny
    depot, c1 = customers[0], customers[1]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(depot)

    inst = Instance(2, capacity, customers, distances)
    path = tmp_path / 'padded.sol'
    save_solution(path, Solution(vehicles=[v]))
    path.write_text(path.read_text() + '\n\n   \n')

    loaded = load_solution(path, inst)
    assert len(loaded) == 1


SINTEF_SNIPPET = """\
Instance name : tiny
Authors       : Someone
Date          : 01.01.2004 12:00:00
Reference     : Some reference

Solution
Route  1 : 1 2
Route  2 : 3
"""


def test_parse_sintef_routes():
    assert parse_sintef_routes(SINTEF_SNIPPET) == [[1, 2], [3]]


def test_parse_sintef_routes_empty_text():
    assert parse_sintef_routes('Instance name : x\n') == []


def test_solution_from_routes_builds_depot_anchored_solution(tiny):
    customers, distances, capacity = tiny
    inst = Instance(2, capacity, customers, distances)

    sol = solution_from_routes([[1, 2], [3]], inst)

    assert len(sol) == 2
    assert [c.cust_id for c in sol.vehicles[0].route.customers] == [0, 1, 2, 0]
    assert [c.cust_id for c in sol.vehicles[1].route.customers] == [0, 3, 0]
    assert verify_solution(sol, inst) == []


def test_solution_from_routes_matches_load_solution(tiny, tmp_path):
    customers, distances, capacity = tiny
    depot, c1, c2 = customers[0], customers[1], customers[2]
    v = Vehicle(capacity, depot, distances)
    v.visit(c1)
    v.visit(c2)
    v.visit(depot)
    inst = Instance(2, capacity, customers, distances)

    path = tmp_path / 'tiny.sol'
    save_solution(path, Solution(vehicles=[v]))
    loaded = load_solution(path, inst)
    built = solution_from_routes([[1, 2]], inst)

    assert [c.cust_id for c in built.vehicles[0].route.customers] == \
        [c.cust_id for c in loaded.vehicles[0].route.customers]
    assert built.vehicles[0].route.time_points == loaded.vehicles[0].route.time_points


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
        save_solution(path, sol=Solution(vehicles=[v]))
        tokens = Path(path).read_text().split()
        # Each stop is a (cust_id, time) pair: "0 0.0000 1 10.0000 2 25.0000 0 50.0000"
        cust_ids = [int(tokens[i]) for i in range(0, len(tokens), 2)]
        assert cust_ids == [0, 1, 2, 0]
    finally:
        os.unlink(path)
