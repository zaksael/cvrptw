import matplotlib
matplotlib.use('Agg')

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.viz import draw_solution


def test_draw_solution_saves_png_file(tmp_path):
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c1 = Customer(1, 10, 0, 1, 0, 1000, 0)
    customers = [depot, c1]
    distances = calculate_distances(customers)

    v = Vehicle(10, depot, distances)
    v.visit(c1)
    v.visit(depot)
    sol = Solution([v])

    save_path = tmp_path / 'solution.png'
    draw_solution(sol, title='test', save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
