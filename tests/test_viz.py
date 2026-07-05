import matplotlib
matplotlib.use('Agg')

from types import SimpleNamespace

from cvrptw.io import calculate_distances
from cvrptw.model import Customer, Solution, Vehicle
from cvrptw.viz import draw_best_solutions, draw_solution, plot_ils_stats


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


def _make_solution():
    depot = Customer(0, 0, 0, 0, 0, 1000, 0)
    c1 = Customer(1, 10, 0, 1, 0, 1000, 0)
    customers = [depot, c1]
    distances = calculate_distances(customers)
    v = Vehicle(10, depot, distances)
    v.visit(c1)
    v.visit(depot)
    return Solution([v])


def test_draw_best_solutions_does_not_raise():
    sol_a = _make_solution()
    sol_b = _make_solution()
    values = [
        ('inst_a.txt', sol_a.distance, len(sol_a), sol_a),
        ('inst_b.txt', sol_b.distance, len(sol_b), sol_b),
    ]
    draw_best_solutions(values)


def test_draw_best_solutions_empty_list_does_not_raise():
    draw_best_solutions([])


def test_plot_ils_stats_does_not_raise():
    stats = [
        SimpleNamespace(elapsed_s=1.0, distance=100.0, improved=True,
                         cross_gain=5.0, intra_relocate_gain=0.0, exchange_gain=0.0),
        SimpleNamespace(elapsed_s=2.0, distance=95.0, improved=False,
                         cross_gain=5.0, intra_relocate_gain=2.0, exchange_gain=0.0),
    ]
    plot_ils_stats(stats, title='test')


def test_plot_ils_stats_empty_list_does_not_raise():
    plot_ils_stats([], title='test')
