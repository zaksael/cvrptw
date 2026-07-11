from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from .io import load_instance, load_solution

if TYPE_CHECKING:
    from .model import Solution


def _draw_routes(ax, sol: Solution, depot_markersize: int) -> None:
    scalar_map = cmx.ScalarMappable(
        norm=colors.Normalize(vmin=0, vmax=len(sol)),
        cmap=plt.get_cmap('gist_rainbow'),
    )
    vehicles = sorted(sol, key=lambda v: v.length())
    for v_i, v in enumerate(vehicles):
        c_x = np.array([c.x for c in v.route.customers])
        c_y = np.array([c.y for c in v.route.customers])
        edges = np.array([[i, i + 1] for i in range(v.route.length() - 1)])
        color = scalar_map.to_rgba(v_i)
        ax.plot(c_x[edges.T], c_y[edges.T],
                linestyle='-', color=color, linewidth=1.5,
                markerfacecolor=color, marker='o', markersize=4)
    if vehicles:
        ax.plot(c_x[0], c_y[0], color='green', marker='s', markersize=depot_markersize)


def _draw_background_routes(ax, sol: Solution) -> None:
    """Draw all routes in one flat light grey behind the foreground solution
    (foreground lines default to zorder 2). No depot marker — the foreground
    draws it."""
    for v in sol:
        c_x = [c.x for c in v.route.customers]
        c_y = [c.y for c in v.route.customers]
        ax.plot(c_x, c_y, linestyle='-', color='0.8', linewidth=1.2,
                markerfacecolor='0.8', marker='o', markersize=3, zorder=1)


def draw_solution(sol: Solution, title: str = '', save_path: Path | str | None = None,
                  show: bool = True, background: Solution | None = None) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    if background is not None:
        _draw_background_routes(ax, background)
    _draw_routes(ax, sol, depot_markersize=20)
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def render_solution_images(
    names: list[str],
    instances_dir: Path | str,
    solutions_dir: Path | str,
    out_dir: Path | str,
    show: bool = False,
    bks_dir: Path | str | None = None,
) -> list[Path]:
    """Render one route plot per instance name from its saved solution.

    For each name, loads instances_dir/{name}.txt and solutions_dir/{name}.sol
    and saves out_dir/{name}.png (out_dir is created if missing). If bks_dir
    holds a {name}.sol, that solution is drawn behind ours in light grey and
    its numbers are appended to the title; a missing file just skips the
    background. Batch renderer, so show defaults to False. Returns the saved
    image paths.
    """
    instances_dir, solutions_dir, out_dir = Path(instances_dir), Path(solutions_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in names:
        inst = load_instance(instances_dir / f'{name}.txt')
        sol = load_solution(solutions_dir / f'{name}.sol', inst)
        title = f'{name}: distance {sol.distance:.2f}, {len(sol)} vehicles'
        bks = None
        if bks_dir is not None and (bks_path := Path(bks_dir) / f'{name}.sol').exists():
            bks = load_solution(bks_path, inst)
            title += f' (BKS {bks.distance:.2f}, {len(bks)} vehicles)'
        save_path = out_dir / f'{name}.png'
        draw_solution(sol, title=title, save_path=save_path, show=show, background=bks)
        saved.append(save_path)
    return saved


def draw_best_solutions(values: list[tuple[str, float, int, Solution]], show: bool = True) -> None:
    n = len(values)
    if n == 0:
        return
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(20, 6 * nrows))
    for i, (name, distance, vehicles, sol) in enumerate(values, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        ax.set_title(f"{name[:-4]}: distance={distance:.2f}, vehicles={vehicles}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        _draw_routes(ax, sol, depot_markersize=10)
    if show:
        plt.show()
    plt.close(fig)


def plot_ils_stats(stats: list, title: str = '', show: bool = True) -> None:
    if not stats:
        return

    elapsed = [s.elapsed_s for s in stats]
    distances = [s.distance for s in stats]

    names = list(stats[0].gains)
    running = dict.fromkeys(names, 0.0)
    cumulative: dict[str, list[float]] = {name: [] for name in names}
    for s in stats:
        for name in names:
            running[name] += s.gains[name]
            cumulative[name].append(running[name])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    if title:
        fig.suptitle(title, fontsize=13)

    ax1.plot(elapsed, distances, color='steelblue', linewidth=1.5, label='best distance')
    imp_t = [elapsed[i] for i, s in enumerate(stats) if s.improved]
    imp_d = [distances[i] for i, s in enumerate(stats) if s.improved]
    ax1.scatter(imp_t, imp_d, color='green', zorder=5, s=40, label='improvement')
    ax1.set_xlabel('elapsed (s)')
    ax1.set_ylabel('best distance')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    for name in names:
        ax2.plot(elapsed, cumulative[name], label=name)
    ax2.set_xlabel('elapsed (s)')
    ax2.set_ylabel('cumulative gain')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)
