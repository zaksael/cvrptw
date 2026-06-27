from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .model import Vehicle


def draw_solution(sol: list[Vehicle], title: str = '') -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    scalar_map = cmx.ScalarMappable(
        norm=colors.Normalize(vmin=0, vmax=len(sol)),
        cmap=plt.get_cmap('gist_rainbow'),
    )
    for v_i, v in enumerate(sorted(sol, key=lambda v: v.length())):
        c_x = np.array([c.x for c in v.route])
        c_y = np.array([c.y for c in v.route])
        edges = np.array([[i, i + 1] for i in range(len(v.route) - 1)])
        color = scalar_map.to_rgba(v_i)
        ax.plot(c_x[edges.T], c_y[edges.T],
                linestyle='-', color=color, linewidth=1.5,
                markerfacecolor=color, marker='o', markersize=4)
    ax.plot(c_x[0], c_y[0], color='green', marker='s', markersize=20)
    plt.title(title)
    plt.show()


def draw_best_solutions(values: np.ndarray) -> None:
    fig = plt.figure(figsize=(20, 30))
    for i, (name, distance, vehicles, sol) in enumerate(values, start=1):
        ax = fig.add_subplot(5, 2, i)
        ax.set_title(f"{name[:-4]}: distance={distance:.2f}, vehicles={vehicles}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        scalar_map = cmx.ScalarMappable(
            norm=colors.Normalize(vmin=0, vmax=len(sol)),
            cmap=plt.get_cmap('gist_rainbow'),
        )
        for v_i, v in enumerate(sorted(sol, key=lambda v: v.length())):
            c_x = np.array([c.x for c in v.route])
            c_y = np.array([c.y for c in v.route])
            edges = np.array([[i, i + 1] for i in range(len(v.route) - 1)])
            color = scalar_map.to_rgba(v_i)
            ax.plot(c_x[edges.T], c_y[edges.T],
                    linestyle='-', color=color, linewidth=1.5,
                    markerfacecolor=color, marker='o', markersize=4)
        ax.plot(c_x[0], c_y[0], color='green', marker='s', markersize=10)
