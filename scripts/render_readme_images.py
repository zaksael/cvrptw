#!/usr/bin/env python3
"""Render the README solution images from saved .sol files.

For each instance name, reads data/instances/solomon/{name}.txt and
data/solutions/solomon/{name}.sol and writes docs/solutions/{name}.png,
with the best-known solution (data/solutions/solomon-bks/{name}.sol, if
fetched) drawn behind ours in light grey. Defaults to the six instances
shown in the README (one per Solomon class).

Usage:
    uv run python scripts/render_readme_images.py [name ...]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from cvrptw.viz import render_solution_images

REPO_ROOT = Path(__file__).parent.parent
README_INSTANCES = ['c101', 'c201', 'r101', 'r201', 'rc101', 'rc201']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='*', default=README_INSTANCES,
                        help=f'instance names to render (default: {" ".join(README_INSTANCES)})')
    args = parser.parse_args()

    saved = render_solution_images(
        args.names,
        instances_dir=REPO_ROOT / 'data' / 'instances' / 'solomon',
        solutions_dir=REPO_ROOT / 'data' / 'solutions' / 'solomon',
        out_dir=REPO_ROOT / 'docs' / 'solutions',
        bks_dir=REPO_ROOT / 'data' / 'solutions' / 'solomon-bks',
    )
    for path in saved:
        print(path.relative_to(REPO_ROOT))


if __name__ == '__main__':
    main()
