#!/usr/bin/env python3
"""Fetch SINTEF best-known solutions and save them as local .sol files.

For each instance name, downloads SINTEF's detailed-solution file, replays
its routes against data/instances/solomon/{name}.txt, verifies the result
(feasible + distance matches SOLOMON_100_BKS), and saves it in this repo's
.sol format to data/solutions/solomon-bks/{name}.sol. Defaults to the six
instances shown in the README.

Usage:
    uv run python scripts/fetch_bks_solutions.py [name ...]
    uv run python scripts/fetch_bks_solutions.py --all
"""
import argparse
import urllib.request
from pathlib import Path

from cvrptw.bks import SOLOMON_100_BKS
from cvrptw.io import load_instance, parse_sintef_routes, save_solution, solution_from_routes
from cvrptw.operators import verify_solution

REPO_ROOT = Path(__file__).parent.parent
README_INSTANCES = ['c101', 'c201', 'r101', 'r201', 'rc101', 'rc201']
SINTEF_URL = 'https://www.sintef.no/contentassets/adf48e65e3a84dd6871eb7586707675d/{name}.txt'
DISTANCE_TOL = 0.005  # SINTEF reports distances rounded to two decimals


def fetch_bks_solution(name: str, instances_dir: Path, out_dir: Path) -> Path:
    """Download, verify, and save one BKS solution; returns the .sol path."""
    # SINTEF returns 403 to urllib's default User-Agent
    request = urllib.request.Request(SINTEF_URL.format(name=name),
                                     headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request) as response:
        text = response.read().decode()
    routes = parse_sintef_routes(text)
    if not routes:
        raise ValueError(f'{name}: no routes found in the SINTEF file')

    inst = load_instance(instances_dir / f'{name}.txt')
    sol = solution_from_routes(routes, inst)

    violations = verify_solution(sol, inst)
    if violations:
        raise ValueError(f'{name}: BKS solution fails verification: {violations}')
    bks_vehicles, bks_distance = SOLOMON_100_BKS[name]
    if len(sol) != bks_vehicles or abs(sol.distance - bks_distance) > DISTANCE_TOL:
        raise ValueError(
            f'{name}: rebuilt BKS is {sol.distance:.2f}/{len(sol)} vehicles, '
            f'expected {bks_distance:.2f}/{bks_vehicles}'
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{name}.sol'
    save_solution(out_path, sol)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='*', default=README_INSTANCES,
                        help=f'instance names to fetch (default: {" ".join(README_INSTANCES)})')
    parser.add_argument('--all', action='store_true',
                        help='fetch every Solomon 100-customer instance')
    args = parser.parse_args()
    names = sorted(SOLOMON_100_BKS) if args.all else args.names

    for name in names:
        out_path = fetch_bks_solution(
            name,
            instances_dir=REPO_ROOT / 'data' / 'instances' / 'solomon',
            out_dir=REPO_ROOT / 'data' / 'solutions' / 'solomon-bks',
        )
        bks_vehicles, bks_distance = SOLOMON_100_BKS[name]
        print(f'{out_path.relative_to(REPO_ROOT)}  ({bks_distance:.2f}, {bks_vehicles} vehicles)')


if __name__ == '__main__':
    main()
