"""Best-known solutions for Solomon's 100-customer VRPTW benchmark.

Values from SINTEF TOP (https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/),
retrieved 2026-07-06. SINTEF reports the hierarchical objective — 1) minimize
vehicles, 2) minimize total distance — with double-precision Euclidean
distances rounded to two decimals, the same distance convention this solver
uses. Since 2026-07-06 the solver pursues the same hierarchical objective by
default (`ils(minimize_vehicles=True)`); with `minimize_vehicles=False` it
minimizes distance only and may beat the BKS distance by using more vehicles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

# instance name -> (vehicles, total distance)
SOLOMON_100_BKS: dict[str, tuple[int, float]] = {
    'c101': (10, 828.94),
    'c102': (10, 828.94),
    'c103': (10, 828.06),
    'c104': (10, 824.78),
    'c105': (10, 828.94),
    'c106': (10, 828.94),
    'c107': (10, 828.94),
    'c108': (10, 828.94),
    'c109': (10, 828.94),
    'c201': (3, 591.56),
    'c202': (3, 591.56),
    'c203': (3, 591.17),
    'c204': (3, 590.60),
    'c205': (3, 588.88),
    'c206': (3, 588.49),
    'c207': (3, 588.29),
    'c208': (3, 588.32),
    'r101': (19, 1650.80),
    'r102': (17, 1486.12),
    'r103': (13, 1292.68),
    'r104': (9, 1007.31),
    'r105': (14, 1377.11),
    'r106': (12, 1252.03),
    'r107': (10, 1104.66),
    'r108': (9, 960.88),
    'r109': (11, 1194.73),
    'r110': (10, 1118.84),
    'r111': (10, 1096.73),
    'r112': (9, 982.14),
    'r201': (4, 1252.37),
    'r202': (3, 1191.70),
    'r203': (3, 939.50),
    'r204': (2, 825.52),
    'r205': (3, 994.43),
    'r206': (3, 906.14),
    'r207': (2, 890.61),
    'r208': (2, 726.82),
    'r209': (3, 909.16),
    'r210': (3, 939.37),
    'r211': (2, 885.71),
    'rc101': (14, 1696.95),
    'rc102': (12, 1554.75),
    'rc103': (11, 1261.67),
    'rc104': (10, 1135.48),
    'rc105': (13, 1629.44),
    'rc106': (11, 1424.73),
    'rc107': (11, 1230.48),
    'rc108': (10, 1139.82),
    'rc201': (4, 1406.94),
    'rc202': (3, 1365.65),
    'rc203': (3, 1049.62),
    'rc204': (3, 798.46),
    'rc205': (4, 1297.65),
    'rc206': (3, 1146.32),
    'rc207': (3, 1061.14),
    'rc208': (3, 828.14),
}


class _ResultLike(Protocol):
    name: str
    distance: float
    n_vehicles: int


@dataclass
class BKSComparison:
    name: str
    distance: float
    n_vehicles: int
    bks_distance: float
    bks_vehicles: int
    gap_pct: float  # (distance - bks_distance) / bks_distance * 100
    extra_vehicles: int


def compare_to_bks(results: Iterable[_ResultLike]) -> list[BKSComparison]:
    """Match results against SOLOMON_100_BKS by instance name.

    `name` may carry a file suffix and any case ('c101', 'C101.txt', ...).
    Results whose name is not a Solomon 100-customer instance are skipped.
    """
    rows = []
    for r in results:
        name = Path(r.name).stem.lower()
        if name not in SOLOMON_100_BKS:
            continue
        bks_vehicles, bks_distance = SOLOMON_100_BKS[name]
        rows.append(BKSComparison(
            name=name,
            distance=r.distance,
            n_vehicles=r.n_vehicles,
            bks_distance=bks_distance,
            bks_vehicles=bks_vehicles,
            gap_pct=round((r.distance - bks_distance) / bks_distance * 100, 2),
            extra_vehicles=r.n_vehicles - bks_vehicles,
        ))
    return rows


def format_bks_table(rows: list[BKSComparison]) -> str:
    """Aligned text table with a mean-gap summary line."""
    header = f'{"instance":<10}{"dist":>10}{"bks":>10}{"gap%":>8}{"veh":>6}{"bks_veh":>9}'
    lines = [header, '-' * len(header)]
    for row in sorted(rows, key=lambda r: r.name):
        lines.append(
            f'{row.name:<10}{row.distance:>10.2f}{row.bks_distance:>10.2f}'
            f'{row.gap_pct:>+8.2f}{row.n_vehicles:>6d}{row.bks_vehicles:>9d}'
        )
    if rows:
        mean_gap = sum(r.gap_pct for r in rows) / len(rows)
        at_bks = sum(1 for r in rows if r.distance <= r.bks_distance + 1e-9)
        at_bks_veh = sum(1 for r in rows if r.extra_vehicles <= 0)
        lines.append('-' * len(header))
        lines.append(
            f'mean gap {mean_gap:+.2f}%, at/below BKS distance: '
            f'{at_bks}/{len(rows)}, at BKS vehicles: {at_bks_veh}/{len(rows)} '
            f'of {len(SOLOMON_100_BKS)} instances'
        )
    return '\n'.join(lines)
