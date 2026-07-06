from dataclasses import dataclass
from pathlib import Path

from cvrptw.bks import SOLOMON_100_BKS, compare_to_bks, format_bks_table

INSTANCES_DIR = Path(__file__).parent.parent / 'data' / 'instances' / 'solomon'


@dataclass
class FakeResult:
    name: str
    distance: float
    n_vehicles: int


def test_bks_covers_exactly_the_shipped_solomon_instances():
    shipped = {p.stem for p in INSTANCES_DIR.glob('*.txt')}
    assert shipped == set(SOLOMON_100_BKS)
    assert len(SOLOMON_100_BKS) == 56


def test_bks_values_are_sane():
    for name, (vehicles, distance) in SOLOMON_100_BKS.items():
        assert 2 <= vehicles <= 19, name
        assert 588 <= distance <= 1697, name


def test_compare_matches_by_stem_case_insensitively_and_skips_unknown():
    results = [
        FakeResult('C101.txt', 828.94, 10),   # exact BKS
        FakeResult('r112', 1080.35, 10),      # 10% above, one extra vehicle
        FakeResult('X999.txt', 1.0, 1),       # not a Solomon instance
    ]
    rows = compare_to_bks(results)
    assert [r.name for r in rows] == ['c101', 'r112']

    c101, r112 = rows
    assert c101.gap_pct == 0.0
    assert c101.extra_vehicles == 0
    assert r112.bks_distance == 982.14
    assert r112.gap_pct == 10.0
    assert r112.extra_vehicles == 1


def test_format_table_sorts_rows_and_summarizes():
    rows = compare_to_bks([
        FakeResult('r112', 1080.35, 10),
        FakeResult('c101', 828.94, 10),
    ])
    table = format_bks_table(rows)
    lines = table.splitlines()
    assert lines[0].split() == ['instance', 'dist', 'bks', 'gap%', 'veh', 'bks_veh']
    assert lines[2].split() == ['c101', '828.94', '828.94', '+0.00', '10', '10']
    assert lines[3].split() == ['r112', '1080.35', '982.14', '+10.00', '10', '9']
    assert lines[-1] == 'mean gap +5.00%, at/below BKS distance: 1/2 of 56 instances'


def test_format_table_empty():
    assert format_bks_table([]).splitlines()[-1].startswith('-')
