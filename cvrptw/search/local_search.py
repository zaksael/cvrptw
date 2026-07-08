import random
from dataclasses import dataclass
from functools import partial

from ..model import Solution
from .budget import AttemptBudget, LimitReached
from .inter import (
    apply_operator, apply_or_opt, apply_relocate,
    cross_gate, cross_suffix, exchange_gate, exchange_suffix,
)
from .intra import intra_or_opt, intra_relocate, intra_two_opt

# First-improvement cascade: each operator is tried only if all previous ones
# fail on this pass; any accepted move restarts from the top. The third column
# marks the inter-route operators that take the granular-neighborhood gate
# (intra routes are short — gating them isn't worth the plumbing).
_CASCADE = (
    ('cross', partial(apply_operator, operator=cross_suffix, gate=cross_gate, with_last=True), True),
    ('intra_relocate', intra_relocate, False),
    ('two_opt', intra_two_opt, False),
    ('intra_or_opt', intra_or_opt, False),
    ('exchange', partial(apply_operator, operator=exchange_suffix, gate=exchange_gate, with_last=False), True),
    ('relocate', apply_relocate, True),
    ('or_opt', apply_or_opt, True),
)

OPERATOR_NAMES = tuple(name for name, _, _ in _CASCADE)


@dataclass
class LSStats:
    n_attempts: int
    improvements: dict[str, int]
    gains: dict[str, float]


def local_search(sol: Solution, max_attempts: int = 200_000, deadline: float | None = None,
                 rng: random.Random = random,
                 neighbors: list[set[int]] | None = None) -> tuple[bool, Solution, LSStats]:
    """neighbors (from build_neighbor_sets) restricts inter-route operators to
    moves creating at least one short arc; None evaluates every candidate."""
    budget = AttemptBudget(max_attempts=max_attempts, deadline=deadline)

    result = sol
    changes_made = False
    improvements = dict.fromkeys(OPERATOR_NAMES, 0)
    gains = dict.fromkeys(OPERATOR_NAMES, 0.0)
    try:
        improved = True
        while improved:
            improved = False
            for name, op, gated in _CASCADE:
                if gated:
                    done, result, gain = op(result, budget=budget, rng=rng, neighbors=neighbors)
                else:
                    done, result, gain = op(result, budget=budget, rng=rng)
                if done:
                    changes_made = True
                    improvements[name] += 1
                    gains[name] += gain
                    improved = True
                    break
    except LimitReached:
        pass
    return changes_made, result, LSStats(n_attempts=budget.n_attempts, improvements=improvements, gains=gains)
