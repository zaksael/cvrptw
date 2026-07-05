from dataclasses import dataclass
from functools import partial

from ..model import Solution
from .budget import AttemptBudget, LimitReached
from .inter import apply_operator, apply_or_opt, apply_relocate, cross_suffix, exchange_suffix
from .intra import intra_or_opt, intra_relocate, intra_two_opt

# First-improvement cascade: each operator is tried only if all previous ones
# fail on this pass; any accepted move restarts from the top.
_CASCADE = (
    ('cross', partial(apply_operator, operator=cross_suffix, with_last=True)),
    ('intra_relocate', intra_relocate),
    ('two_opt', intra_two_opt),
    ('intra_or_opt', intra_or_opt),
    ('exchange', partial(apply_operator, operator=exchange_suffix, with_last=False)),
    ('relocate', apply_relocate),
    ('or_opt', apply_or_opt),
)

OPERATOR_NAMES = tuple(name for name, _ in _CASCADE)


@dataclass
class LSStats:
    n_attempts: int
    improvements: dict[str, int]
    gains: dict[str, float]


def local_search(sol: Solution, max_attempts: int = 200_000, deadline: float | None = None) -> tuple[bool, Solution, LSStats]:
    budget = AttemptBudget(max_attempts=max_attempts, deadline=deadline)

    result = sol
    changes_made = False
    improvements = dict.fromkeys(OPERATOR_NAMES, 0)
    gains = dict.fromkeys(OPERATOR_NAMES, 0.0)
    try:
        improved = True
        while improved:
            improved = False
            for name, op in _CASCADE:
                done, result, gain = op(result, budget=budget)
                if done:
                    changes_made = True
                    improvements[name] += 1
                    gains[name] += gain
                    improved = True
                    break
    except LimitReached:
        pass
    return changes_made, result, LSStats(n_attempts=budget.n_attempts, improvements=improvements, gains=gains)
