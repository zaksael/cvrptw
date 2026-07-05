from dataclasses import dataclass

from ..model import Solution
from .budget import AttemptBudget, LimitReached
from .inter import apply_operator, apply_or_opt, apply_relocate, cross_suffix, exchange_suffix
from .intra import intra_or_opt, intra_relocate, intra_two_opt


@dataclass
class LSStats:
    n_attempts: int
    cross_improvements: int
    intra_relocate_improvements: int
    exchange_improvements: int
    two_opt_improvements: int
    intra_or_opt_improvements: int
    or_opt_improvements: int
    relocate_improvements: int
    cross_gain: float
    intra_relocate_gain: float
    exchange_gain: float
    two_opt_gain: float
    intra_or_opt_gain: float
    or_opt_gain: float
    relocate_gain: float


def local_search(sol: Solution, max_attempts: int = 200_000, deadline: float | None = None) -> tuple[bool, Solution, LSStats]:
    budget = AttemptBudget(max_attempts=max_attempts, deadline=deadline)

    result = sol
    changes_made = False
    can_move = True
    cross_impr = intra_impr = exch_impr = two_opt_impr = 0
    intra_or_opt_impr = or_opt_impr = relocate_impr = 0
    cross_gain = intra_gain = exch_gain = two_opt_gain = 0.0
    intra_or_opt_gain = or_opt_gain = relocate_gain = 0.0
    while can_move:
        try:
            done, result, gain = apply_operator(result, cross_suffix, with_last=True, budget=budget)
            if done:
                changes_made = True
                cross_impr += 1
                cross_gain += gain
            else:
                done, result, gain = intra_relocate(result, budget)
                if done:
                    changes_made = True
                    intra_impr += 1
                    intra_gain += gain
                else:
                    done, result, gain = intra_two_opt(result, budget)
                    if done:
                        changes_made = True
                        two_opt_impr += 1
                        two_opt_gain += gain
                    else:
                        done, result, gain = intra_or_opt(result, budget)
                        if done:
                            changes_made = True
                            intra_or_opt_impr += 1
                            intra_or_opt_gain += gain
                        else:
                            done, result, gain = apply_operator(result, exchange_suffix, with_last=False, budget=budget)
                            if done:
                                changes_made = True
                                exch_impr += 1
                                exch_gain += gain
                            else:
                                done, result, gain = apply_relocate(result, budget)
                                if done:
                                    changes_made = True
                                    relocate_impr += 1
                                    relocate_gain += gain
                                else:
                                    done, result, gain = apply_or_opt(result, budget)
                                    if done:
                                        changes_made = True
                                        or_opt_impr += 1
                                        or_opt_gain += gain
                                    else:
                                        can_move = False
        except LimitReached:
            break
    return changes_made, result, LSStats(
        n_attempts=budget.n_attempts,
        cross_improvements=cross_impr,
        intra_relocate_improvements=intra_impr,
        exchange_improvements=exch_impr,
        two_opt_improvements=two_opt_impr,
        intra_or_opt_improvements=intra_or_opt_impr,
        or_opt_improvements=or_opt_impr,
        relocate_improvements=relocate_impr,
        cross_gain=cross_gain,
        intra_relocate_gain=intra_gain,
        exchange_gain=exch_gain,
        two_opt_gain=two_opt_gain,
        intra_or_opt_gain=intra_or_opt_gain,
        or_opt_gain=or_opt_gain,
        relocate_gain=relocate_gain,
    )
