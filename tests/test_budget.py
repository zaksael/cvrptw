import time

import pytest

from cvrptw.search.budget import AttemptBudget, LimitReached


def test_attempt_budget_does_not_raise_under_limit():
    budget = AttemptBudget(max_attempts=5)
    for _ in range(4):
        budget.tick()
    assert budget.n_attempts == 4


def test_attempt_budget_raises_at_max_attempts():
    budget = AttemptBudget(max_attempts=3)
    budget.tick()
    budget.tick()
    with pytest.raises(LimitReached):
        budget.tick()
    assert budget.n_attempts == 3


def test_attempt_budget_zero_max_attempts_raises_immediately():
    budget = AttemptBudget(max_attempts=0)
    with pytest.raises(LimitReached):
        budget.tick()


def test_attempt_budget_raises_at_deadline():
    budget = AttemptBudget(max_attempts=1_000_000, deadline=time.perf_counter() - 1.0)
    with pytest.raises(LimitReached):
        budget.tick()
    assert budget.n_attempts == 1
