import time
from dataclasses import dataclass


class LimitReached(Exception):
    pass


@dataclass
class AttemptBudget:
    """deadline is an absolute time.perf_counter() timestamp, not a duration."""
    max_attempts: int
    deadline: float | None = None
    n_attempts: int = 0

    def tick(self) -> None:
        self.n_attempts += 1
        if self.n_attempts >= self.max_attempts:
            raise LimitReached
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise LimitReached
