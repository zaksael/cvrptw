import time
from dataclasses import dataclass


class LimitReached(Exception):
    pass


@dataclass
class AttemptBudget:
    """deadline is an absolute time.perf_counter() timestamp, not a duration.

    The deadline is checked on the first tick and every 64th after that —
    reading the clock on every one of the millions of ticks per run costs
    more than the up-to-63-tick (microseconds) overshoot it prevents.
    """
    max_attempts: int
    deadline: float | None = None
    n_attempts: int = 0

    def tick(self) -> None:
        self.n_attempts += 1
        if self.n_attempts >= self.max_attempts:
            raise LimitReached
        if (self.deadline is not None and self.n_attempts & 63 == 1
                and time.perf_counter() >= self.deadline):
            raise LimitReached
