import math
from typing import Iterator
import itertools


constant_scheduler = itertools.repeat

def monotonic_annealing_scheduler(
        n_steps: int, start: float = 0.0, stop: float = 1.0, ratio: float = 0.5
    ) -> Iterator[float]:
    yield from cyclical_annealing_scheduler(
            n_steps=n_steps, start=start, stop=stop, n_cycle=1, ratio=ratio)

def cyclical_annealing_scheduler(
        n_steps: int, start: float = 0.0, stop: float = 1.0, n_cycle: int = 4, 
        ratio: float = 0.5) -> Iterator[float]:
    """Reference: https://github.com/haofuml/cyclical_annealing"""
    
    period = math.ceil(n_steps / n_cycle)
    step = (stop - start) / (period * ratio) # linear schedule

    for c in range(n_cycle):
        v = start
        for i in range(period):
            yield v
            v = min(v + step, stop)

if __name__ == "__main__":
    monotonic_schedule = list(monotonic_annealing_scheduler(100))
    cyclical_schedule = list(cyclical_annealing_scheduler(100))
