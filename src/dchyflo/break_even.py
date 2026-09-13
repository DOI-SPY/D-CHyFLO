"""Find deterministic conditions where 30-year cumulative net carbon reaches zero."""

from __future__ import annotations

from collections.abc import Callable


def bisect_zero(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    iterations: int = 70,
) -> float:
    """Find a single-factor break-even boundary by bisection."""
    low, high = float(lower), float(upper)
    f_low, f_high = function(low), function(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        raise ValueError("search bounds do not bracket a zero")
    for _ in range(iterations):
        middle = (low + high) / 2
        f_middle = function(middle)
        if f_low * f_middle <= 0:
            high = middle
            f_high = f_middle
        else:
            low = middle
            f_low = f_middle
    return (low + high) / 2
