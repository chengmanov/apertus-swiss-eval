"""Wilson score intervals -- reported next to every headline number."""

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, low, high) as fractions. n == 0 -> (0, 0, 1)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def fmt_pct(successes: int, n: int) -> str:
    p, lo, hi = wilson(successes, n)
    return f"{100*p:.1f}% [{100*lo:.0f}-{100*hi:.0f}]"
