import math


def rate(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson 95% interval. (0.0, 0.0, 0.0) when total is 0."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))
