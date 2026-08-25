from __future__ import annotations


def memory_cliff(early_pp: float | None, final_pp: float | None, threshold: float = 0.65) -> bool:
    if not early_pp or not final_pp or early_pp <= 0:
        return False
    return final_pp / early_pp < threshold


def severe_regression(value: float, best: float, threshold: float = 0.50) -> bool:
    return best > 0 and value < best * threshold


def phase_should_stop(scores: list[float], min_improvement: float = 0.03, consecutive: int = 2) -> bool:
    if len(scores) < consecutive + 1:
        return False
    best_before = max(scores[:-(consecutive)])
    tail_best = max(scores[-consecutive:])
    return tail_best <= best_before * (1 + min_improvement)
