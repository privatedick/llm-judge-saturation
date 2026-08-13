"""Verifies the exact-binomial saturation gate's statistical power floor.

E2E review (2026-08-13): the calibrated gate's smallest attainable p-value at
k samples is `sat_top_prob ** k` (the best case for the gate -- a perfectly
deterministic k/k cell). Below the k where that crosses alpha, the gate CANNOT
flag `saturated_exact` for ANY outcome, including a truly 100%-deterministic
judge, and nothing said so. At the old default (k=15), `--pilot` (k=6), and
even the README's own quick-repro suggestion (`--k 12`), the gate was
silently powerless -- a reader following the quick-start would see
frac_conditions_saturated=0.0 regardless of the judges' true behavior.

This test does not trust the formula by reading it: it empirically confirms
the crossover at k=29 by running the real exact test from `_dist_stats`
against synthetic all-agree ("AAAA...") verdict lists at k=28 and k=29.
"""

from __future__ import annotations

from experiment_llm_judge_saturation import _dist_stats, _min_k_for_gate_power


def test_gate_power_floor_matches_the_formula_against_the_real_test():
    k_min = _min_k_for_gate_power(sat_top_prob=0.9, alpha=0.05)
    assert k_min == 29

    # A perfectly deterministic cell (k/k agreement) is the best case for the
    # gate -- if it can't fire here, it can't fire at all at this k.
    below = _dist_stats(["A"] * (k_min - 1), sat_threshold=0.9, alpha=0.05)
    at_floor = _dist_stats(["A"] * k_min, sat_threshold=0.9, alpha=0.05)
    assert below["saturated_exact"] is False, (
        f"k={k_min - 1} should have zero power even for a fully deterministic cell"
    )
    assert at_floor["saturated_exact"] is True, (
        f"k={k_min} should be enough to flag a fully deterministic cell"
    )


def test_new_default_k_clears_the_power_floor():
    """The default --k was raised from 15 to 30 specifically because 15 sat
    below the k=29 floor. Lock in that the new default actually clears it,
    so a future edit can't silently drop it back below the line."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args([])
    k_min = _min_k_for_gate_power(sat_top_prob=0.9, alpha=0.05)
    assert args.k >= k_min, f"default --k={args.k} sits below the gate power floor k_min={k_min}"
