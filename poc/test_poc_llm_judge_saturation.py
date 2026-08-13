"""Tests for the offline PoC — the properties its own docstrings claim.

E2E review round 2 (2026-08-13): the PoC had zero test coverage. These tests
assert exactly the properties the script's docstrings claim as fixes over an
earlier, buggy version (saturation knob inert/inverted, identifiable flag
overridden by the raw heuristic, regime classifier misfiling the ideal
judge) — empirically, not by re-reading the code.
"""

from __future__ import annotations

import math

import pytest

from poc_llm_judge_saturation import (
    SyntheticJudge,
    classify_regime,
    cost_illustration,
    crossed_design,
    decompose,
    demo_judges,
    exact_binomial_onesided_pvalue,
    min_k_for_gate_power,
    run_cell,
)


# --------------------------------------------------------------------------- #
# Saturation knob: must be monotonic, in every regime, not just the obvious one
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label_bias,base_pref", [
    (0.0, 0.55),   # no token prior
    (0.0, 0.50),   # exact tie -- the case the old per-sample collapse inverted
    (0.4, 0.55),   # a real label prior present
    (-0.3, 0.5),   # negative label prior
])
def test_saturation_knob_is_monotonic_in_top_prob(label_bias, base_pref):
    """Turning saturation up must never turn a cell's top-token probability
    down, in ANY regime -- this is exactly the bug (`sample()`-level collapse
    re-randomised per draw) the docstring says used to invert the knob at
    label_bias != 0 and leave it inert at label_bias == 0."""
    saturations = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    top_probs = []
    for s in saturations:
        judge = SyntheticJudge(
            name="probe", base_preference=base_pref, label_bias=label_bias,
            position_bias=0.0, saturation=s, seed=7,
        )
        # Average top_prob over many cells to wash out per-cell RNG noise --
        # this test is about the population trend, not one sample path.
        probs = []
        for trial_seed in range(30):
            j = SyntheticJudge(
                name="probe", base_preference=base_pref, label_bias=label_bias,
                position_bias=0.0, saturation=s, seed=trial_seed,
            )
            verdicts = j.sample_cell(True, True, k=200)
            n_a = sum(1 for v in verdicts if v == "A")
            probs.append(max(n_a, 200 - n_a) / 200)
        top_probs.append(sum(probs) / len(probs))

    for i in range(len(top_probs) - 1):
        assert top_probs[i] <= top_probs[i + 1] + 0.02, (
            f"non-monotonic at label_bias={label_bias}, base_pref={base_pref}: "
            f"saturation {saturations[i]}->{saturations[i+1]} gave "
            f"top_prob {top_probs[i]:.3f}->{top_probs[i+1]:.3f}"
        )
    # And it must actually move, not just fail to decrease.
    assert top_probs[-1] > top_probs[0] + 0.05, (
        f"knob is inert at label_bias={label_bias}, base_pref={base_pref}: "
        f"{top_probs[0]:.3f} -> {top_probs[-1]:.3f}"
    )


def test_saturation_one_gives_a_constant_cell():
    judge = SyntheticJudge(name="probe", base_preference=0.6, label_bias=0.0,
                            position_bias=0.0, saturation=1.0, seed=3)
    verdicts = judge.sample_cell(True, True, k=50)
    assert len(set(verdicts)) == 1, "saturation=1.0 must collapse to a single verdict"


def test_marginal_p_emit_a_matches_closed_form_via_large_sample():
    """The closed-form marginal_p_emit_A should match what sample() actually
    draws, at saturation=0 (no collapse -- pure marginal sampling)."""
    judge = SyntheticJudge(name="probe", base_preference=0.55, label_bias=0.3,
                            position_bias=0.1, saturation=0.0, seed=11)
    p_expected = judge.marginal_p_emit_A(True, True)
    n = 20000
    n_a = sum(1 for _ in range(n) if judge.sample(True, True) == "A")
    p_observed = n_a / n
    assert abs(p_observed - p_expected) < 0.02, (
        f"expected {p_expected:.3f}, observed {p_observed:.3f} over {n} draws"
    )


# --------------------------------------------------------------------------- #
# Dead-branch removal: lambda<0 must still mirror lambda>0 exactly
# --------------------------------------------------------------------------- #

def test_negative_label_bias_mirrors_positive():
    """The removed `if lam < 0` branch restated the same formula (differing
    only by a dead `+ 0.0`) -- confirm the single expression that replaced it
    is still exactly symmetric under label_bias -> -label_bias with A/B
    swapped."""
    pos = SyntheticJudge(name="pos", base_preference=0.5, label_bias=0.6,
                          position_bias=0.0, saturation=0.0, seed=1)
    neg = SyntheticJudge(name="neg", base_preference=0.5, label_bias=-0.6,
                          position_bias=0.0, saturation=0.0, seed=1)
    p_pos_a = pos.marginal_p_emit_A(True, True)
    p_neg_a = neg.marginal_p_emit_A(True, True)
    assert p_pos_a == pytest.approx(1.0 - p_neg_a, abs=1e-9)


# --------------------------------------------------------------------------- #
# `identifiable` must be driven by the calibrated gate only, not the raw
# top_prob>=0.9 heuristic sitting next to it
# --------------------------------------------------------------------------- #

def test_identifiable_tracks_the_calibrated_gate_not_the_raw_heuristic():
    """Construct a judge dispersed enough that top_prob can cross 0.90 by
    chance in some cells without the EXACT gate firing (k below where 0.90 is
    itself significant at alpha=0.05 for that specific count) -- confirm
    `identifiable` follows `n_saturated_cells` (the calibrated gate), not
    `n_near_saturated` (the raw >=0.90 heuristic)."""
    judge = SyntheticJudge(name="probe", base_preference=0.85, label_bias=0.0,
                            position_bias=0.0, saturation=0.3, seed=5)
    cells = crossed_design(judge, k=12)  # k=12 is below the k=29 gate-power floor
    stats = decompose(cells)
    assert stats["n_saturated_cells"] == 0, (
        "k=12 is below the exact gate's power floor -- it cannot fire, "
        "so this is a meaningless test unless n_saturated_cells is 0"
    )
    assert stats["identifiable"] is True, (
        "identifiable must be True whenever the calibrated gate hasn't fired, "
        "regardless of whether any cell's raw top_prob happened to cross 0.90"
    )


def test_identifiable_false_exactly_when_gate_fires():
    judge = SyntheticJudge(name="probe", base_preference=0.5, label_bias=0.9,
                            position_bias=0.0, saturation=0.99, seed=2)
    cells = crossed_design(judge, k=40)
    stats = decompose(cells)
    assert stats["n_saturated_cells"] > 0
    assert stats["identifiable"] is False


# --------------------------------------------------------------------------- #
# Regime classifier: exhaustive, mutually exclusive, and the ideal judge
# must land in "dispersed_clean" -- not the "low-signal" catch-all the old
# two-sided dead band used to swallow it into.
# --------------------------------------------------------------------------- #

def test_regime_classifier_is_exhaustive_and_mutually_exclusive():
    from poc_llm_judge_saturation import REGIME_KEYS
    for judge in demo_judges():
        cells = crossed_design(judge, k=40)
        stats = decompose(cells)
        regime, lines = classify_regime(stats)
        assert regime in REGIME_KEYS, f"{judge.name} -> unknown regime {regime!r}"
        assert lines, f"{judge.name} -> {regime} has no advice lines"


def test_ideal_dispersed_clean_judge_is_not_swallowed_by_a_dead_band():
    """The judge with a mild, genuine content preference, no label prior, no
    position bias, and low saturation is the textbook 'nothing to correct
    for' case. The old two-sided 0.10/0.12 threshold pair left a gap that
    filed borderline judges like this one under a 'low-signal, inspect
    manually' catch-all instead of the clean regime."""
    judge = SyntheticJudge(name="ideal", base_preference=0.55, label_bias=0.0,
                            position_bias=0.0, saturation=0.10, seed=62)
    cells = crossed_design(judge, k=40)
    stats = decompose(cells)
    regime, _ = classify_regime(stats)
    assert regime == "dispersed_clean", (
        f"the textbook clean judge landed in {regime!r} instead"
    )


def test_saturated_content_driven_regime_is_reachable():
    """Confirm at least one demo judge actually exercises the 'benign
    saturation, verdict tracks content' branch -- added because the classifier
    rewrite added a 4th judge specifically to cover this branch."""
    from poc_llm_judge_saturation import REGIME_KEYS
    assert "saturated_content_driven" in REGIME_KEYS
    regimes_hit = set()
    for judge in demo_judges():
        cells = crossed_design(judge, k=40)
        stats = decompose(cells)
        regime, _ = classify_regime(stats)
        regimes_hit.add(regime)
    assert "saturated_content_driven" in regimes_hit, (
        "no demo judge exercises the benign-saturation branch"
    )


# --------------------------------------------------------------------------- #
# Cost model: matches the real script's documented ~2x, and refuses a
# sub-floor pilot rather than reporting a fictional saving
# --------------------------------------------------------------------------- #

def test_crossed_design_costs_exactly_2x_naive():
    costs = cost_illustration(n_items=500, k=40)
    assert costs["crossing_multiplier_full"] == pytest.approx(2.0)


def test_gated_pilot_below_power_floor_is_refused():
    floor = min_k_for_gate_power(0.9, 0.05)
    assert floor == 29
    with pytest.raises(ValueError, match="power floor"):
        cost_illustration(n_items=500, k=40, k_pilot=floor - 1)


def test_gated_pilot_at_floor_is_accepted_and_cheaper_than_full():
    costs = cost_illustration(n_items=500, k=40, k_pilot=None)
    assert costs["k_pilot"] == costs["gate_power_floor_k"]
    assert costs["calls_crossed_gated"] < costs["calls_crossed_full"]
    assert costs["crossing_multiplier_gated"] < costs["crossing_multiplier_full"]


# --------------------------------------------------------------------------- #
# Gate power floor + exact p-value: cross-check against the real script's
# formula independently (not by importing it -- this file has no dependency
# on the private repo)
# --------------------------------------------------------------------------- #

def test_gate_power_floor_matches_independent_computation():
    for sat, alpha, expected in [(0.9, 0.05, 29), (0.85, 0.05, 19), (0.9, 0.01, 44)]:
        assert min_k_for_gate_power(sat, alpha) == expected == math.ceil(
            math.log(alpha) / math.log(sat)
        )


def test_exact_pvalue_monotonic_in_successes():
    p1 = exact_binomial_onesided_pvalue(28, 40, p0=0.9)
    p2 = exact_binomial_onesided_pvalue(40, 40, p0=0.9)
    assert p2 < p1
    assert p2 < 0.05
    assert p1 > 0.05
