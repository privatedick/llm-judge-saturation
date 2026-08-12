"""Tests for qq_cbd: QQ equality, order-sensitivity, cyclic-system CbD.

Validated against distributions with KNOWN structure, so correctness does not
depend on any LLM API. Run: `pytest python/test_qq_cbd.py`
"""

from __future__ import annotations

import numpy as np
import pytest

from qq_cbd import (
    cbd_cyclic,
    interference,
    order_effect,
    qq_cbd_discriminant,
    qq_residual,
    s_odd,
)


def _independent_joint(p_first_yes: float, p_second_yes: float) -> np.ndarray:
    """Joint indexed [first][second] for two independent binary questions."""
    p = np.zeros((2, 2))
    for f in (0, 1):
        for s in (0, 1):
            pf = p_first_yes if f == 1 else 1 - p_first_yes
            ps = p_second_yes if s == 1 else 1 - p_second_yes
            p[f, s] = pf * ps
    return p


# --------------------------------------------------------------------------- #
# QQ equality
# --------------------------------------------------------------------------- #

def test_qq_residual_zero_for_independent_process():
    # Independent answers cannot produce a QQ imbalance.
    p_ab = _independent_joint(0.7, 0.3)   # [A][B]
    p_ba = _independent_joint(0.3, 0.7)   # [B][A]
    assert abs(qq_residual(p_ab, p_ba)) < 1e-12


def test_qq_residual_recovers_known_imbalance():
    p_ab = np.array([[0.1, 0.1], [0.4, 0.4]])      # off-diagonal sum 0.5
    p_ba = np.array([[0.45, 0.05], [0.05, 0.45]])  # off-diagonal sum 0.1
    assert qq_residual(p_ab, p_ba) == pytest.approx(0.4)


def test_qq_residual_antisymmetric():
    p_ab = np.array([[0.1, 0.1], [0.4, 0.4]])
    p_ba = np.array([[0.45, 0.05], [0.05, 0.45]])
    assert qq_residual(p_ab, p_ba) == pytest.approx(-qq_residual(p_ba, p_ab))


def test_joint_validation_rejects_malformed_input():
    with pytest.raises(ValueError):
        qq_residual(np.zeros((2, 2)), _independent_joint(0.5, 0.5))   # sums to 0
    with pytest.raises(ValueError):
        qq_residual(np.ones((3, 3)) / 9, _independent_joint(0.5, 0.5))  # wrong shape


# --------------------------------------------------------------------------- #
# Order sensitivity
# --------------------------------------------------------------------------- #

def test_order_effect_zero_when_orders_agree():
    p_ab = _independent_joint(0.6, 0.4)
    p_ba = _independent_joint(0.4, 0.6)   # same (A,B) content, transposed index
    oe = order_effect(p_ab, p_ba)
    assert oe["total_variation"] < 1e-12
    assert oe["order_sensitive"] is False
    assert abs(oe["a_yes_shift"]) < 1e-12


def test_order_effect_detects_marginal_shift():
    # A-yes is 0.9 in one order and 0.1 in the other.
    p_ab = _independent_joint(0.9, 0.5)
    p_ba = _independent_joint(0.5, 0.1)
    oe = order_effect(p_ab, p_ba)
    assert oe["order_sensitive"] is True
    assert oe["a_yes_shift"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Interference
# --------------------------------------------------------------------------- #

def test_interference_zero_iff_total_probability_holds():
    p_b_given_a = (0.2, 0.8)
    classical = 0.5 * 0.2 + 0.5 * 0.8
    assert interference(p_b_given_a, 0.5, classical) == pytest.approx(0.0)
    assert interference(p_b_given_a, 0.5, classical + 0.15) == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
# Cyclic-system Contextuality-by-Default
# --------------------------------------------------------------------------- #

def test_s_odd_known_values():
    assert s_odd([1, 1, 1, 1]) == 2.0     # must flip an odd number of signs
    assert s_odd([1, 1, 1, -1]) == 4.0    # flip the -1 -> all positive


def test_s_odd_empty_raises():
    """s_odd([]) used to silently return -inf (n=0 has no odd-parity sign
    pattern at all, so the max-over-nothing default never got overwritten)
    instead of raising on degenerate input."""
    with pytest.raises(ValueError):
        s_odd([])


def test_order_effect_o_ss_and_total_variation_can_maximally_diverge():
    """TV of the two joints and O_SS (the marginal-shift sum) are not just
    occasionally different -- they can hit OPPOSITE extremes on the same
    input. Applying Kang's |q|<=O_SS with TV instead of o_ss gets the wrong
    verdict here, not a slightly-off one."""
    p_ab = np.array([[0.5, 0.0], [0.0, 0.5]])   # diagonal joint, [A][B]
    p_ba = np.array([[0.0, 0.5], [0.5, 0.0]])   # anti-diagonal joint, [B][A]
    oe = order_effect(p_ab, p_ba)
    assert oe["total_variation"] == pytest.approx(1.0)
    assert oe["o_ss"] == pytest.approx(0.0)


def test_pr_box_is_contextual_and_hits_algebraic_max():
    res = cbd_cyclic([1, 1, 1, -1], [0, 0, 0, 0])
    assert res.contextual is True
    assert res.cnt == pytest.approx(2.0)   # CNT = 4 - 2 - 0


def test_local_correlations_are_not_contextual():
    res = cbd_cyclic([0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0])
    assert res.contextual is False
    assert res.cnt < 0


def test_direct_influence_can_explain_away_correlation():
    # Same correlations as the PR box, but with large marginal mismatch:
    # not TRUE contextuality (this is the Dzhafarov-Kujala correction).
    res = cbd_cyclic([1, 1, 1, -1], [1.0, 1.0, 0, 0])
    assert res.contextual is False


def test_rank_mismatch_is_rejected():
    with pytest.raises(ValueError, match="marginal mismatches"):
        cbd_cyclic([1, 1, 1, -1], [0, 0])         # 4 bunches, 2 mismatches
    with pytest.raises(ValueError):
        cbd_cyclic([1.5, 1, 1, -1], [0, 0, 0, 0])  # correlation out of [-1,1]


# --------------------------------------------------------------------------- #
# Discriminant
# --------------------------------------------------------------------------- #

def test_discriminant_separates_the_three_signatures():
    assert qq_cbd_discriminant(q=0.0, o_ss=0.0, cnt=0.0) == "order-insensitive"
    assert qq_cbd_discriminant(q=0.0, o_ss=0.3, cnt=0.0) == "quantum-consistent"
    assert qq_cbd_discriminant(q=0.4, o_ss=0.3, cnt=0.0) == "qq-imbalanced"
    assert qq_cbd_discriminant(q=0.0, o_ss=0.3, cnt=0.5) == "quantum-consistent +contextual"
