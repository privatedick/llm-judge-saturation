"""E1/E2 measurement library: QQ equality, order-sensitivity, and cyclic-system
Contextuality-by-Default (CbD).

All functions are pure and operate on already-estimated distributions, so they
are unit-testable against synthetic distributions with known structure — no LLM
API call is required to validate the math. The population layer that *produces*
these distributions is the caller's concern.

Conventions
-----------
A two-question sequential design has two orders. For a fixed order (say A then
B), the order-conditioned joint is a 2x2 array ``p`` indexed ``p[first][second]``
where index 1 == "yes", 0 == "no". Rows/cols therefore mean:

    p[1][0] = P(first answered yes, second answered no)

References
----------
- Wang & Busemeyer (2013) *Topics in Cognitive Science*; Wang, Solloway,
  Shiffrin & Busemeyer (2014) *PNAS* — the QQ equality.
- Kujala, Dzhafarov & Larsson (2015) *PRL*; Dzhafarov & Kujala — CbD for
  cyclic systems. Basieva et al. (2019) *JEP:General* — contextuality in
  human decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

Array2 = np.ndarray  # a normalised 2x2 joint, p[first][second], index 1==yes


def _validate_joint(p: Array2, tol: float = 1e-9) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.shape != (2, 2):
        raise ValueError(f"joint must be 2x2, got {a.shape}")
    if (a < -tol).any():
        raise ValueError("joint has negative probabilities")
    s = a.sum()
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"joint must sum to 1, got {s}")
    return a


def qq_residual(p_ab: Array2, p_ba: Array2) -> float:
    """Wang-Busemeyer QQ equality residual q.

    q = [P_AB(A=yes,B=no) + P_AB(A=no,B=yes)]
        - [P_BA(B=yes,A=no) + P_BA(B=no,A=yes)]

    The projective quantum question-order model predicts q == 0 *exactly*,
    parameter-free, even when the order effect is large. A generic classical
    process is under no such constraint. ``p_ab`` is the A-then-B joint indexed
    [A][B]; ``p_ba`` is the B-then-A joint indexed [B][A].
    """
    a = _validate_joint(p_ab)
    b = _validate_joint(p_ba)
    ab = a[1, 0] + a[0, 1]
    ba = b[1, 0] + b[0, 1]
    return float(ab - ba)


def order_effect(p_ab: Array2, p_ba: Array2) -> dict:
    """Magnitude of the order effect, independent of whether QQ holds.

    Returns the marginal shift of each question's "yes" rate between the two
    orders, their sum (``o_ss``), and the total variation between the two
    order-conditioned joints (mapped onto a common (A,B) index).

    ``o_ss`` = ``|a_yes_shift| + |b_yes_shift|`` is Kang's (2026) ``O_SS``
    order-sensitivity score — the quantity the CbD translation actually pairs
    with ``|q|`` in ``|q_QQ| <= O_SS`` — verified against the paper (round-2
    review, 2026-08-11). An earlier version of this docstring named
    ``total_variation`` as O_SS instead; TV of the two joints and the sum of
    marginal shifts are NOT interchangeable (TV can exceed the marginal sum),
    so anything applying Kang's inequality must use ``o_ss``, not ``tv``.
    ``total_variation`` is kept as a separate, real statistic in its own right
    (e.g. for a direct quantum-vs-classical joint-distance comparison) — just
    not the one the discriminant below wants.
    """
    a = _validate_joint(p_ab)  # [A][B]
    b = _validate_joint(p_ba)  # [B][A]
    # A-yes marginal: order AB -> sum over B of a[1][*]; order BA -> sum over B of b[*][1]
    a_yes_ab = a[1, :].sum()
    a_yes_ba = b[:, 1].sum()
    b_yes_ab = a[:, 1].sum()
    b_yes_ba = b[1, :].sum()
    a_yes_shift = float(a_yes_ab - a_yes_ba)
    b_yes_shift = float(b_yes_ab - b_yes_ba)
    # Re-index the BA joint onto (A,B): b is [B][A] -> ab_from_ba[A][B] = b[B][A]
    ab_from_ba = b.T
    tv = 0.5 * float(np.abs(a - ab_from_ba).sum())
    return {
        "a_yes_shift": a_yes_shift,
        "b_yes_shift": b_yes_shift,
        "o_ss": float(abs(a_yes_shift) + abs(b_yes_shift)),
        "total_variation": tv,
        "order_sensitive": tv > 1e-6,
    }


def interference(p_b_yes_given_a: tuple[float, float], p_a: float,
                 p_b_yes_direct: float) -> float:
    """Interference term for a two-stage design (categorisation-then-decision).

    Classical law of total probability: P(B=yes) = sum_a P(A=a) P(B=yes|A=a).
    Quantum interference = P(B=yes) measured *without* the intervening A
    (``p_b_yes_direct``) minus the classical prediction. Non-zero interference
    is the formal signature behind the disjunction effect (mechanism #3).

    ``p_b_yes_given_a`` = (P(B=yes|A=no), P(B=yes|A=yes)); ``p_a`` = P(A=yes).
    """
    pa = float(p_a)
    classical = (1 - pa) * p_b_yes_given_a[0] + pa * p_b_yes_given_a[1]
    return float(p_b_yes_direct - classical)


# --------------------------------------------------------------------------- #
# Cyclic-system Contextuality-by-Default (E2)
# --------------------------------------------------------------------------- #

def s_odd(values: list[float] | np.ndarray) -> float:
    """max over sign patterns with an ODD number of -1 of sum(eps_i * x_i).

    Brute force over 2**n patterns (n<=~10 in practice; CHSH is n=4). Exact.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    best = -np.inf
    for signs in product((1.0, -1.0), repeat=n):
        if signs.count(-1.0) % 2 == 1:
            best = max(best, float(np.dot(signs, x)))
    return best


@dataclass
class CbDResult:
    cnt: float                 # CbD contextuality functional; >0 == contextual
    contextual: bool
    s_odd: float
    direct_influence: float    # sum of marginal-mismatch (ODΔ) terms
    n: int


def cbd_cyclic(bunch_expectations: list[float],
               marginal_mismatches: list[float],
               tol: float = 1e-9) -> CbDResult:
    """Contextuality-by-Default criterion for a cyclic system of rank n.

    Kujala-Dzhafarov-Larsson (2015). A cyclic system of rank n has n content
    variables, each measured in exactly 2 of the n contexts.

      CNT = s_odd({<R_i R_{i+1}>}) - (n - 2) - sum_i ODΔ_i

    where <R_i R_{i+1}> are the n within-context (bunch) product expectations
    in [-1, 1], and ODΔ_i = |<R_i>^{c} - <R_i>^{c'}| is the "direct influence"
    (marginal-mismatch) of content i across its two contexts. The system is
    *contextual* iff CNT > 0 — i.e. only when the odd-cycle correlation exceeds
    what direct influences alone can account for. CHSH is the rank-4 case; a
    Tsirelson-bound PR-ish box gives CNT>0, a classical/local system CNT<=0.
    """
    corr = np.asarray(bunch_expectations, dtype=float)
    n = len(corr)
    if n < 2:
        raise ValueError("cyclic system needs rank n>=2")
    if (np.abs(corr) > 1 + 1e-9).any():
        raise ValueError("bunch expectations must lie in [-1, 1]")
    if len(marginal_mismatches) != n:
        raise ValueError(
            f"a rank-{n} cyclic system needs {n} marginal mismatches, "
            f"got {len(marginal_mismatches)}"
        )
    di = float(np.sum(np.abs(marginal_mismatches)))
    so = s_odd(corr)
    cnt = so - (n - 2) - di
    return CbDResult(cnt=float(cnt), contextual=bool(cnt > tol),
                     s_odd=float(so), direct_influence=di, n=n)


def qq_cbd_discriminant(q: float, o_ss: float, cnt: float) -> str:
    """Kang (2026) two-layer discriminant: separate the three signatures rather
    than treat them as interchangeable.

    - order-insensitive           : |O_SS| ~ 0            -> "order-insensitive"
    - order effect, QQ-consistent  : |q| ~ 0, O_SS > 0     -> "quantum-consistent"
    - order effect, QQ-violating   : |q| > 0               -> "qq-imbalanced"
    - plus residual contextuality  : cnt > 0               -> "+contextual"
    """
    parts = []
    if abs(o_ss) < 1e-6:
        parts.append("order-insensitive")
    elif abs(q) < 1e-3:
        parts.append("quantum-consistent")
    else:
        parts.append("qq-imbalanced")
    if cnt > 1e-9:
        parts.append("+contextual")
    return " ".join(parts)
