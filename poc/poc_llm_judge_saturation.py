#!/usr/bin/env python3
"""
PoC: LLM-Judge Saturation Gate & Position×Label Decomposition
=============================================================

Demonstrates the core measurement hygiene from
https://github.com/privatedick/llm-judge-saturation

What this PoC shows
-------------------
1. A fully crossed content-POSITION × LABEL-ASSIGNMENT design.
2. A calibrated saturation / identifiability gate (exact one-sided binomial),
   including the gate's own power floor.
3. Decomposition of apparent "order effect" into:
   - pure content-position effect
   - pure label/token preference
   - how strongly the verdict tracks content at all
4. Cost implication: crossing position with label BUYS identifiability and
   PAYS ~2x for it; the gate claws part of that back, and when the gate
   fires, additional samples buy almost no identifying information.

No external LLM API is required. Synthetic judges with controllable
saturation and bias parameters are used so the PoC is fully offline and
reproducible.

HONEST SCOPE (read before citing the cost numbers): the "gated" row of the
cost table models an ADAPTIVE early-stopping sampling strategy (pilot at the
gate's power floor, escalate only where the gate has not already fired) that
this repository's actual measurement script
(../python/experiment_llm_judge_saturation.py) does NOT implement -- that
script always samples the full k per condition. The monthly figure is what
such a strategy COULD save if built, not a result this repo delivers today.
Don't cite it as a measured outcome. The position×label decomposition and
the calibrated gate itself, on the other hand, are exactly what the real
measurement script does.

CONVENTIONS vs the real measurement script (the differences are deliberate;
they are listed here so a reader can line the two outputs up)
-------------------------------------------------------------------------
* `label_bias` is the RAW P(emit "A") -- the same convention as the real
  script's `label_bias`, where 0.5 means "no token prior". A centred
  companion `label_bias_centered` (= label_bias - 0.5) is also reported
  because it reads better next to a signed position effect; the real script
  has no such field.
* `position_effect` here is SIGNED (positive = content_1 is favoured when it
  is presented first). The real script's `order_effect` is the ABSOLUTE
  value of the same difference-in-differences, so compare
  |position_effect| against `order_effect`.
* `position_bias` is a per-side parameter (±position_bias around
  base_preference), so the measured `position_effect` comes out close to 2x
  the parameter value -- the crossed design's difference-in-differences, not
  a bug. That 2x relation holds only near `label_bias == 0`: an independent
  token prior dilutes the content signal and compresses the measured
  position effect (at label_bias=0.5 with heavy saturation the ratio falls
  below 0.2). Don't read it as a calibration constant.
* `saturation` here is a pure DISPERSION knob: it concentrates a cell's
  verdicts onto a single token WITHOUT shifting the judge's mean preference.
  In a real judge, saturation is emergent rather than a dial.

Run:
    python poc_llm_judge_saturation.py

Tests:
    python -m pytest poc/ -v
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Minimal exact one-sided binomial test (no scipy dependency)
# H0: p <= p0   vs   H1: p > p0
# Returns the p-value of observing >= k successes in n trials under Bin(n, p0).
# ---------------------------------------------------------------------------


def _binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    # log-space for stability
    log_c = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_c + k * math.log(p) + (n - k) * math.log(1 - p))


def exact_binomial_onesided_pvalue(successes: int, n: int, p0: float = 0.9) -> float:
    """P(X >= successes | X ~ Bin(n, p0)).

    Matches ``scipy.stats.binomtest(..., alternative="greater")`` to ~1e-14 on
    the range this file uses. Two deliberate edge differences: ``n == 0``
    returns 1.0 (scipy raises), and ``p0`` must be strictly inside (0, 1).
    """
    if n == 0:
        return 1.0
    total = 0.0
    for k in range(successes, n + 1):
        total += _binom_pmf(k, n, p0)
    return min(1.0, total)


def min_k_for_gate_power(saturation_threshold: float = 0.9,
                         alpha: float = 0.05) -> int:
    """Smallest k at which the exact one-sided gate CAN fire at all.

    Mirrors ``_min_k_for_gate_power`` in the real measurement script. The
    smallest p-value the gate can attain at k samples is
    ``saturation_threshold ** k`` (a perfectly deterministic k/k cell). Below
    this k the gate returns False for EVERY possible outcome, including a
    genuinely 100%-deterministic judge -- zero power, silently.

    At the defaults (0.9, 0.05) the floor is 29. That is a real constraint on
    the cost model below: an adaptive pilot smaller than the floor could never
    stop early, so any "saving" it reported would be fictional.
    """
    return math.ceil(math.log(alpha) / math.log(saturation_threshold))


# ---------------------------------------------------------------------------
# Synthetic judge
# ---------------------------------------------------------------------------

Verdict = Literal["A", "B"]


@dataclass
class SyntheticJudge:
    """
    Controllable synthetic pairwise judge.

    - base_preference: P(prefer content_1 over content_2) when unbiased.
    - label_bias: extra probability mass pushed toward emitting the token "A"
      (positive) or "B" (negative), independent of content and position.
    - position_bias: extra preference for the content that appears first.
    - saturation: WITHIN-CELL determinism, the analogue of a low sampling
      temperature. The cell has one modal verdict -- the argmax of its
      marginal P(emit "A") -- and the judge emits that mode with probability
      `saturation`, otherwise it re-samples from the marginal. 0.0 = fully
      stochastic, 1.0 = the cell is a constant.

      Saturation is deliberately a property ACROSS the samples of a cell,
      because that is what output saturation means for a real judge: the mode
      is a fixed function of the prompt, not something re-rolled per token.
      Applying the collapse per-sample instead (to a mode re-randomised on
      every call) leaves the knob inert at label_bias = 0 and INVERTS it for
      0 < |label_bias| < 0.5 -- turning it up would make the judge more
      dispersed. See `sample_cell`.
    """
    name: str
    base_preference: float = 0.55   # slight preference for content_1
    label_bias: float = 0.0         # P(say "A") shift
    position_bias: float = 0.0      # preference for first-presented content
    saturation: float = 0.85        # P(emit the cell's latent verdict)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _soft_p_prefer_first(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,  # deliberately unread: content preference is
                                 # label-independent by construction, and that
                                 # is what makes the decomposition valid.
    ) -> float:
        """Probability of preferring the first-presented content."""
        # Start from base preference for content_1
        p_prefer_1 = self.base_preference

        # Position bias: boost preference for whichever content is first
        if content_first_is_1:
            p_prefer_1 += self.position_bias
        else:
            p_prefer_1 -= self.position_bias

        p_prefer_1 = max(0.02, min(0.98, p_prefer_1))

        # Now map to "prefer first"
        if content_first_is_1:
            p_prefer_first = p_prefer_1
        else:
            p_prefer_first = 1.0 - p_prefer_1

        return p_prefer_first

    def marginal_p_emit_A(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,
    ) -> float:
        """Closed-form P(emit "A") for this cell, BEFORE saturation collapse.

        Generative story
        ----------------
        1. Soft content preference (base + position) → a preferred *content*.
        2. Map the preferred content to a letter under the current labelling.
        3. Mix with an independent label prior λ:
             P(emit A | soft) = clamp((1 - |λ|) · 1[soft = A] + max(λ, 0))
           so λ=+0.7 → P(A) ≈ 0.7 even when soft says B, and ≈ 1.0 when soft
           says A. The single expression is already symmetric under A↔B:
           max(λ, 0) is exactly 0.0 for λ < 0, so negative λ degrades
           A-verdicts toward B by exactly the mirror amount. (A separate
           `if λ < 0` branch used to sit here restating that; it differed from
           this expression only by `+ 0.0` and was bit-exactly dead.)

        Marginalising step 3 over step 2 gives the returned probability, which
        is exactly the distribution an unsaturated cell samples from.
        """
        p_prefer_first = self._soft_p_prefer_first(content_first_is_1,
                                                   label_first_is_A)
        p_prefer_content_1 = (
            p_prefer_first if content_first_is_1 else 1.0 - p_prefer_first
        )

        if label_first_is_A:
            letter_for_content_1: Verdict = "A" if content_first_is_1 else "B"
        else:
            letter_for_content_1 = "B" if content_first_is_1 else "A"

        p_soft_A = (
            p_prefer_content_1 if letter_for_content_1 == "A"
            else 1.0 - p_prefer_content_1
        )

        lam = max(-0.99, min(0.99, self.label_bias))
        p_A_if_soft_A = max(0.01, min(0.99, (1.0 - abs(lam)) + max(lam, 0.0)))
        p_A_if_soft_B = max(0.01, min(0.99, max(lam, 0.0)))
        return p_soft_A * p_A_if_soft_A + (1.0 - p_soft_A) * p_A_if_soft_B

    def sample(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,
    ) -> Verdict:
        """One verdict drawn from this cell's marginal preference.

        Saturation is NOT applied here -- it is a property across the samples
        of a cell, not of a single draw. Use `sample_cell` to run a cell.
        """
        p = self.marginal_p_emit_A(content_first_is_1, label_first_is_A)
        return "A" if self._rng.random() < p else "B"

    def cell_mode(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,
    ) -> Verdict:
        """The verdict this cell collapses onto under saturation.

        The argmax of the cell's marginal -- a deterministic function of the
        prompt, exactly like a real judge's modal token at low temperature.
        An EXACT tie (marginal == 0.5, reachable with a tie item, no label
        prior and no position bias) has no argmax, so it is broken by a coin
        flip from the judge's own RNG: an undetermined collapse is the honest
        model there, and pooling four such cells is what makes the
        `saturated_undetermined` regime visible.
        """
        p = self.marginal_p_emit_A(content_first_is_1, label_first_is_A)
        if p > 0.5:
            return "A"
        if p < 0.5:
            return "B"
        return "A" if self._rng.random() < 0.5 else "B"

    def sample_cell(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,
        k: int,
    ) -> list[Verdict]:
        """Draw k verdicts for ONE cell, applying saturation within the cell.

        The cell's top-token probability is `s + (1-s)·max(p, 1-p)` for
        saturation `s` and marginal `p`: monotonically increasing in `s` in
        EVERY regime (including p = 0.5 and label_bias = 0), reaching a
        constant cell at s = 1.0. That monotonicity is the property the knob
        is supposed to have and is asserted in
        `test_poc_llm_judge_saturation.py`.

        Sharpening toward the mode also SHIFTS the mean toward it -- that is
        what a low sampling temperature does, and it is why a saturated judge
        on a tie item reads as a pure token preference.

        Replace this method with k real judge calls to run the same design
        against a production model.
        """
        p = self.marginal_p_emit_A(content_first_is_1, label_first_is_A)
        mode = self.cell_mode(content_first_is_1, label_first_is_A)
        out: list[Verdict] = []
        for _ in range(k):
            if self._rng.random() < self.saturation:
                out.append(mode)
            else:
                out.append("A" if self._rng.random() < p else "B")
        return out


# ---------------------------------------------------------------------------
# Experiment design helpers
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    content_first_is_1: bool
    label_first_is_A: bool
    verdicts: list[Verdict]   # kept so a reader can audit the raw draws, the
                              # way the real script keeps per-sample verdicts
    n: int
    n_A: int
    top_prob: float
    saturated: bool
    gate_pvalue: float


def run_cell(
    judge: SyntheticJudge,
    content_first_is_1: bool,
    label_first_is_A: bool,
    k: int,
    saturation_threshold: float = 0.9,
    alpha: float = 0.05,
) -> CellResult:
    verdicts = judge.sample_cell(content_first_is_1, label_first_is_A, k)
    n_A = sum(1 for v in verdicts if v == "A")
    n = len(verdicts)
    top_count = max(n_A, n - n_A)
    top_prob = top_count / n if n else 0.0

    # Calibrated gate: exact one-sided binomial test
    # H0: true majority probability ≤ threshold  vs  H1: > threshold
    # At k=40, p0=0.9, α=0.05 this requires 40/40; below k=29 it can never
    # fire at all (see min_k_for_gate_power).
    pvalue = exact_binomial_onesided_pvalue(top_count, n, p0=saturation_threshold)
    saturated = pvalue < alpha

    return CellResult(
        content_first_is_1=content_first_is_1,
        label_first_is_A=label_first_is_A,
        verdicts=verdicts,
        n=n,
        n_A=n_A,
        top_prob=top_prob,
        saturated=saturated,
        gate_pvalue=pvalue,
    )


def crossed_design(judge: SyntheticJudge, k: int = 40) -> dict[str, CellResult]:
    """
    Fully crossed 2×2:
      content position  ×  label assignment
    """
    cells = {}
    for content_first_is_1 in (True, False):
        for label_first_is_A in (True, False):
            key = f"content1_first={content_first_is_1}|labelA_first={label_first_is_A}"
            cells[key] = run_cell(judge, content_first_is_1, label_first_is_A, k)
    return cells


# ---------------------------------------------------------------------------
# Statistics: position effect vs label bias vs content signal
# ---------------------------------------------------------------------------

def decompose(cells: dict[str, CellResult]) -> dict:
    """
    Marginalise to obtain:
      - pure content-position effect (signed ΔP when only position flips)
      - pure label bias              (P(say "A"), marginalised over position)
      - content preference           (P(pick content_1), marginalised over both)
    """
    # Helper to get rate of saying "A"
    def rate_A(c: CellResult) -> float:
        return c.n_A / c.n if c.n else 0.5

    # Four cells
    c11 = cells["content1_first=True|labelA_first=True"]   # content1=A, content2=B
    c10 = cells["content1_first=True|labelA_first=False"]  # content1=B, content2=A
    c01 = cells["content1_first=False|labelA_first=True"]  # content2=A, content1=B
    c00 = cells["content1_first=False|labelA_first=False"] # content2=B, content1=A

    # Label bias: P(emit "A") pooled across all cells (A is attached to each
    # content and each slot equally often)
    all_A = c11.n_A + c10.n_A + c01.n_A + c00.n_A
    all_n = c11.n + c10.n + c01.n + c00.n
    p_A = all_A / all_n

    # Content-position effect: preference for content_1 when it is first
    # versus when it is second, after marginalising label.
    # When content1 is first: cells c11 and c10
    #   In c11 content1 is labelled A → rate of choosing content1 = rate_A
    #   In c10 content1 is labelled B → rate of choosing content1 = 1 - rate_A
    prefer_1_when_first = 0.5 * (rate_A(c11) + (1 - rate_A(c10)))
    prefer_1_when_second = 0.5 * ((1 - rate_A(c01)) + rate_A(c00))
    position_effect = prefer_1_when_first - prefer_1_when_second

    # Content preference: does the verdict track CONTENT at all, once both
    # position and label are marginalised out? 0.5 = no content signal.
    content_preference = 0.5 * (prefer_1_when_first + prefer_1_when_second)

    # Saturation summary
    n_saturated = sum(1 for c in cells.values() if c.saturated)
    n_near_sat = sum(1 for c in cells.values() if c.top_prob >= 0.90)
    mean_top_prob = sum(c.top_prob for c in cells.values()) / len(cells)

    return {
        # RAW P(emit "A") -- same convention as the real script's `label_bias`.
        "label_bias": p_A,
        "label_bias_centered": p_A - 0.5,
        # SIGNED; the real script's `order_effect` is abs() of this.
        "position_effect": position_effect,
        "content_preference": content_preference,
        "n_saturated_cells": n_saturated,
        "n_near_saturated": n_near_sat,
        "mean_top_prob": mean_top_prob,
        # Calibrated gate ONLY, and ANY-condition, matching the real script's
        # `any_condition_saturated_exact`. `n_near_saturated` is the raw
        # top_prob >= 0.9 heuristic this repo argues against; it is reported
        # beside the gate as a descriptive contrast and deliberately does NOT
        # feed this flag (it used to, which let the heuristic override the
        # calibrated gate in the very demo arguing against it).
        "identifiable": n_saturated == 0,
    }


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
# One threshold per quantity: a value is either >= it or < it, so every judge
# lands in exactly one branch. (A two-sided 0.10 / 0.12 pair used to sit here
# and left a dead band that swallowed borderline judges -- and the ideal judge
# -- into a "low-signal" catch-all.)

POSITION_EFFECT_THRESHOLD = 0.10     # on |position_effect|
LABEL_BIAS_THRESHOLD = 0.10          # on |label_bias - 0.5|
CONTENT_PREFERENCE_THRESHOLD = 0.20  # on |content_preference - 0.5|

REGIME_KEYS = (
    "saturated_token_driven",
    "saturated_content_driven",
    "saturated_undetermined",
    "dispersed_position_effect",
    "dispersed_clean",
)


def classify_regime(stats: dict) -> tuple[str, list[str]]:
    """Map a `decompose()` result to exactly one regime + its advice lines.

    The five keys in `REGIME_KEYS` are exhaustive and mutually exclusive by
    construction: `identifiable` splits the space in two and each half is then
    split by single-threshold comparisons with no gap between them.
    """
    identifiable = stats["identifiable"]
    large_pos = abs(stats["position_effect"]) >= POSITION_EFFECT_THRESHOLD
    large_label = abs(stats["label_bias"] - 0.5) >= LABEL_BIAS_THRESHOLD
    strong_content = (
        abs(stats["content_preference"] - 0.5) >= CONTENT_PREFERENCE_THRESHOLD
    )

    if not identifiable:
        caveat = []
        if large_pos:
            caveat = [
                "NOTE: a large position statistic is printed above, but it is "
                "not identifying at this saturation -- that is take-away 3.",
            ]
        if large_label:
            return "saturated_token_driven", [
                "PATHOLOGICAL saturation: the calibrated gate fires and the",
                "collapse is TOKEN-driven -- label bias is large while the",
                "verdict barely tracks content.",
                "DO NOT interpret an order-effect statistic from this judge.",
            ] + caveat
        if strong_content:
            return "saturated_content_driven", [
                "BENIGN saturation: the gate fires, but the collapse tracks",
                "CONTENT -- label bias is ~0 and the verdict follows the same",
                "answer across label flips.",
                "The VERDICT is trustworthy; the ORDER statistic still is not,",
                "because there is no dispersion left to measure a flip rate in.",
            ] + caveat
        return "saturated_undetermined", [
            "Saturated with no clear driver: the gate fires, but neither a",
            "token prior nor a content winner explains the collapse.",
            "Non-identifying -- inspect per-cell results before pooling.",
        ] + caveat

    if large_pos:
        return "dispersed_position_effect", [
            "Position effect is measurable AND the distribution still has",
            "dispersion -- the calibrated gate does not fire.",
            "Double-ordering or other position mitigations are justified here.",
        ]
    return "dispersed_clean", [
        "Identifiable, and no material position effect: the gate does not fire",
        "and the crossed design finds nothing to correct for.",
        "Interpret the order statistics freely -- this is the clean case.",
    ]


# ---------------------------------------------------------------------------
# Cost illustration
# ---------------------------------------------------------------------------

def cost_illustration(
    n_items: int = 500,
    k: int = 40,
    k_pilot: int | None = None,
    cost_per_call: float = 0.008,
    sat_frac: float = 0.60,
    saturation_threshold: float = 0.9,
    alpha: float = 0.05,
) -> dict:
    """Three-way cost model.

    Rows
    ----
    naive_2cond   The confounded baseline: both content orders, ONE label
                  assignment, full k. 2 conditions × k per item.
    crossed_full  The design this PoC recommends: position × label fully
                  crossed, full k. 4 conditions × k per item -- exactly 2x the
                  naive baseline. That 2x is what the real script's DESIGN
                  NOTE 1 means by "~2x the API cost": crossing BUYS
                  identifiability and PAYS for it. It is not a saving.
    crossed_gated The crossed design with an adaptive pilot: run `k_pilot`
                  first, stop on cells the calibrated gate already flags,
                  escalate to full k on the rest.

    `k_pilot` cannot be chosen freely. Below `min_k_for_gate_power` the exact
    gate has ZERO power, so no cell could ever stop early and the reported
    saving would be fictional -- this function refuses such a pilot rather
    than quietly producing a number. It defaults to exactly that floor.
    """
    floor = min_k_for_gate_power(saturation_threshold, alpha)
    if k_pilot is None:
        k_pilot = floor
    if k_pilot < floor:
        raise ValueError(
            f"k_pilot={k_pilot} is below the exact-gate power floor "
            f"(k>={floor} at saturation_threshold={saturation_threshold}, "
            f"alpha={alpha}); the gate can never fire, so no cell could stop "
            f"early and the 'saving' would be fictional."
        )
    if k < k_pilot:
        raise ValueError(f"k={k} must be at least k_pilot={k_pilot}")

    calls_naive = n_items * k * 2
    calls_crossed_full = n_items * k * 4
    calls_crossed_gated = int(round(
        n_items * 4 * (sat_frac * k_pilot + (1.0 - sat_frac) * k)
    ))

    def usd(calls: float) -> float:
        return round(calls * cost_per_call, 2)

    return {
        "n_items": n_items,
        "k": k,
        "k_pilot": k_pilot,
        "gate_power_floor_k": floor,
        "sat_frac": sat_frac,
        "cost_per_call": cost_per_call,
        "calls_naive_2cond": calls_naive,
        "calls_crossed_full": calls_crossed_full,
        "calls_crossed_gated": calls_crossed_gated,
        "cost_naive_2cond_usd": usd(calls_naive),
        "cost_crossed_full_usd": usd(calls_crossed_full),
        "cost_crossed_gated_usd": usd(calls_crossed_gated),
        # What identifiability costs over the confounded baseline.
        # DESIGN NOTE 1 in the real script: "~2x the API cost".
        "crossing_multiplier_full": calls_crossed_full / calls_naive,
        "crossing_multiplier_gated": calls_crossed_gated / calls_naive,
        # What the gate claws back off the crossed design.
        "gate_savings_usd": usd(calls_crossed_full - calls_crossed_gated),
        "gate_savings_pct": round(
            100.0 * (1.0 - calls_crossed_gated / calls_crossed_full), 1),
    }


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def demo_judges() -> list[SyntheticJudge]:
    """The four judges the demo walks through, one per regime worth seeing."""
    return [
        # The clean baseline: dispersed, no token prior, no position effect.
        # Listed first so the reader sees what "nothing to correct for" looks
        # like before the three failure modes. (The previous classifier filed
        # exactly this judge under "mixed / low-signal -- inspect manually",
        # which is the opposite of the truth.)
        SyntheticJudge(
            name="dispersed_clean",
            base_preference=0.55,   # mild, genuine content preference
            label_bias=0.00,
            position_bias=0.00,
            saturation=0.10,
            seed=62,
        ),
        # Regime the original repo highlights: near-deterministic outputs
        # dominated by token/label preference, not content position.
        SyntheticJudge(
            name="saturated_label_biased",
            base_preference=0.50,   # pure tie on content
            label_bias=+0.70,       # strong preference for token "A"
            position_bias=0.00,     # zero genuine position effect
            saturation=0.99,        # high within-cell collapse
            seed=42,
        ),
        # Opposite regime: real position effect, still dispersed enough
        # that the gate stays open.
        SyntheticJudge(
            name="dispersed_position_biased",
            base_preference=0.55,
            label_bias=0.0,
            position_bias=0.20,     # real position effect
            saturation=0.05,        # mostly stochastic
            seed=40,
        ),
        # Clear content winner → benign saturation (judge correctly
        # collapses onto the better answer, and the classifier says so).
        SyntheticJudge(
            name="saturated_content_driven",
            base_preference=0.95,   # clear winner
            label_bias=0.02,
            position_bias=0.01,
            saturation=0.99,
            seed=99,
        ),
    ]


def main() -> None:
    k = 40
    sat_threshold = 0.9
    alpha = 0.05
    floor = min_k_for_gate_power(sat_threshold, alpha)

    print("=" * 72)
    print("PoC: LLM-Judge Saturation Gate & Position × Label Decomposition")
    print("=" * 72)
    print()
    print(f"Samples per cell (k) = {k}")
    print(f"Saturation gate: exact one-sided binomial, "
          f"H0: p ≤ {sat_threshold}, α = {alpha}")
    print(f"Gate power floor   : k ≥ {floor} (below this the gate cannot fire "
          f"for ANY outcome)")
    print()

    for judge in demo_judges():
        print("-" * 72)
        print(f"Judge: {judge.name}")
        print(f"  (base_pref={judge.base_preference}, label_bias={judge.label_bias},"
              f" position_bias={judge.position_bias}, saturation={judge.saturation})")
        print()

        cells = crossed_design(judge, k=k)
        stats = decompose(cells)

        print("  Cell summary (top_prob / saturated?):")
        for key, cell in cells.items():
            flag = "SAT" if cell.saturated else "ok "
            print(f"    [{flag}] {key}: top_prob={cell.top_prob:.3f}  "
                  f"p_gate={cell.gate_pvalue:.4f}  n_A={cell.n_A}/{cell.n}")

        print()
        print("  Decomposition (after marginalising the crossed design):")
        print(f"    Label/token bias P(emit A): {stats['label_bias']:.3f}   "
              f"(centred: {stats['label_bias_centered']:+.3f}; 0.500 = no prior)")
        print(f"    Content-position effect   : {stats['position_effect']:+.3f}   "
              f"(signed; real script reports |·|)")
        print(f"    Content preference        : {stats['content_preference']:.3f}   "
              f"(0.500 = verdict ignores content)")
        print(f"    Calibrated gate saturated : {stats['n_saturated_cells']}/4")
        print(f"    Raw top_prob ≥ 0.90       : {stats['n_near_saturated']}/4   "
              f"(the heuristic this repo argues against -- shown for contrast, "
              f"not used)")
        print(f"    Mean top-token prob       : {stats['mean_top_prob']:.3f}")
        print(f"    Identifiable for order    : {stats['identifiable']}")
        print()

        regime, lines = classify_regime(stats)
        print(f"  → Regime: {regime}")
        for line in lines:
            print(f"    {line}")
        print()

    # Cost illustration
    costs = cost_illustration(
        n_items=500,
        k=k,
        cost_per_call=0.008,
        saturation_threshold=sat_threshold,
        alpha=alpha,
    )
    print("=" * 72)
    print(f"Cost illustration ({costs['n_items']} pairwise items, "
          f"${costs['cost_per_call']:.3f} / judge call)")
    print("=" * 72)
    print(f"  Naive 2-condition (confounded, k={costs['k']}) : "
          f"{costs['calls_naive_2cond']:,} calls  →  "
          f"${costs['cost_naive_2cond_usd']:,.2f}")
    print(f"  Crossed 2×2, full k (recommended)     : "
          f"{costs['calls_crossed_full']:,} calls  →  "
          f"${costs['cost_crossed_full_usd']:,.2f}   "
          f"({costs['crossing_multiplier_full']:.2f}× naive)")
    print(f"  Crossed 2×2 + adaptive gate (pilot k={costs['k_pilot']}) : "
          f"{costs['calls_crossed_gated']:,} calls  →  "
          f"${costs['cost_crossed_gated_usd']:,.2f}   "
          f"({costs['crossing_multiplier_gated']:.2f}× naive)")
    print()
    print(f"  Crossing BUYS identifiability and costs "
          f"{costs['crossing_multiplier_full']:.2f}× the confounded baseline "
          f"-- that is")
    print(f"  the real script's DESIGN NOTE 1 (\"~2x the API cost\"), not a saving.")
    print(f"  The gate claws back ${costs['gate_savings_usd']:,.2f} of it "
          f"({costs['gate_savings_pct']} % off the crossed design),")
    print(f"  landing identifiable measurement at "
          f"{costs['crossing_multiplier_gated']:.2f}× naive instead of "
          f"{costs['crossing_multiplier_full']:.2f}×.")
    print()
    scale_items = 50_000
    scale_factor = scale_items / costs["n_items"]
    print(f"  Scaling note: at {scale_items:,} items / month the same ratios "
          f"imply roughly")
    print(f"  ${costs['gate_savings_usd'] * scale_factor:,.0f} / month of gate "
          f"savings for a single evaluation stream")
    print(f"  (= ${costs['gate_savings_usd']:,.2f} × {scale_factor:.0f}), before "
          f"counting avoided engineering rework.")
    print()
    print(f"  The pilot cannot go below k={costs['gate_power_floor_k']}: that is "
          f"the exact gate's power floor.")
    print("  A smaller pilot would report a bigger saving that could never be")
    print("  realised, because the gate would never fire to stop anything early.")
    print()
    print("  CAVEAT: the gated row models an ADAPTIVE early-stopping strategy")
    print("  that the real measurement script does NOT implement -- it always")
    print("  samples the full k. That row is an illustrative saving from a")
    print("  strategy that could be built, not a result the repo delivers today.")
    print()

    print("=" * 72)
    print("Take-away")
    print("=" * 72)
    print("""
  1. Always cross content position with label assignment; otherwise
     position and token preference are confounded. It costs ~2x.
  2. Apply a calibrated saturation gate (exact binomial, not a raw
     top-prob threshold) before interpreting order-effect numbers --
     and check the gate's power floor before choosing k.
  3. When the gate fires, further samples or naive double-ordering
     buy almost no additional identifying information -- that is pure
     cost with no measurement return.
  4. Report label bias, position effect and content preference as
     separate statistics: saturation driven by a token prior and
     saturation driven by a genuine content winner need opposite
     responses.

  This PoC is fully offline. Replace SyntheticJudge.sample_cell with k real
  OpenRouter / provider calls to run the same design on production judges.
""")


if __name__ == "__main__":
    main()
