#!/usr/bin/env python3
"""
PoC: LLM-Judge Saturation Gate & Position×Label Decomposition
=============================================================

Demonstrates the core measurement hygiene from
https://github.com/privatedick/llm-judge-saturation

What this PoC shows
-------------------
1. A fully crossed content-POSITION × LABEL-ASSIGNMENT design.
2. A calibrated saturation / identifiability gate (exact one-sided binomial).
3. Decomposition of apparent "order effect" into:
   - pure content-position effect
   - pure label/token preference
4. Cost implication: when the gate fails, additional samples (or naïve
   double-ordering) buy almost no identifying information.

No external LLM API is required. Synthetic judges with controllable
saturation and bias parameters are used so the PoC is fully offline and
reproducible.

HONEST SCOPE (read before citing the cost numbers): the cost illustration
below models an ADAPTIVE early-stopping sampling strategy (small pilot,
escalate only if not confidently saturated) that this repository's actual
measurement script (../python/experiment_llm_judge_saturation.py) does NOT
implement -- that script always samples the full k per condition. The
$10k-$25k/month figure is what such a strategy COULD save if built, not a
result this repo currently delivers. Don't cite it as a measured outcome.
The position×label decomposition and the calibrated gate itself, on the
other hand, are exactly what the real measurement script does.

Also note: `position_bias` is a per-side parameter (±position_bias around
base_preference), so the measured `position_effect` statistic in the output
comes out close to 2x the parameter value -- that's the crossed design's
difference-in-differences, not a bug, but don't expect the two numbers to
match 1:1.

Run:
    python poc_llm_judge_saturation.py
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
    """P(X >= successes | X ~ Bin(n, p0))."""
    if n == 0:
        return 1.0
    total = 0.0
    for k in range(successes, n + 1):
        total += _binom_pmf(k, n, p0)
    return min(1.0, total)


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
    - saturation: probability that the judge collapses to its mode (near-
      deterministic). When not saturated it samples from the soft preference.
    """
    name: str
    base_preference: float = 0.55   # slight preference for content_1
    label_bias: float = 0.0         # P(say "A") shift
    position_bias: float = 0.0      # preference for first-presented content
    saturation: float = 0.85        # P(collapse to mode)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _soft_p_prefer_first(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,
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

    def sample(
        self,
        content_first_is_1: bool,
        label_first_is_A: bool,
    ) -> Verdict:
        """
        Return a single verdict letter ("A" or "B").

        Generative story
        ----------------
        1. Soft content preference (base + position) → preferred *content*.
        2. Map preferred content to a letter under the current labelling.
        3. Mix with an independent label prior:
             P(emit A) = (1 - |λ|) * 1[soft=A] + max(λ, 0)
           so λ > 0 pulls toward "A", λ < 0 pulls toward "B".
        4. With probability `saturation` collapse to the mode of that
           mixture (near-deterministic output).
        """
        p_prefer_first = self._soft_p_prefer_first(content_first_is_1, label_first_is_A)

        prefer_first = self._rng.random() < p_prefer_first
        preferred_content_is_1 = (
            prefer_first if content_first_is_1 else (not prefer_first)
        )

        if label_first_is_A:
            letter_for_content_1: Verdict = "A" if content_first_is_1 else "B"
        else:
            letter_for_content_1 = "B" if content_first_is_1 else "A"

        soft_is_A = preferred_content_is_1 == (letter_for_content_1 == "A")

        # λ ∈ [-1, 1]: absolute strength of an independent label prior.
        # Final P(A) = (1-|λ|)·1[soft=A] + max(λ,0)
        # so λ=+0.7 → P(A) ≈ 0.7 even when soft says B, and ≈ 1.0 when soft says A.
        lam = max(-0.99, min(0.99, self.label_bias))
        p_emit_A = (1.0 - abs(lam)) * (1.0 if soft_is_A else 0.0) + max(lam, 0.0)
        if lam < 0:
            # symmetric pull toward B
            p_emit_A = (1.0 - abs(lam)) * (1.0 if soft_is_A else 0.0)
        p_emit_A = max(0.01, min(0.99, p_emit_A))

        if self._rng.random() < self.saturation:
            letter: Verdict = "A" if p_emit_A >= 0.5 else "B"
        else:
            letter = "A" if self._rng.random() < p_emit_A else "B"

        return letter


# ---------------------------------------------------------------------------
# Experiment design helpers
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    content_first_is_1: bool
    label_first_is_A: bool
    verdicts: list[Verdict]
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
    verdicts = [
        judge.sample(content_first_is_1, label_first_is_A) for _ in range(k)
    ]
    n_A = sum(1 for v in verdicts if v == "A")
    n = len(verdicts)
    top_count = max(n_A, n - n_A)
    top_prob = top_count / n if n else 0.0

    # Calibrated gate: exact one-sided binomial test
    # H0: true majority probability ≤ threshold  vs  H1: > threshold
    # At k=40, p0=0.9, α=0.05 this requires 40/40 (see README rationale
    # in the original repo: point-estimate thresholds over-flag).
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
# Statistics: position effect vs label bias
# ---------------------------------------------------------------------------

def decompose(cells: dict[str, CellResult]) -> dict:
    """
    Marginalise to obtain:
      - pure content-position effect  (average |ΔP| when only position flips)
      - pure label bias               (P(say "A") - 0.5, marginalised over position)
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
    label_bias = p_A - 0.5

    # Content-position effect: preference for content_1 when it is first
    # versus when it is second, after marginalising label.
    # When content1 is first: cells c11 and c10
    #   In c11 content1 is labelled A → rate of choosing content1 = rate_A
    #   In c10 content1 is labelled B → rate of choosing content1 = 1 - rate_A
    prefer_1_when_first = 0.5 * (rate_A(c11) + (1 - rate_A(c10)))
    prefer_1_when_second = 0.5 * ((1 - rate_A(c01)) + rate_A(c00))
    position_effect = prefer_1_when_first - prefer_1_when_second

    # Saturation summary
    n_saturated = sum(1 for c in cells.values() if c.saturated)
    n_near_sat = sum(1 for c in cells.values() if c.top_prob >= 0.90)
    mean_top_prob = sum(c.top_prob for c in cells.values()) / len(cells)

    return {
        "label_bias": label_bias,
        "position_effect": position_effect,
        "p_emit_A": p_A,
        "n_saturated_cells": n_saturated,
        "n_near_saturated": n_near_sat,
        "mean_top_prob": mean_top_prob,
        # Strict gate + descriptive near-sat both considered
        "identifiable": n_saturated < 2 and n_near_sat < 3,
    }


# ---------------------------------------------------------------------------
# Cost illustration
# ---------------------------------------------------------------------------

def cost_illustration(
    n_items: int = 500,
    k_naive: int = 40,
    k_gated: int = 12,
    cost_per_call: float = 0.008,
    double_order_naive: bool = True,
) -> dict:
    """
    Rough cost model.

    Naive practice: always run both orders (or a fixed large k) regardless of
    saturation.

    Gated practice: run a small pilot; if the cell is confidently saturated,
    stop; otherwise continue to full k.  Also drop pure double-ordering when
    position effect is negligible after label counterbalancing.
    """
    calls_naive = n_items * k_naive * (2 if double_order_naive else 1)
    cost_naive = calls_naive * cost_per_call

    # Assume ~60 % of cells are saturated (matches the order of magnitude in
    # the original repo). On those we only pay the pilot cost.
    sat_frac = 0.60
    calls_gated = n_items * (
        sat_frac * k_gated + (1 - sat_frac) * k_naive
    )
    # No systematic double-ordering once label is counterbalanced
    cost_gated = calls_gated * cost_per_call

    return {
        "n_items": n_items,
        "calls_naive": calls_naive,
        "cost_naive_usd": round(cost_naive, 2),
        "calls_gated": int(calls_gated),
        "cost_gated_usd": round(cost_gated, 2),
        "savings_usd": round(cost_naive - cost_gated, 2),
        "savings_pct": round(100 * (1 - cost_gated / cost_naive), 1),
    }


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("PoC: LLM-Judge Saturation Gate & Position × Label Decomposition")
    print("=" * 72)
    print()

    # Three synthetic judges that illustrate the key regimes
    judges = [
        # Regime the original repo highlights: near-deterministic outputs
        # dominated by token/label preference, not content position.
        SyntheticJudge(
            name="saturated_label_biased",
            base_preference=0.50,   # pure tie on content
            label_bias=+0.70,       # strong preference for token "A"
            position_bias=0.00,     # zero genuine position effect
            saturation=0.90,        # high collapse rate
            seed=42,
        ),
        # Opposite regime: real position effect, still dispersed enough
        # that the gate stays open.
        SyntheticJudge(
            name="dispersed_position_biased",
            base_preference=0.55,
            label_bias=0.0,
            position_bias=0.20,     # real position effect
            saturation=0.15,        # mostly stochastic
            seed=7,
        ),
        # Clear content winner → benign saturation (judge correctly
        # collapses onto the better answer).
        SyntheticJudge(
            name="saturated_content_driven",
            base_preference=0.95,   # clear winner
            label_bias=0.02,
            position_bias=0.01,
            saturation=0.98,
            seed=99,
        ),
    ]

    k = 40
    print(f"Samples per cell (k) = {k}")
    print(f"Saturation gate: exact one-sided binomial, H0: p ≤ 0.9, α = 0.05")
    print()

    for judge in judges:
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
        print(f"    Pure label/token bias   : {stats['label_bias']:+.3f}  "
              f"(P(emit A) = {stats['p_emit_A']:.3f})")
        print(f"    Pure content-position   : {stats['position_effect']:+.3f}")
        print(f"    Strict-gate saturated   : {stats['n_saturated_cells']}/4")
        print(f"    Near-saturated (p≥0.90) : {stats['n_near_saturated']}/4")
        print(f"    Mean top-token prob     : {stats['mean_top_prob']:.3f}")
        print(f"    Identifiable for order  : {stats['identifiable']}")
        print()

        high_sat = stats["n_saturated_cells"] >= 2 or stats["n_near_saturated"] >= 3
        small_pos = abs(stats["position_effect"]) < 0.10
        large_pos = abs(stats["position_effect"]) > 0.12
        large_label = abs(stats["label_bias"]) > 0.12

        if high_sat and small_pos:
            print("  → Gate recommendation: DO NOT interpret a large order-effect")
            print("    statistic from this judge. Distribution is near-deterministic;")
            print("    any measured 'flip rate' is largely non-identifying.")
            if large_label:
                print("    Dominant observable is LABEL/TOKEN preference, not position.")
        elif large_pos and not high_sat:
            print("  → Gate recommendation: position effect is measurable and the")
            print("    distribution still has dispersion. Double-ordering or other")
            print("    position mitigations may be justified.")
        elif high_sat and not small_pos:
            print("  → Mixed: high saturation AND non-trivial position effect.")
            print("    Treat order statistics with caution; prefer more samples")
            print("    or a less saturated judge before acting on the number.")
        else:
            print("  → Mixed / low-signal regime: inspect per-cell results before pooling.")
        print()

    # Cost illustration
    print("=" * 72)
    print("Cost illustration (500 pairwise items, $0.008 / judge call)")
    print("=" * 72)
    costs = cost_illustration(
        n_items=500,
        k_naive=40,
        k_gated=12,
        cost_per_call=0.008,
        double_order_naive=True,
    )
    print(f"  Naive (always k=40, both orders) : "
          f"{costs['calls_naive']:,} calls  →  ${costs['cost_naive_usd']:,.2f}")
    print(f"  Gated  (pilot k=12, full only if")
    print(f"          not saturated; no auto")
    print(f"          double-order)            : "
          f"{costs['calls_gated']:,} calls  →  ${costs['cost_gated_usd']:,.2f}")
    print(f"  Estimated savings                : "
          f"${costs['savings_usd']:,.2f}  ({costs['savings_pct']} %)")
    print()
    print("  Scaling note: at 50 k items / month the same ratios imply")
    print("  roughly $10 k-$25 k monthly savings for a single evaluation")
    print("  stream, before counting avoided engineering rework.")
    print()
    print("  CAVEAT: this models an ADAPTIVE early-stopping strategy (pilot")
    print("  k=12, escalate only if not confidently saturated) that the real")
    print("  measurement script does NOT implement -- it always samples the")
    print("  full k. These are illustrative savings from a strategy that")
    print("  could be built, not a result the current repo delivers today.")
    print()

    print("=" * 72)
    print("Take-away")
    print("=" * 72)
    print("""
  1. Always cross content position with label assignment; otherwise
     position and token preference are confounded.
  2. Apply a calibrated saturation gate (exact binomial, not a raw
     top-prob threshold) before interpreting order-effect numbers.
  3. When the gate fails, further samples or naive double-ordering
     buy almost no additional identifying information -- that is pure
     cost with no measurement return.
  4. Report label bias and position effect as separate statistics.

  This PoC is fully offline. Replace SyntheticJudge.sample with a real
  OpenRouter / provider call to run the same design on production judges.
""")


if __name__ == "__main__":
    main()
