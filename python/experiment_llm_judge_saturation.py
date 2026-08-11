"""Empirical: output saturation and the identifiability gate in real LLM judges.

Tests the measurement-saturation hypothesis with real data via OpenRouter: LLM
judges are frequently near-deterministic on pairwise verdicts, and where they
are, the verdict distribution is degenerate — so an order-effect statistic
computed on it is non-identifying (the Kang 2026 saturation caveat).

Design (per model x item): a fully crossed 2x2 — content POSITION (which answer
appears first in the prompt text) x LABEL ASSIGNMENT (whether the first-presented
answer is labeled "A" or "B"), K samples per cell at temperature T.

DESIGN NOTE 1 (label counterbalancing, round-2 review 2026-08-11): the original
2-condition design (slot A = ans1-first, slot A = ans2-first) perfectly
confounded content position with label — the first-presented answer was ALWAYS
labeled "A". A judge that flips on a tie item could be following spatial
position, or could just have a raw token prior for emitting "A" (or "B") —
indistinguishable in that design, and the caveat box said so without fixing it.
Crossing label assignment with position (this file) separates them for ~2x the
API cost: `order_effect` now marginalizes over label (a pure content-position
effect) and `label_bias` marginalizes over position (a pure token-prior effect),
both measured from the same run instead of one being a footnote.

DESIGN NOTE 2 (calibrated saturation gate, round-2 review): flagging a cell
"saturated" from a bare `top_prob >= threshold` point-estimate comparison is
itself statistically uncalibrated. At k samples, finite-sample noise pushes a
truly-dispersed judge over the line a real fraction of the time — e.g. a judge
whose long-run majority probability is genuinely 0.85 still crosses an observed
top_prob>=0.9 line 28% of the time at k=12 (exact binomial), and the bias
persists, smaller, at larger k. `_dist_stats` now runs a one-sided exact
binomial test (H0: p<=threshold vs H1: p>threshold) and reports `saturated_exact`
from that instead of the raw comparison; `top_prob` and a Clopper-Pearson 95% CI
remain as descriptive point estimates, not as the gate criterion itself.

DESIGN NOTE 3 (empty-cell stats, round-2 review): `_dist_stats([])` used to
return `top_prob=1.0, entropy=0.0, n_eff=1.0` — i.e. an all-parse-failure cell
read as PERFECTLY DETERMINISTIC, the opposite of what "no data" means. It now
returns None for every derived field; callers must (and do) filter None before
aggregating, so a partially-failed run degrades to "excluded", not "counted as
maximally saturated on zero evidence".

Honest scope: this measures the model's *sampled* verdict distribution at T>0, a
practitioner-realistic observable, not raw next-token logprobs. It quantifies how
often the gate would fire and how order/label stats read under saturation; it
does not claim saturation *causes* the order effect (it impairs its measurement).

Cost-controlled: 3 cheap models, ~6 items, 4 conditions, modest K. Use --pilot to
price it first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from datetime import date
from math import comb
from pathlib import Path

import numpy as np
import requests
from joblib import Parallel, delayed
from scipy import stats as spstats

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-8b",
    "mistralai/mistral-nemo",
]

# Pairwise items. `better` marks the higher-quality answer (or "tie"). A
# position-robust judge picks the SAME actual answer regardless of slot order;
# "tie" items maximally expose position/label bias (no content signal to anchor on).
ITEMS = [
    {"id": "arith", "question": "What is 17 * 23?",
     "ans1": "391.", "ans2": "It's roughly 400, I'm not certain.", "better": "ans1"},
    {"id": "hashmap", "question": "In one sentence, what is a hash map?",
     "ans1": "A data structure mapping keys to values via a hash function for average O(1) lookup.",
     "ans2": "It's a kind of list where you look things up somehow.", "better": "ans1"},
    {"id": "capital", "question": "What is the capital of Australia?",
     "ans1": "Canberra.", "ans2": "Sydney.", "better": "ans1"},
    {"id": "tie_paraphrase", "question": "Define recursion briefly.",
     "ans1": "A function that calls itself on a smaller input until a base case.",
     "ans2": "When a routine invokes itself with a reduced argument until it hits a stopping condition.",
     "better": "tie"},
    {"id": "tie_greeting", "question": "Write a one-line friendly greeting.",
     "ans1": "Hi there, great to see you today!",
     "ans2": "Hello, lovely to have you here today!", "better": "tie"},
    {"id": "bignum", "question": "Is 2**10 equal to 1024?",
     "ans1": "Yes, 2 to the 10th is 1024.", "ans2": "No, it's 1000.", "better": "ans1"},
]

JUDGE_SYSTEM = (
    "You are a strict evaluator. You will see a question and two candidate "
    "answers, A and B. Reply with exactly one character: 'A' if A is better, "
    "'B' if B is better. No explanation."
)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "Content-Type": "application/json",
    }


def _judge_messages(question: str, first_content: str, first_label: str,
                     second_content: str) -> list[dict]:
    """Build the prompt for one (position, label) condition.

    `first_content` is always presented first in the text; `first_label` (A or
    B) is the label attached to it. Decoupling these two is the fix for
    DESIGN NOTE 1 — the caller varies both independently instead of first
    always meaning "A".
    """
    second_label = "B" if first_label == "A" else "A"
    # Keep the labels in A-then-B reading order in the prompt regardless of
    # which content each is attached to, so the text always reads "Answer A: …
    # Answer B: …" — otherwise "label order in text" would become a third
    # confounded factor.
    if first_label == "A":
        block = f"Answer A: {first_content}\n\nAnswer B: {second_content}"
    else:
        block = f"Answer A: {second_content}\n\nAnswer B: {first_content}"
    user = f"Question: {question}\n\n{block}\n\nWhich answer is better? Reply with only 'A' or 'B'."
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


# CASE-SENSITIVE on purpose. An earlier version uppercased the text first, which
# made the English article "a" match \b([AB])\b — so "I think a good choice is B"
# parsed as A. Because position bias is measured as slot-A preference, that bug
# could manufacture the very effect being measured. Lowercase "a" must never match.
_VERDICT_RE = re.compile(r"\b([AB])\b")


def _parse_verdict(content: str) -> str | None:
    """Parse an 'A'/'B' verdict. Returns None rather than guessing."""
    if not content:
        return None
    s = content.strip()
    if s.upper() in ("A", "B"):          # the requested format
        return s.upper()
    if s[0] in "AB" and (len(s) == 1 or not s[1].isalpha()):
        return s[0]                       # "B." / "B is better"
    m = _VERDICT_RE.search(s)             # "Answer A is correct"
    return m.group(1) if m else None


def _call(model: str, messages: list[dict], temperature: float, timeout: int = 90,
          max_retries: int = 4, max_tokens: int = 1024) -> tuple[str | None, dict]:
    # max_tokens is generous because some judges are reasoning models that spend
    # the budget thinking before emitting the verdict; too small a cap truncates
    # them (finish_reason='length', empty content) and looks like a failure.
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens}
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=_headers(), json=payload, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.RequestException(f"retryable {r.status_code}")
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {}) or {}
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            verdict = _parse_verdict(content)
            via_reasoning = False
            if verdict is None:
                # Reasoning-model fallback: the verdict concludes the trace, so
                # take the LAST match — but case-sensitively (see _VERDICT_RE).
                reasoning = msg.get("reasoning") or ""
                hits = _VERDICT_RE.findall(reasoning)
                verdict = hits[-1] if hits else None
                via_reasoning = verdict is not None
            return verdict, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "via_reasoning": via_reasoning,
            }
        except (requests.RequestException, KeyError, IndexError, ValueError):
            if attempt < max_retries - 1:
                # exponential backoff with jitter
                time.sleep((2 ** attempt) * 0.5 + random.random() * 0.3)
                continue
            return None, {"prompt_tokens": 0, "completion_tokens": 0, "via_reasoning": False}
    return None, {"prompt_tokens": 0, "completion_tokens": 0, "via_reasoning": False}


def _fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]] (no scipy dependency).

    Sums the hypergeometric probabilities of all tables at least as extreme as
    the observed one, conditioning on the margins.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, c1 = a + b, a + c

    def _p(x: int) -> float:
        return (comb(r1, x) * comb(n - r1, c1 - x)) / comb(n, c1)

    p_obs = _p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    total = sum(_p(x) for x in range(lo, hi + 1) if _p(x) <= p_obs + 1e-12)
    return float(min(1.0, total))


def _holm_significant(pvals: list[float], alpha: float = 0.05) -> int:
    """Number of hypotheses surviving Holm-Bonferroni at ``alpha``.

    Testing 18 cells at a nominal 0.05 makes ~1 false positive the expected
    outcome, so the corrected count is the one worth quoting.
    """
    m = len(pvals)
    if m == 0:
        return 0
    k = 0
    for i, p in enumerate(sorted(pvals)):
        if p <= alpha / (m - i):
            k += 1
        else:
            break
    return k


def _dist_stats(verdicts: list[str], sat_threshold: float = 0.9, alpha: float = 0.05) -> dict:
    """Dispersion + CALIBRATED saturation test of a verdict distribution over
    {A, B} (drop None). See DESIGN NOTE 2 (gate) and DESIGN NOTE 3 (n=0) above.
    """
    vs = [v for v in verdicts if v in ("A", "B")]
    n = len(vs)
    if n == 0:
        # Missing data, not maximal saturation — every field propagates None.
        return {"n": 0, "top_prob": None, "top_count": None, "entropy_bits": None,
                "n_eff": None, "p_A": None, "top_prob_ci95": None,
                "saturated_exact": None, "saturated_pvalue": None}
    c = Counter(vs)
    p = np.array([c.get("A", 0), c.get("B", 0)], dtype=float) / n
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    n_eff = float(1.0 / np.sum(p ** 2))
    top_count = int(max(c.get("A", 0), c.get("B", 0)))
    # Descriptive exact 95% CI (Clopper-Pearson) on the point estimate.
    ci = spstats.binomtest(top_count, n).proportion_ci(confidence_level=0.95, method="exact")
    # The GATE: one-sided exact test, H0: p <= sat_threshold vs H1: p > sat_threshold.
    # "saturated_exact" is True only when H0 is rejected at `alpha` — confident
    # evidence of saturation, not a lucky point estimate.
    bt = spstats.binomtest(top_count, n, sat_threshold, alternative="greater")
    return {"n": n, "top_prob": float(top_count / n), "top_count": top_count,
            "entropy_bits": entropy, "n_eff": n_eff, "p_A": float(p[0]),
            "top_prob_ci95": [round(float(ci.low), 4), round(float(ci.high), 4)],
            "saturated_exact": bool(bt.pvalue < alpha),
            "saturated_pvalue": float(bt.pvalue)}


def _run_cell(model: str, item: dict, k: int, temperature: float,
              sat_threshold: float, alpha: float) -> dict:
    """One (model, item) cell: fully crossed 2x2 of content POSITION (ans1 or
    ans2 presented first) x LABEL ASSIGNMENT (first-presented content labeled
    A or B) — see DESIGN NOTE 1. Four conditions, k samples each.
    """
    raw: dict[tuple[int, bool], list[str | None]] = {}
    n_via_reasoning = 0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    for p in (1, 2):
        content_first = item[f"ans{p}"]
        content_second = item[f"ans{3 - p}"]
        for swap in (False, True):
            first_label = "B" if swap else "A"
            calls = [_call(model, _judge_messages(item["question"], content_first,
                                                   first_label, content_second),
                           temperature) for _ in range(k)]
            raw[(p, swap)] = [v for v, _ in calls]
            for _, u in calls:
                usage_total["prompt_tokens"] += u["prompt_tokens"]
                usage_total["completion_tokens"] += u["completion_tokens"]
                n_via_reasoning += int(u.get("via_reasoning", False))

    dist = {key: _dist_stats(vs, sat_threshold, alpha) for key, vs in raw.items()}

    def chose_ans1_frac(p: int, swap: bool) -> float | None:
        vs = [v for v in raw[(p, swap)] if v in ("A", "B")]
        if not vs:
            return None
        first_label = "B" if swap else "A"
        second_label = "A" if swap else "B"
        want = first_label if p == 1 else second_label  # label attached to ans1 here
        return sum(v == want for v in vs) / len(vs)

    chose_ans1 = {(p, swap): chose_ans1_frac(p, swap) for p in (1, 2) for swap in (False, True)}

    # order_effect: content-POSITION effect on which answer is judged better,
    # marginalized over label assignment (pure position signal).
    pos1_vals = [chose_ans1[(1, s)] for s in (False, True) if chose_ans1[(1, s)] is not None]
    pos2_vals = [chose_ans1[(2, s)] for s in (False, True) if chose_ans1[(2, s)] is not None]
    p_ans1_first = float(np.mean(pos1_vals)) if pos1_vals else None
    p_ans1_second = float(np.mean(pos2_vals)) if pos2_vals else None
    order_effect = (abs(p_ans1_first - p_ans1_second)
                     if p_ans1_first is not None and p_ans1_second is not None else None)

    # label_bias: raw preference for emitting the TOKEN "A", marginalized over
    # content and position (fully crossed design — "A" is attached to ans1 and
    # ans2, first and second, equally often across the 4 conditions). This is
    # what DESIGN NOTE 1 exists to make measurable instead of confounded.
    all_verdicts = [v for vs in raw.values() for v in vs if v in ("A", "B")]
    label_bias = (sum(v == "A" for v in all_verdicts) / len(all_verdicts)) if all_verdicts else None

    # Fisher exact on the position effect: pool "chose ans1 | ans1 first" vs
    # "chose ans1 | ans1 second" as 2x2 counts across both label conditions.
    def counts_ans1_given_position(p: int) -> tuple[int, int]:
        chosen = total = 0
        for swap in (False, True):
            vs = [v for v in raw[(p, swap)] if v in ("A", "B")]
            first_label = "B" if swap else "A"
            second_label = "A" if swap else "B"
            want = first_label if p == 1 else second_label
            chosen += sum(v == want for v in vs)
            total += len(vs)
        return chosen, total

    a1, n1 = counts_ans1_given_position(1)
    a2, n2 = counts_ans1_given_position(2)
    order_p_value = _fisher_exact_2x2(a1, n1 - a1, a2, n2 - a2)

    # Two-sided gate over all 4 conditions (fails if ANY is exact-saturated).
    valid_dist = [d for d in dist.values() if d["n"] > 0]
    worst_top_prob = max((d["top_prob"] for d in valid_dist), default=None)
    worst_entropy = min((d["entropy_bits"] for d in valid_dist), default=None)
    worst_n_eff = min((d["n_eff"] for d in valid_dist), default=None)
    any_saturated_exact = (any(d["saturated_exact"] for d in valid_dist) if valid_dist else None)

    tag = lambda p, swap: f"pos{p}_{'swap' if swap else 'norm'}"  # noqa: E731
    return {
        "model": model, "item": item["id"], "better": item["better"],
        "conditions": {tag(p, s): dist[(p, s)] for p in (1, 2) for s in (False, True)},
        "order_effect": order_effect, "order_effect_fisher_p": order_p_value,
        "label_bias": label_bias,
        "worst_condition_top_prob": worst_top_prob,
        "worst_condition_entropy_bits": worst_entropy,
        "worst_condition_n_eff": worst_n_eff,
        "any_condition_saturated_exact": any_saturated_exact,
        # raw per-sample verdicts so a reader can audit parsing/verdicts directly
        "raw_verdicts": {tag(p, s): raw[(p, s)] for p in (1, 2) for s in (False, True)},
        "n_via_reasoning_fallback": n_via_reasoning,
        "usage": usage_total,
    }


def _price(cells: list[dict]) -> float:
    # rough: use ~$0.2/1M in, $0.4/1M out as a conservative blended estimate
    it = sum(c["usage"]["prompt_tokens"] for c in cells)
    ot = sum(c["usage"]["completion_tokens"] for c in cells)
    return it / 1e6 * 0.2 + ot / 1e6 * 0.4


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15, help="samples per (model,item,condition)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=2, help="parallel API threads")
    ap.add_argument("--sat-top-prob", type=float, default=0.9,
                    help="null-hypothesis threshold for the exact saturation test")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the exact saturation test")
    ap.add_argument("--pilot", action="store_true", help="1 model, 2 items, k=6")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")

    models = MODELS[:1] if args.pilot else MODELS
    items = ITEMS[:2] if args.pilot else ITEMS
    k = 6 if args.pilot else args.k

    tasks = [(m, it) for m in models for it in items]
    cells = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(_run_cell)(m, it, k, args.temperature, args.sat_top_prob, args.alpha)
        for m, it in tasks
    )

    valid = [c for c in cells if c["order_effect"] is not None]
    # CELL level: a cell fails the two-sided gate if ANY of its 4 conditions is
    # exact-saturated (calibrated test, not a bare top_prob comparison).
    n_sat = sum(1 for c in valid if c["any_condition_saturated_exact"])
    interp = [c for c in valid if not c["any_condition_saturated_exact"]]
    # CONDITION level: the unconflated count across all 4 conditions per cell.
    all_conditions = [d for c in valid for d in c["conditions"].values() if d["n"] > 0]
    n_cond_sat = sum(1 for d in all_conditions if d["saturated_exact"])
    # Significance: how many order effects are distinguishable from zero at this k.
    n_sig = sum(1 for c in valid if (c["order_effect_fisher_p"] or 1.0) < 0.05)
    n_sig_holm = _holm_significant([c["order_effect_fisher_p"] or 1.0 for c in valid])
    ties = [c for c in valid if c["better"] == "tie" and c["label_bias"] is not None]
    tie_label_bias = [c["label_bias"] for c in ties]
    tie_order_effect = [c["order_effect"] for c in ties if c["order_effect"] is not None]
    result = {
        "experiment": "llm_judge_saturation",
        "date": date.today().isoformat(),
        "config": {"models": models, "n_items": len(items), "k": k,
                   "temperature": args.temperature, "sat_top_prob": args.sat_top_prob,
                   "alpha": args.alpha, "design": "position x label, fully crossed 2x2"},
        "cells": cells,
        "summary": {
            "n_cells": len(valid),
            "n_cells_failing_two_sided_gate": n_sat,
            "frac_cells_failing_two_sided_gate": (n_sat / len(valid)) if valid else 0.0,
            "n_conditions": len(all_conditions),
            "n_conditions_saturated": n_cond_sat,
            "frac_conditions_saturated": (n_cond_sat / len(all_conditions)) if all_conditions else 0.0,
            "n_interpretable": len(interp),
            "n_order_effects_significant_p05_raw": n_sig,
            "n_order_effects_significant_holm": n_sig_holm,
            "mean_order_effect_all": float(np.mean([c["order_effect"] for c in valid])) if valid else 0.0,
            "mean_order_effect_interpretable": float(np.mean([c["order_effect"] for c in interp])) if interp else None,
            "mean_worst_condition_n_eff": float(np.mean([c["worst_condition_n_eff"] for c in valid
                                                          if c["worst_condition_n_eff"] is not None])) if valid else None,
            "mean_condition_level_n_eff": float(np.mean([d["n_eff"] for d in all_conditions])) if all_conditions else None,
            "mean_label_bias_tie_items": float(np.mean(tie_label_bias)) if tie_label_bias else None,
            "mean_order_effect_tie_items": float(np.mean(tie_order_effect)) if tie_order_effect else None,
            "estimated_cost_usd": round(_price(cells), 4),
            "conclusion": (
                "Real LLM judges are frequently near-deterministic on pairwise "
                "verdicts. Saturation is now gated by an exact one-sided binomial "
                "test (H0: top_prob<=sat_top_prob) rather than a bare point-estimate "
                "comparison — a raw threshold check over-flags dispersed judges from "
                "finite-sample noise. Position and label are counterbalanced "
                "(fully crossed 2x2): order_effect is a pure content-position "
                "signal, label_bias is a pure token-prior signal, previously "
                "confounded into one. Reported at TWO levels because they differ: "
                "frac_conditions_saturated counts individual (position,label) "
                "distributions; frac_cells_failing_two_sided_gate counts "
                "(model,item) cells where ANY of the 4 conditions is exact-"
                "saturated (the conservative gate for interpreting an order "
                "statistic). On a saturated distribution you still get a point "
                "verdict, but no variance and no error bars, so the QQ-style "
                "structural discriminant is non-identifying (Kang 2026's "
                "saturation caveat). Order effects are reported with Fisher "
                "exact p per cell — at this k a difference of one sample is not "
                "a measured magnitude."
            ),
        },
    }
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "pilot_" if args.pilot else ""
    out_path = out_dir / f"experiment_llm_judge_saturation_{tag}{date.today().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    s = result["summary"]
    print(f"[llm_judge_saturation] cells={s['n_cells']} "
          f"conditions_saturated={s['n_conditions_saturated']}/{s['n_conditions']} "
          f"cells_failing_gate={s['n_cells_failing_two_sided_gate']}/{s['n_cells']} "
          f"sig_raw={s['n_order_effects_significant_p05_raw']} "
          f"sig_holm={s['n_order_effects_significant_holm']} "
          f"label_bias_ties={s['mean_label_bias_tie_items']} "
          f"cost=${s['estimated_cost_usd']} wrote {out_path.name}")
    return result


if __name__ == "__main__":
    main()
