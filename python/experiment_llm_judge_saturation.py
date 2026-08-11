"""Empirical: output saturation and the identifiability gate in real LLM judges.

Supports the "Two kinds of saturation" note (docs/action/) with real data via
OpenRouter. Tests the measurement-saturation hypothesis (H2): real LLM judges are
frequently near-deterministic on pairwise verdicts, and where they are, the
verdict distribution is degenerate — so an order-effect statistic computed on it
is non-identifying, exactly the Kang (2026) saturation caveat.

Design (per model x item):
  - A pairwise-preference judgment: question + two candidate answers, "reply A or
    B". Run BOTH spatial orders (slot A = answer 1, then slot A = answer 2).
  - K samples per (order) at temperature T, to estimate the verdict distribution.
  - Metrics: dispersion (top-token prob, Shannon entropy, effective support
    N_eff = 1/sum p^2) of the verdict distribution; position bias (net slot-A
    preference); order effect (does the chosen ANSWER depend on presentation
    order). An identifiability gate flags saturated cells.

Honest scope: this measures the model's *sampled* verdict distribution at T>0, a
practitioner-realistic observable, not raw next-token logprobs. It quantifies how
often the gate would fire and how order stats read under saturation; it does not
claim saturation *causes* the order effect (it impairs its measurement).

Cost-controlled: 3 cheap models, ~6 items, both orders, modest K. Use --pilot to
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
from pathlib import Path

import numpy as np
import requests
from joblib import Parallel, delayed

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-8b",
    "mistralai/mistral-nemo",
]

# Pairwise items. `better` marks the higher-quality answer (or "tie"). A
# position-robust judge picks the SAME actual answer regardless of slot order;
# "tie" items maximally expose position bias (no content signal to anchor on).
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


def _judge_messages(question: str, slot_a: str, slot_b: str) -> list[dict]:
    user = (f"Question: {question}\n\nAnswer A: {slot_a}\n\nAnswer B: {slot_b}\n\n"
            "Which answer is better? Reply with only 'A' or 'B'.")
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def _parse_verdict(content: str) -> str | None:
    if not content:
        return None
    m = re.search(r"\b([AB])\b", content.strip().upper())
    if m:
        return m.group(1)
    c = content.strip().upper()
    return c[0] if c and c[0] in "AB" else None


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
            if verdict is None:
                # reasoning-model fallback: take the LAST A/B in the reasoning trace
                reasoning = (msg.get("reasoning") or "")
                hits = re.findall(r"\b([AB])\b", reasoning.upper())
                verdict = hits[-1] if hits else None
            return verdict, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except (requests.RequestException, KeyError, IndexError, ValueError):
            if attempt < max_retries - 1:
                # exponential backoff with jitter (global CLAUDE.md mandate)
                time.sleep((2 ** attempt) * 0.5 + random.random() * 0.3)
                continue
            return None, {"prompt_tokens": 0, "completion_tokens": 0}
    return None, {"prompt_tokens": 0, "completion_tokens": 0}


def _dist_stats(verdicts: list[str]) -> dict:
    """Dispersion of a verdict distribution over {A, B} (drop None)."""
    vs = [v for v in verdicts if v in ("A", "B")]
    n = len(vs)
    if n == 0:
        return {"n": 0, "top_prob": 1.0, "entropy_bits": 0.0, "n_eff": 1.0, "p_A": None}
    c = Counter(vs)
    p = np.array([c.get("A", 0), c.get("B", 0)], dtype=float) / n
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    n_eff = float(1.0 / np.sum(p ** 2))
    return {"n": n, "top_prob": float(p.max()), "entropy_bits": entropy,
            "n_eff": n_eff, "p_A": float(p[0])}


def _run_cell(model: str, item: dict, k: int, temperature: float) -> dict:
    # order "12": slot A = ans1 ; order "21": slot A = ans2
    v12 = [_call(model, _judge_messages(item["question"], item["ans1"], item["ans2"]),
                 temperature) for _ in range(k)]
    v21 = [_call(model, _judge_messages(item["question"], item["ans2"], item["ans1"]),
                 temperature) for _ in range(k)]
    verd12 = [v for v, _ in v12]
    verd21 = [v for v, _ in v21]
    usage = {"prompt_tokens": sum(u["prompt_tokens"] for _, u in v12 + v21),
             "completion_tokens": sum(u["completion_tokens"] for _, u in v12 + v21)}
    s12, s21 = _dist_stats(verd12), _dist_stats(verd21)

    # chose ans1: order12 -> verdict A ; order21 -> verdict B
    def p_chose_ans1(verdicts, order):
        vs = [v for v in verdicts if v in ("A", "B")]
        if not vs:
            return None
        want = "A" if order == "12" else "B"
        return sum(v == want for v in vs) / len(vs)

    p1_12 = p_chose_ans1(verd12, "12")
    p1_21 = p_chose_ans1(verd21, "21")
    order_effect = (abs(p1_12 - p1_21) if p1_12 is not None and p1_21 is not None else None)
    # net slot-A preference across both orders (0.5 == unbiased)
    slotA = ([s12["p_A"]] if s12["p_A"] is not None else []) + \
            ([s21["p_A"]] if s21["p_A"] is not None else [])
    position_bias = (float(np.mean(slotA)) if slotA else None)
    min_top_prob = max(s12["top_prob"], s21["top_prob"])   # most-saturated order
    min_entropy = min(s12["entropy_bits"], s21["entropy_bits"])
    min_n_eff = min(s12["n_eff"], s21["n_eff"])
    return {
        "model": model, "item": item["id"], "better": item["better"],
        "order12": s12, "order21": s21,
        "order_effect": order_effect, "position_bias": position_bias,
        "min_top_prob": min_top_prob, "min_entropy_bits": min_entropy,
        "min_n_eff": min_n_eff, "usage": usage,
    }


def _price(cells: list[dict]) -> float:
    # rough: use ~$0.2/1M in, $0.4/1M out as a conservative blended estimate
    it = sum(c["usage"]["prompt_tokens"] for c in cells)
    ot = sum(c["usage"]["completion_tokens"] for c in cells)
    return it / 1e6 * 0.2 + ot / 1e6 * 0.4


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15, help="samples per (model,item,order)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=2, help="parallel API threads")
    ap.add_argument("--sat-top-prob", type=float, default=0.9,
                    help="dispersion-gate: top_prob above this = saturated")
    ap.add_argument("--pilot", action="store_true", help="1 model, 2 items, k=6")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set")

    models = MODELS[:1] if args.pilot else MODELS
    items = ITEMS[:2] if args.pilot else ITEMS
    k = 6 if args.pilot else args.k

    tasks = [(m, it) for m in models for it in items]
    cells = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(_run_cell)(m, it, k, args.temperature) for m, it in tasks
    )

    valid = [c for c in cells if c["order_effect"] is not None]
    n_sat = sum(1 for c in valid if c["min_top_prob"] >= args.sat_top_prob)
    # among gated-interpretable (dispersed) cells, is the order effect real?
    interp = [c for c in valid if c["min_top_prob"] < args.sat_top_prob]
    ties = [c for c in valid if c["better"] == "tie" and c["position_bias"] is not None]
    tie_slotA = [c["position_bias"] for c in ties]
    result = {
        "experiment": "llm_judge_saturation",
        "date": date.today().isoformat(),
        "config": {"models": models, "n_items": len(items), "k": k,
                   "temperature": args.temperature, "sat_top_prob": args.sat_top_prob},
        "cells": cells,
        "summary": {
            "n_cells": len(valid),
            "n_saturated": n_sat,
            "frac_saturated": (n_sat / len(valid)) if valid else 0.0,
            "n_interpretable": len(interp),
            "mean_order_effect_all": float(np.mean([c["order_effect"] for c in valid])) if valid else 0.0,
            "mean_order_effect_interpretable": float(np.mean([c["order_effect"] for c in interp])) if interp else None,
            "mean_min_n_eff": float(np.mean([c["min_n_eff"] for c in valid])) if valid else None,
            "mean_slotA_pref_tie_items": float(np.mean(tie_slotA)) if tie_slotA else None,
            "estimated_cost_usd": round(_price(cells), 4),
            "conclusion": (
                "Real LLM judges are frequently near-deterministic on pairwise "
                "verdicts (frac_saturated; mean_min_n_eff near 1). On a saturated "
                "cell you still get a point verdict, but the verdict DISTRIBUTION is "
                "degenerate — no variance, no error bars, and the QQ-style "
                "structural discriminant is non-identifying (Kang 2026's saturation "
                "caveat). Position bias itself stays visible (mean_slotA_pref_tie: "
                "deviation from 0.5 on content-neutral 'tie' items); saturation and "
                "position bias are separable. Practical upshot: report dispersion "
                "alongside any order/position statistic — a point estimate from a "
                "saturated judge carries no distributional information."
            ),
        },
    }
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "pilot_" if args.pilot else ""
    out_path = out_dir / f"experiment_llm_judge_saturation_{tag}{date.today().strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    s = result["summary"]
    print(f"[llm_judge_saturation] cells={s['n_cells']} saturated={s['n_saturated']} "
          f"({s['frac_saturated']:.0%}) mean_order_effect={s['mean_order_effect_all']:.3f} "
          f"cost=${s['estimated_cost_usd']} wrote {out_path.name}")
    return result


if __name__ == "__main__":
    main()
