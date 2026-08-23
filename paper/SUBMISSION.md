# arXiv submission checklist (operator-gated)

The paper compiles clean (pdflatex, 0 errors, 2 passes). Everything below is
the operator's half; the agent can prepare but not submit.

1. **arXiv account** — register as `privatedick`. ⚠️ DO NOT link ORCID: the
   ORCID record is public and carries the legal name (known trap, see
   project memory 2026-08-17). arXiv works fine without ORCID.
2. **Endorsement** (the open gate) — cs.CL submissions from a new account
   need an endorsed author. Any active cs.CL/arXiv endorser works; the
   endorsement is about the *substance* being plausibly on-topic, which this
   paper is. Target category: cs.CL primary, cs.LG cross-list.
3. **Submit** `paper/` (source + this README's numbers are all traceable to
   `results/*.json`). License: arXiv non-exclusive; code stays MIT.
4. **After posting** — the Corrigibility prize manager monitors arXiv (among
   LW/AF); posting by mid-September 2026 is inside the ~$40k end-Sep window
   (registered prediction P3, docs/funding/predictions_2026-08-23.md).

Numbers provenance (all from repo artifacts, not memory):
- Tables: results/experiment_llm_judge_saturation_20260811.json,
  _20260812.json, _frontier_20260812.json
- k=29 power floor: python/test_gate_power.py (verifies against the real
  exact test)
- Lean theorem names: lean/EvoEcos/ClampedUpdateCommute.lean,
  PersuasionOperator.lean
- Kang caveat: README references, arXiv:2607.17219
