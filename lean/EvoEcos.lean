/-
llm-judge-saturation: formal companions to "Two kinds of saturation"
=====================================================================

Two independent Lean modules backing the essay's claims:

  ClampedUpdateCommute — `interior_commute`: two clamped belief updates
    commute whenever each single step stays inside `[lo, hi]`, even if
    their sum would saturate. Plus a boundary witness showing
    non-commutation is possible once a boundary is actually hit.

  PersuasionOperator — `power_eq`, `power_le_abs_sin`, `power_tight`: a
    tight bound on how far a rotation ("framing") of a 2-D projective
    decision state can move `P(yes)`.

Usage:
  lake exe cache get   # fetch prebuilt Mathlib (this repo pins a Mathlib rev)
  lake build           # builds the proofs AND runs the axiom audit

`lake build` is the gate, not just a build: `AxiomAudit.lean` is a default
target, so the same command fails if any declaration in these modules depends on
an axiom outside Lean's standard three — which is what catches a stray `sorry`
(it shows up as `sorryAx`) in any syntactic position. Lean itself only *warns*
on `sorry` and would otherwise exit 0.
-/

import EvoEcos.ClampedUpdateCommute
import EvoEcos.PersuasionOperator
