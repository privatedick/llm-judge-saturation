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
  lake build           # 0 sorry / 0 axiom on a clean build
-/

import EvoEcos.ClampedUpdateCommute
import EvoEcos.PersuasionOperator
