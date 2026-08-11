import Lake
open Lake DSL

package «llmJudgeSaturation»

-- Built against Mathlib for the toolchain in `lean-toolchain` (Lean 4.29.1).
-- Pin `@ "<rev>"` if master has drifted past that toolchain.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «EvoEcos» where
