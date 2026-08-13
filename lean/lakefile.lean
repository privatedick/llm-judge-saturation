import Lake
open Lake DSL

package «llmJudgeSaturation»

-- Pinned to the exact Mathlib revision these proofs were built against, matching
-- the toolchain in `lean-toolchain` (Lean 4.29.1). `lake-manifest.json` is
-- committed so `lake exe cache get && lake build` is reproducible; an unpinned
-- `master` would drift past this toolchain and fail.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "5e932f97dd25535344f80f9dd8da3aab83df0fe6"

@[default_target]
lean_lib «EvoEcos» where

-- The "0 sorry / 0 repo-declared axiom" gate, as a build target rather than a
-- CI-only grep: `AxiomAudit.lean` imports `EvoEcos` and fails elaboration if any
-- repo declaration depends on an axiom outside Lean's standard three. Because it
-- is a default target, plain `lake build` enforces it — locally and in CI alike.
@[default_target]
lean_lib «AxiomAudit» where
