# Two kinds of saturation

*Why an LLM judge flips when you swap the options — and why measuring that flip
is subtle. One name, two mechanisms: saturation can **manufacture** an order
effect, and, separately, it can **hide** one.*

---

If you have used an LLM as a judge — "which of these two answers is better?" —
you have probably watched it flip its verdict when you swap the order of the two
candidates. The standard fix is to run both orders and check that they agree. It
works well enough. But it answers neither of the questions underneath it: **why**
does order matter at all, and why is **measuring** the effect surprisingly easy
to get wrong?

The two questions have two *different* answers, and keeping them apart is the
whole point of this note. Roadmap:

1. Order effects require non-commuting, state-dependent transformations.
2. *Internal* saturation (a clamped accumulator) is one thing that can **create**
   that state dependence — provably.
3. *Output* saturation (near-deterministic model outputs) is a different thing
   that **destroys the observability** needed to measure it.
4. A small perturbation bound ties the "how far can order/framing move a
   decision" question to the same state→transformation→observation picture.

The unifying object is not "saturation." It is loss of commutativity — whether
from non-commuting operators or from an update whose effective increment becomes
state-dependent — plus a separate loss of distinguishability at the observation.
Same geometry of *state, transformation, observable* — not one physical
mechanism.

## Order effects are non-commuting operations

Write `T_A`, `T_B` for the operations of processing item A, item B. An order
effect is simply `T_B(T_A(x)) ≠ T_A(T_B(x))`: the operations don't commute, so
which one runs first changes the state the second one acts on. That can happen
two ways: the operators themselves may be linear but non-commuting (projectors,
below, are the standard example — `AB ≠ BA` holds regardless of the state they
act on), or an update that is *additive with a constant increment* may become
non-commuting by having that increment start depending on the current state.
Constant additive updates alone (`x ↦ x + δ` for fixed `δ`) always commute —
translations compose order-free — so it takes something like a clamp turning
the effective increment into a function of state to make an *additive* update
non-commuting (next section).

Quantum cognition models this with vectors and projectors, and offers a
parameter-free structural invariant, the **QQ equality** (Wang & Busemeyer 2013;
Wang et al. 2014). Writing `A_y` for "answered A yes," it reads

> `p(A_y, B_n) + p(A_n, B_y) = p(B_y, A_n) + p(B_n, A_y)`

— the two orders must agree on the total "changed my mind" probability, however
large the individual order effect is. It is a sharp test for whether observed
order effects show the *specific* regularity the projective model predicts.
Two honest caveats: satisfying QQ does **not** establish the Hilbert-space model
(Boyer-Kassem et al. 2016 — the same data can satisfy QQ while violating other
constraints of that model), and it certainly is not evidence of "quantum
cognition" as an ontology. Treat QQ as one structural invariant among several,
not as a structure-vs-noise oracle.

## Internal saturation can *create* non-commutation

Here is the interesting part, and it needs no quantum framing at all. Take a
bounded belief update: a value that moves by `±δ` but is clamped to `[lo, hi]`,
i.e. `x ↦ clamp(x + δ)`. The clamp is nonlinear, so composition becomes
state-dependent, and that is exactly what non-commutation needs.

The precise lemma (Lean, `interior_commute`, 0 sorry): two clamped increments
commute whenever *each single step* keeps the value inside `[lo, hi]` — their
**sum may still saturate**, and both orders then land on the same clamped value.
Informally: if each piece of evidence alone keeps the belief in band, order does
not matter even if the total does not fit — and this isn't a rare edge case:
sampling random `(x, a, b)` triples where each single step lands in-band,
roughly a third to a tenth of them (the exact fraction is a property of the
sampling distribution, not of the theorem — checked independently at both
ends) have the *combined* step `x+a+b` land outside `[lo, hi]` anyway, with
commutation holding regardless every time, as the proof guarantees.
Contrapositive: if two orders
disagree, **some single step left the band**. Boundary
contact is *necessary* for non-commutation — but not *sufficient*: same-sign
increments that all saturate to the same bound stay order-independent despite
hitting it (start `0.9`, `+0.2` then `−0.2` → `0.8`; the other order → `0.9`, but
`+0.2, +0.2` → `1.0` either way). So the clean, one-directional statement is
*interior ⇒ order-free*, and the mechanism is: **a nonlinear boundary makes the
update state-dependent, which permits non-commutation.** A thermostat with a
capped integrator has order effects for the same reason — no minds required.

## Output saturation *destroys the ability to measure* one

Now the LLM judge, and this is a genuinely different mechanism. Kang (2026) ran
the QQ-equality audit on an LLM judge and hit a wall: although every
pre-specified health gate passed, the model's next-token distributions were
near-deterministic — **saturated** — for 17 of 18 item pairs under the
direct-evaluation framing (and 7 of 8 under a persona framing) — the audit was
healthy by its own pre-registered criteria and the measurement still failed.
When almost all probability mass sits on one token, many distinct latent states
map to the same observable, and the order statistic you compute no longer
identifies anything.

Note what this is *not*: it does not show that saturation *caused* the order
effect. It shows that saturation *impaired the audit's ability to assess the
structure*. Internal saturation changes the dynamics; output saturation collapses
the observation. Both reduce distinguishability, which is why they share a name —
but one creates an effect and the other hides it.

To put numbers on it, I ran three open judges (deepseek-v4-flash, qwen3-8b,
mistral-nemo) on six pairwise items, a fully crossed content-**position** ×
**label-assignment** design (four conditions per item — decoupling *where* an
answer sits from *which letter* it is called; see "Honest limits" below for why
that matters), 40 samples per condition **at temperature 1.0** — full sampling
temperature, not the T→0 regime where saturation would be a trivial artifact of
the setup rather than a finding about the judge (via OpenRouter, ~$0.21). Saturation is
flagged by an exact one-sided binomial test against the 0.9 floor, not a bare
point-estimate comparison — the gate box below explains why the raw comparison
over-flags. **49 of 72 (position, label) distributions were confidently
saturated**, 14 of 18 cells fail the two-sided gate (any of their 4 conditions
saturated), mean effective support `N_eff ≈ 1.20`. On the four items with a
clearly better answer, every judge collapses onto the correct one: benign
saturation.

Decoupling position from label changes the *finding*, not just the numbers.
**Zero of eighteen cells show a content-position order effect surviving
Holm-Bonferroni correction** — genuine position-driven order effects are small
(mean 0.058 on ties) and don't survive multiple-comparison correction at this
k. One cell (qwen3-8b on `tie_paraphrase`) does cross the nominal p<0.05 line
uncorrected (Fisher exact p=0.0049) — with 18 tests at alpha=0.05, ~1 such hit
is exactly what chance predicts, and Holm correction removes it, which is the
textbook reason the corrected count is the one worth quoting, not the raw one.
This result **replicated independently**: a second stochastic run (same code,
fresh samples, temperature 1.0 — `results/` has both raw JSON files) landed on
the same qualitative finding — 0/18 Holm-significant both times — despite the
individual numbers moving between runs, as real sampling noise does. What *is*
large and consistent across both runs is a pure **label/token preference**,
separable from position for the first time *in this measurement series* because
the design crosses the two. Kang's v2 arrives at the same requirement from the
other direction — label assignment there "materially changed several
mapping-specific QQ verdicts," and the paper closes by arguing that saturation
screening *and label counterbalancing* should precede structural
interpretation. Two designs converging on that is better evidence it matters
than either alone; the claim here is not priority, it is that the crossed
design makes the label effect a number rather than a caveat. On tie items,
the judge's raw rate of answering "A" — pooled across content and position, so
"A" is attached to each answer and each slot equally often — deviates from the
unbiased 0.5 by a mean of 0.183, roughly three times the size of the position
effect (3.1× in this run, 3.0× in the replication), and by as much as 0.39 for
one judge (deepseek-v4-flash on
`tie_paraphrase`: P(says "A") = 0.11, essentially the mirror image of
mistral-nemo's 0.76 on the same item — two different judges, two different and
large label preferences, in opposite directions). The earlier design could not
tell any of this apart — position and label were perfectly confounded, and the
caveat box below named "label asymmetry" as a possibility without a way to test
it. This run tests it: what looked like an order effect was, substantially, the
judge's opinion of the *letter*, not the *slot*.

The point verdict survives saturation; the distribution — what a QQ-style
structural test or a confidence interval needs — does not.

> **The practical consequence — an identifiability gate.** Interpret an
> order-effect statistic only if the output distribution clears a dispersion
> floor: token entropy above a threshold, top-token probability below one, or an
> effective support size `N_eff = 1 / Σ pᵢ² > N_min` are three equivalent ways to
> state the same floor (`experiment_llm_judge_saturation.py` implements the
> top-token-probability form; the numbers above are gated on that one). A large
> measured "order effect" under low identifiability is not the same as one under
> high identifiability, and should not be reported as though it were. A confident
> bias score from a saturated judge is **non-identifying**, not necessarily a real
> effect — it is consistent with genuine strong preference, label asymmetry,
> prompt-induced determinism, or simply insufficient resolution. The gate itself
> also needs calibration: with finite `k`, a point estimate of top-token
> probability crossing 0.9 is not the same as being confidently above 0.9 — the
> implementation tests that with an exact one-sided binomial test rather than a
> bare threshold comparison, for exactly the reason the saturation counts above
> need it.

## A bound on how far framing can move a decision

The same state→transformation→observation picture gives a clean sensitivity
result. Model "framing" as a rotation `θ` of the decision state before the choice
is read, so `P(yes) = cos²(φ)` becomes `cos²(φ + θ)`. By a product-to-sum
identity the shift is exactly

> `cos²(φ + θ) − cos²(φ) = −sin(2φ + θ)·sin θ`,

so `|ΔP| ≤ |sin θ|`, tight when `|sin(2φ+θ)| = 1` (identity and bound Lean-proved,
0 sorry). A bounded framing budget caps how far a decision can be moved, by the
sine of that budget — a low-observability decision map, the second failure mode
alongside non-commuting transformations.

For a threshold decision (yes iff score ≥ ½) **in this 2-D projective model**,
framing is harmless whenever the perturbation does not cross the switching
surface `S`: with robustness margin `m(x) = dist(x, S)`, the decision is
invariant while `‖θ‖ < m(x)`. This is a property of *this* model, not a universal
law about sign-based decisions — but within it, robustness is exactly distance to
the surface.

## The through-line

The common object is not saturation itself but **loss of commutativity** —
whether from non-commuting linear operators or from an additive update whose
effective increment becomes state-dependent — together with a separate **loss
of distinguishability at the observation**. In the clamped accumulator, a
boundary creates the state dependence. In the projective model, non-commuting
measurements create it directly, with no state-dependent increment involved at
all. In an LLM audit, output saturation does something else
entirely: it collapses distinguishable states into nearly identical observables,
making any order effect hard to *identify*. Three cases, one geometry of state /
transformation / observability — not a single physical mechanism.

## Honest limits

The validation here is synthetic and the bounds are for a 2-D projective model;
the "quantum" is a modeling vocabulary, not a claim about the physics in your
GPU. This is a conceptual separation plus a few proved bounds — not a product,
not a judge-fixing tool. The saturation numbers above are a small study (three
models, six items) — enough to show the effect is real and common, not enough to
be representative; a proper study would sweep more judges, items, and a standard
preference set. And one analytic check remains open: whether boundary proximity
actually raises the commutator in the toy dynamics.

The rest of this section is the note's own error log, in the order the errors
were found. An earlier version of this measurement reported something far more
dramatic: a judge flipping its verdict with order on 83% of samples. **It did
not replicate.** Two things were wrong. The verdict parser uppercased the text
before matching
`\b([AB])\b`, so the English article "a" could match as a verdict of *A* —
biasing toward the slot-A answer, which is to say manufacturing the very effect
being measured. And at k=12 a difference of one sample looks like an effect.
Both are fixed here (case-sensitive parsing; per-cell Fisher exact with
Holm-Bonferroni; raw per-sample verdicts persisted in `results/` so the parse can
be re-audited). The episode is this note's own argument demonstrated on its
author: absent a dispersion-and-significance gate, a confident-looking bias
number was an artifact.

A second, subtler version of the same lesson showed up one round later. The
saturation *gate* itself was a bare `top_prob >= 0.9` point-estimate comparison
— exactly the kind of ungated threshold this note argues against for order
effects, just applied one level up, to the measurement of saturation instead of
the measurement of order. And position and label were confounded (the
first-presented answer was always labeled "A"), so a judge that flips on a tie
could be following spatial position or a raw token prior for "A"/"B" —
indistinguishable. Both are fixed here too (an exact one-sided binomial test
against the threshold; a fully crossed position × label design), and the fix
changed the finding, not just the precision: the "order effect" the first run
reported was substantially a label preference, not a position preference — see
above.

A third round switched from reading the code to executing it: installing the
dependencies, running the parser against realistic model completions, and
property-testing the Lean statements numerically. Every claim was checked
before being acted on — not all of them held up. Two real, narrow bugs
confirmed and fixed: a public function (`s_odd([])` in `qq_cbd.py`) silently
returned a sentinel instead of raising on degenerate input, and the parser's
fallback took the *first* verdict letter in content rather than the last, so
"I think A is worse, B is better" parsed as A instead of the actual verdict B
— fixed by taking the last match, consistent with the reasoning-fallback path
a few lines below. But most of that round's other claimed reproductions did
NOT hold when actually run: three of four cited parser-bug examples returned
the correct letter against the code as shipped, and a claimed missing
length-check in `cbd_cyclic` was already there and already raised correctly.
Neither this note nor the code changed on the strength of an unverified claim
— every fix above was applied only after reproducing the failure directly,
the same discipline this note itself argues for. The measurement was re-run
with the real fix live; `results/` has both stochastic runs, and the core
finding (0/18 Holm-significant) replicated across them.

Then an external review (2026-08-12) asked for two extensions specifically:
proprietary frontier judges, and the interaction between sampling temperature
and measured saturation. One extended cleanly, one didn't. Added
`openai/gpt-5-mini` — a proprietary frontier judge, not open-weight like the
three above — on the two tie items plus one clear-winner item, same design
(k=40, T=1.0, ~$0.04): it reproduces the same qualitative pattern. `capital`
(clear winner) saturates correctly onto the right answer, same as the
open-weight judges. On the two ties, content-position order effects stay near
zero (0.012 both) while label bias is real and item-specific: 0.49
(essentially unbiased) on `tie_paraphrase`, 0.11 (strongly "B"-preferring) on
`tie_greeting` — the same judge, two very different biases depending on the
specific prompt, which is more evidence for "item-specific token quirk," not
"universal position bias." The temperature sweep did not extend as cleanly:
two attempts to run judges across T ∈ {0, 0.5, 1} both stalled mid-run —
twice to the same cause, an OpenRouter connection that outlived its own
timeout, because `requests`' timeout is a per-read inactivity timer rather
than a wall-clock deadline and a trickling connection never lets it fire (the
code now enforces a hard deadline; it did not exist in time to save those
runs) — and were abandoned rather than reported from a killed, incomplete
run. The temperature question stays genuinely open — this note does not claim
an answer it doesn't have.

If you take one thing away:

> **Gate your order-effect statistics on identifiability. A bias number from a
> saturated judge measures the interface, not the judge.**


## What's in this repository

Everything needed to check the claims above.

| Path | What it is |
|---|---|
| `lean/EvoEcos.lean` | Root module importing both proof files — what `lake build` and CI actually build. |
| `lean/AxiomAudit.lean` | Machine-checked backing for the "0 sorry / 0 axiom" claim, as a default build target rather than a CI-only grep — `lake build` fails if any declaration depends on an axiom outside Lean's standard three. |
| `lean/EvoEcos/ClampedUpdateCommute.lean` | `interior_commute` (interior clamped updates commute) + a boundary witness. 0 sorry / 0 axiom. |
| `lean/EvoEcos/PersuasionOperator.lean` | `power_eq`, `power_le_abs_sin` (the `\|sin θ\|` bound), `power_tight`. 0 sorry / 0 axiom. |
| `python/experiment_llm_judge_saturation.py` | The dispersion/order-effect measurement across judges (OpenRouter). `--models`/`--item-ids`/`--temperatures` let you target a subset or sweep temperature without disturbing the headline run. |
| `python/qq_cbd.py` | QQ equality, order-sensitivity, and the cyclic-system CbD functional. |
| `python/test_qq_cbd.py`, `python/test_experiment_llm_judge_saturation.py` | Unit tests, against distributions/completions with known structure (`pytest python/`). |
| `python/test_gate_power.py` | Empirically confirms the exact-binomial gate's power floor — the smallest `k` at which it can flag saturation at all (k=29 at the default `--sat-top-prob 0.9` / `--alpha 0.05`). |
| `python/test_hardening.py` | Spins up a real trickling TCP server to prove the hard wall-clock deadline actually cuts off a hung OpenRouter call, and that a saturated thread pool can't fake one. |
| `python/pyproject.toml` | `pip install -e python/` for a real package instead of a script. |
| `.github/workflows/ci.yml` | Runs the Python tests, the PoC + its tests, and the Lean build (0 sorry / 0 axiom check) on every push. |
| `results/` | Raw JSON from all runs reported above (14/18 cells fail the two-sided gate in both independent 3-model runs; 0/18 order effects survive Holm-Bonferroni in both; the `openai/gpt-5-mini` frontier-coverage addendum). |
| `poc/poc_llm_judge_saturation.py` | Standalone, offline (no API key) demo of the position×label decomposition and calibrated gate against synthetic judges — a faster way to see the method work than reading the prose. Read its own docstring before citing the cost numbers: those model a sampling strategy this repo doesn't implement yet. |
| `poc/test_poc_llm_judge_saturation.py` | Asserts the properties the PoC's docstrings claim: the saturation knob is monotonic (not inert or inverted) in every regime, `identifiable` tracks the calibrated gate rather than the raw `top_prob>=0.9` heuristic, the regime classifier is exhaustive and doesn't swallow the ideal judge into a dead band, and the cost model matches the real script's ~2x and refuses a sub-power-floor pilot. |
| `CITATION.cff` | Machine-readable citation metadata. |

**See the method work, no API key, in under a second** — pure standard-library
Python, synthetic judges, the same calibrated gate and position×label
decomposition the real script uses:

```bash
python poc/poc_llm_judge_saturation.py       # 0 dependencies, offline
python -m pytest poc/ -v                     # the knob/gate/classifier properties, asserted
```

Read the PoC's own docstring before citing its cost numbers — one row models
a sampling strategy this repo doesn't implement yet (flagged explicitly in
the script's own output).

**Reproduce the measurement** (~$0.16 at the default `--k 30`, ~$0.21 at the
`--k 40` used above; cost is linear in `k`, so `--k 12` would be ~$0.06):

```bash
pip install -r python/requirements.txt
export OPENROUTER_API_KEY=...
python python/experiment_llm_judge_saturation.py          # --k defaults to 30
```

**Do not lower `--k` below 29.** The exact binomial gate's smallest attainable
p-value at `k` samples is `sat_top_prob ** k` — even a perfectly deterministic
`k`-of-`k` cell. At the default `--sat-top-prob 0.9` / `--alpha 0.05` that
crosses 0.05 only at `k = 29`, so at any smaller `k` the gate reports
`saturated_exact = False` for *every* possible outcome, including a completely
deterministic judge. An earlier version of this README suggested `--k 12` as a
cheap quick-start; at that `k` the gate is silently powerless and a reader
following it would see `frac_conditions_saturated = 0.0` and reasonably (but
wrongly) conclude the headline result doesn't replicate. `python/test_gate_power.py`
pins this down empirically. `--pilot` (1 model, 2 items, k=6) is a cheap
end-to-end smoke test of the plumbing, not a small version of the result — it
is far below the power floor and will report no saturation whatever the judge
does. Use `--k 40` to reproduce the numbers quoted above exactly.

**Check the proofs:**

```bash
cd lean && lake exe cache get && lake build
```

Lean 4.29.1 + Mathlib, pinned in `lake-manifest.json`. A clean build emitting no
`declaration uses 'sorry'` warning is the actual check — note that grepping for
the string "sorry" matches the header comments, so it is not a useful test.

---

*References: Wang & Busemeyer (2013), "A Quantum Question Order Model…," Topics in
Cognitive Science 5(4), 689–710; Wang, Solloway, Shiffrin & Busemeyer (2014),
"Context effects produced by question orders reveal quantum nature of human
judgments," PNAS 111(26), 9431–9436; Boyer-Kassem, Duchêne & Guerci (2016),
"Testing quantum-like models of judgment for question order effect," arXiv:1501.04901
(QQ satisfaction does not establish the Hilbert model); Pilsung Kang (Dankook
University, 2026), "Auditing Question-Order Effects in Large Language Models with
the QQ Equality: Mechanism Characterization and a Saturation Caveat,"
arXiv:2607.17219 (recent — verify at arxiv.org/abs/2607.17219). The
clamped-update commutation and framing-power bounds are Lean-formalized (0 sorry
/ 0 axiom) and included in this repository.*
