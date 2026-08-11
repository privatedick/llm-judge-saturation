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

The unifying object is not "saturation." It is loss of commutativity under a
state-dependent transformation, plus a separate loss of distinguishability at the
observation. Same geometry of *state, transformation, observable* — not one
physical mechanism.

## Order effects are non-commuting operations

Write `T_A`, `T_B` for the operations of processing item A, item B. An order
effect is simply `T_B(T_A(x)) ≠ T_A(T_B(x))`: the operations don't commute, so
which one runs first changes the state the second one acts on. That requires the
transformations to be **state-dependent** — a linear, state-independent update
always commutes.

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
not matter even if the total does not fit. Contrapositive: if two orders
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
the QQ-equality audit on an LLM judge and hit a wall: the model's next-token
distributions were near-deterministic — **saturated** — for 17 of 18 item pairs.
When almost all probability mass sits on one token, many distinct latent states
map to the same observable, and the order statistic you compute no longer
identifies anything.

Note what this is *not*: it does not show that saturation *caused* the order
effect. It shows that saturation *impaired the audit's ability to assess the
structure*. Internal saturation changes the dynamics; output saturation collapses
the observation. Both reduce distinguishability, which is why they share a name —
but one creates an effect and the other hides it.

To put numbers on it, I ran three open judges (deepseek-v4-flash, qwen3-8b,
mistral-nemo) on six pairwise items, both orders, 40 samples per order (via
OpenRouter, ~$0.11). **29 of 36 order-conditioned distributions were
near-deterministic** — top verdict probability ≥ 0.9, mean effective support
`N_eff ≈ 1.17` where two outcomes allow up to 2. On the four items with a clearly
better answer, every judge collapses onto the correct one: benign saturation.

The order effects concentrate entirely on the two content-ties. Three of eighteen
cells show an effect surviving Holm-Bonferroni correction (Fisher exact,
p = 0.0007–0.0019, magnitude 0.25–0.33), and **all three are
dispersion-asymmetric**: one presentation order leaves the judge genuinely
uncertain (`N_eff` 1.6–2.0) while the other collapses it to near-determinism
(`N_eff` 1.0–1.2). The two cells where *both* orders are dispersed show no
significant effect at all (p = 0.65, 0.78). So the order effect here is not a
uniform slot preference — the mean slot-A rate on ties is 0.47, i.e. no global
position bias — but an asymmetry in *how certain* the judge becomes depending on
presentation order. A single reported "position-bias %" hides that entirely.

The point verdict survives saturation; the distribution — what a QQ-style
structural test or a confidence interval needs — does not.

> **The practical consequence — an identifiability gate.** Interpret an
> order-effect statistic only if the output distribution clears a dispersion
> floor: token entropy above a threshold, top-token probability below one, or an
> effective support size `N_eff = 1 / Σ pᵢ² > N_min`. A large measured "order
> effect" under low identifiability is not the same as one under high
> identifiability, and should not be reported as though it were. A confident bias
> score from a saturated judge is **non-identifying**, not necessarily a real
> effect — it is consistent with genuine strong preference, label asymmetry,
> prompt-induced determinism, or simply insufficient resolution.

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

The common object is not saturation itself but **loss of commutativity under a
state-dependent transformation**, together with a separate **loss of
distinguishability at the observation**. In the clamped accumulator, a boundary
creates the state dependence. In the projective model, non-commuting measurements
create it directly. In an LLM audit, output saturation does something else
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
actually raises the commutator in the toy dynamics. An earlier version of this measurement reported something far more dramatic: a
judge flipping its verdict with order on 83% of samples. **It did not replicate.**
Two things were wrong. The verdict parser uppercased the text before matching
`\b([AB])\b`, so the English article "a" could match as a verdict of *A* —
biasing toward the slot-A answer, which is to say manufacturing the very effect
being measured. And at k=12 a difference of one sample looks like an effect.
Both are fixed here (case-sensitive parsing; per-cell Fisher exact with
Holm-Bonferroni; raw per-sample verdicts persisted in `results/` so the parse can
be re-audited). The episode is this note's own argument demonstrated on its
author: absent a dispersion-and-significance gate, a confident-looking bias
number was an artifact.

If you take one thing away:

> **Gate your order-effect statistics on identifiability. A bias number from a
> saturated judge measures the interface, not the judge.**


## What's in this repository

Everything needed to check the claims above.

| Path | What it is |
|---|---|
| `lean/EvoEcos/ClampedUpdateCommute.lean` | `interior_commute` (interior clamped updates commute) + a boundary witness. 0 sorry / 0 axiom. |
| `lean/EvoEcos/PersuasionOperator.lean` | `power_eq`, `power_le_abs_sin` (the `\|sin θ\|` bound), `power_tight`. 0 sorry / 0 axiom. |
| `python/experiment_llm_judge_saturation.py` | The dispersion/order-effect measurement across judges (OpenRouter). |
| `python/qq_cbd.py` | QQ equality, order-sensitivity, and the cyclic-system CbD functional. |
| `python/test_qq_cbd.py` | Unit tests for the above, against distributions with known structure (`pytest python/`). |
| `results/` | Raw JSON from the run reported above (16/18 cells near-deterministic). |

**Reproduce the measurement** (~$0.03):

```bash
pip install -r python/requirements.txt
export OPENROUTER_API_KEY=...
python python/experiment_llm_judge_saturation.py --k 12
```

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
