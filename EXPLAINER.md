# Explainer: Adaptive Opponent Modeling for Adversarial Co-Training in MARL

A self-contained walkthrough of this project, written to give you a deep,
defensible understanding before presenting. Read it top to bottom once; it builds.
Numbers are from the committed results (3 seeds, 300 eval episodes each).

---

## 0. The one sentence, and the one mental model

**One sentence:** *Model your opponent's hidden strategy with calibrated
uncertainty, and plan against that belief instead of reacting to a guess.*

**One mental model (memorize this four-link chain):**

> **Infer → Exploit → Uncertainty → Plan**

Every result in the project is one link in this chain, and each link has a number
you can defend. If you internalize nothing else, internalize the chain and its five
numbers (2.68, 4.05, 1.42, 3.07, 4.31). The rest of this doc is just explaining
each link until it's obvious.

---

## 1. Why this problem exists (the proposal context)

The umbrella proposal is **adversarial co-training**: two teams (or an agent and an
opponent) train against each other and keep adapting. The central failure mode is
**overfitting to the current opponent** — you learn a great response to *this*
opponent, then it changes strategy and your response collapses. A robust agent must
therefore (a) figure out *which* strategy it's facing, (b) know *how sure* it is,
and (c) act in a way that's robust to being wrong.

This repo doesn't try to solve the whole co-training loop at once. It isolates and
validates the **first two pieces** in a controlled game where every quantity is
measurable against ground truth:

- **Part 1** — encode the opponent's latent strategy into a *calibrated belief*.
- **Part 2** — *plan* against that belief (don't just react to a point estimate).

Naming convention used everywhere: **`red` = the opponent (the prey)**, **`blue` =
the team we control (the predators)**.

---

## 2. The task: a *hidden* opponent strategy

Most "strategy" benchmarks leak the strategy through the map (put resources in the
corners vs in a circle, and the optimal route is visually obvious). We deliberately
avoid that. Here the strategy lives **inside the opponent**, not in the
environment:

- Three predators (`blue`) chase one prey (`red`) on JaxMARL's `MPE_simple_tag_v3`
  (5 discrete actions, 25-step episodes, two fixed obstacles).
- At every episode reset, the prey is assigned a **hidden intent** `z ∈ {0,1,2,3}` —
  one of the four corners. The prey gets a small bonus for *loitering near corner
  z*, so it learns to haunt its assigned corner when it's safe.
- **The prey sees `z`. The predators do not.** The arena, dynamics, and obstacles
  are identical across intents, so a predator cannot read the strategy off a single
  frame — it must **infer it from the prey's motion over time**.
- Predators are slowed (`PRED_MAX_SPEED=0.6`) so the prey can actually express its
  strategy. (If predators were too fast, the prey would just evade and there'd be no
  strategy to infer — we have that negative on record from earlier experiments.)

> **Why this design is clever (say this):** "The strategy is an *intrinsic, hidden
> property of the opponent*, not a feature of the map. Rotating or relabeling the
> arena doesn't change the problem. That's what makes it a genuine opponent-modeling
> task rather than a perception task."

Three predator conditions are trained (they share one architecture):

| condition | what the predator gets | role |
|-----------|------------------------|------|
| **unaware** | nothing about intent | opponent-blind baseline |
| **oracle**  | the true intent `z` (one-hot) | upper bound on "knowing" |
| **belief**  | a *soft* belief over corners, of random sharpness each episode | the uncertainty-aware model |

The `belief` condition is the important one: by training on beliefs of *random
sharpness* (`intent_belief_noise=True`), the predator learns to use the belief
*proportionally to how confident it is* — to hedge when unsure and commit when sure.
That is the whole point of "uncertainty-aware."

---

## 3. Part 1 — reading the opponent (the belief)

**The claim:** the prey's hidden strategy is recoverable from its trajectory, and
how *confident* you should be is itself recoverable.

We train an encoder that reads the prey's first *k* steps and outputs a posterior
over the four corners. As *k* grows:

- **Accuracy climbs 0.37 → 0.97** over 3 → 25 observed steps.
- **Posterior entropy collapses 1.35 → 0.03 nats** (≈ "no idea" → "certain").

Intuition: early in an episode the prey is scrambling away from predators (looks the
same regardless of intent), so you genuinely can't tell. As it commits toward its
corner, the motion becomes diagnostic and the posterior sharpens. The encoder learns
the *natural* uncertainty curve of the task.

### The single most important result in the project

A predator that is **confident but wrong** is **worse than a predator that knows
nothing.** Feed the planner a confident point-estimate that's incorrect and captures
collapse to **1.42 — that's −47% versus the opponent-blind baseline (2.68).**

> **Why this matters (this is the thesis):** you cannot act on a point estimate of
> the opponent's strategy. A wrong guess, held confidently, actively hurts. You must
> carry the *full belief* and let your confidence scale your commitment. This is the
> mechanism by which opponent modeling overfits and fails in co-training — and why
> "uncertainty-aware" is in the title, not decoration.

---

## 4. Part 2 — planning against the belief

Now: what do you *do* with the belief? Two options.

- **React** (the naive way): map the belief to an action with a trained policy.
- **Plan** (our way): sample a likely intent from the belief, *imagine* where the
  prey is heading (toward that corner), and move to **intercept**. Average over a few
  sampled intents so you hedge proportionally to your uncertainty.

Formally, the planner scores candidate predator joint-actions `u` by their expected
capture value under the belief:

```
Q(s, u) = E_{z ~ belief} [ value of u against a prey pursuing corner z ]
```

and picks the best `u`. The prey's moves inside the imagination come from the actual
trained prey policy conditioned on the sampled `z`, so the rollouts are realistic.

**Result: the belief-conditioned planner reaches 4.31 captures/episode — +61% over
opponent-blind, and it beats the oracle (4.05).**

> **Anticipated pushback — "beating the oracle is suspicious."** It isn't, and here's
> the honest framing to say out loud: *the oracle is **reactive** with the true
> intent; we **plan**. Planning beats reacting even with perfect information, because
> lookahead lets you pre-position for the interception. An oracle **planner** would be
> the true ceiling — I'm comparing against the oracle **reactive policy**.*

---

## 5. The results ladder (memorize this table)

This is your centerpiece slide. Each row is designed to isolate one effect.

| `blue` predator | captures/ep | vs blind | what it isolates |
|-----------------|:-----------:|:--------:|------------------|
| confident-but-wrong (point estimate) | **1.42** | **−47%** | acting on a wrong guess *hurts* |
| opponent-blind (no intent) | **2.68** | — | the baseline |
| reactive, inferred belief | 2.82 | +5% | reacting to the belief barely helps |
| planner, **flat** belief (ablation) | 3.07 | +15% | lookahead alone, no opponent info |
| oracle, true intent (reactive) | 4.05 | +51% | the *value* of knowing the strategy |
| **planner, inferred belief (Part 1+2)** | **4.31** | **+61%** | the full method |

Two cross-row arguments you should be able to make on the spot:

1. **The belief is the source of the gain, not just the planner.** Planner with a
   *flat* (uninformative) belief gets only 3.07; planner with the *inferred* belief
   gets 4.31. So the opponent inference contributes **+40%** on top of lookahead.
   (This is the flat-belief ablation — it exists precisely to rule out "the planner is
   doing all the work.")
2. **Planning > reacting.** Reactive-with-belief is 2.82; planner-with-belief is 4.31.
   Same information, very different value — because we look ahead.

---

## 6. The JEPA extension (the "I pushed past the baseline" part)

JEPA = **Joint-Embedding Predictive Architecture** (LeCun's line, e.g. I-JEPA). The
idea: instead of **reconstructing** the input (like a VAE), **predict the
representation of the future** — via an EMA "target" encoder and a variance
regularizer, with **no decoder**. Applied to opponent modeling: encode the prey's
motion to predict *where it's going*, and throw away the unpredictable detail.

There are four sub-results. Three are wins; one is an honest negative that you should
present *on purpose*.

### 6a. Predict beats reconstruct (the encoder)
Same prey trajectories, same 2-D latent, no labels.

| encoder | probe acc | unsup. ARI |
|---------|:---------:|:----------:|
| VAE (reconstruct) | 0.54 | 0.14 |
| **JEPA (predict)** | **0.88** | **0.67** |
| supervised classifier | 0.85 | — |

> **Why predict > reconstruct here:** a prey trajectory is *mostly evasion noise plus
> a thin strategy signal* (which corner). A VAE spends its capacity reconstructing the
> noise. JEPA only has to predict what's *predictable* — where the prey heads — which
> **is** the strategy. So it keeps the signal and drops the noise, and even beats the
> supervised classifier with no labels.

### 6b. It's an *anytime* encoder (reads ~2× faster)
JEPA recovers the strategy from about **half** the observation: JEPA at ~11 steps ≈
VAE at ~20 steps. Because it predicts the *destination* instead of waiting for the
prey to arrive.

### 6c. It closes the loop *without labels*
A fully self-supervised belief — JEPA encoder + a readout to the predicted arrival
position + a softmax over the four *known* corner coordinates (no intent labels
anywhere) — drives the same planner to **4.08 captures, matching the supervised
pipeline (4.31)** with zero opponent-strategy labels. This matters because a real
opponent never hands you strategy labels; the geometry of the arena is all you're
allowed to assume.

### 6d. Where JEPA *stops* helping (the honest negative)
We also tried JEPA for the planner's **world model** — predict the next *latent*
instead of the next *state*. It **failed**: 0.57 captures, worse than the state-space
model (2.53) and even worse than reacting (2.82); its 5-step prediction RMSE was 0.24
vs the state-space model's 0.07.

> **Present this deliberately — it shows judgment.** "JEPA's *discard the unpredictable
> detail* principle is **right** for the opponent encoder (the detail is evasion noise)
> and **wrong** for a world model, which needs *accurate dynamics* — discarding detail
> throws away the physics the planner depends on. The two roles want opposite things. I
> report it straight rather than tuning it away." (Recent work — Terver et al. 2025,
> *What Drives Success in Physical Planning with JEPA World Models* — characterizes
> exactly this: latent planning only helps when the discarded detail is
> control-irrelevant.)

---

## 7. How the code is wired

The data flow, end to end:

```
simple_tag_intent.py   ──>  prey draws hidden z, emits obs (z one-hot appended for prey;
  (the environment)            soft belief appended for predators iff reveal_to_pred)
        │
mappo_teams_mlp.py     ──>  trains the three conditions (unaware/oracle/belief) via
  (MAPPO + CTDE)               MAPPO with a centralized critic; 3 seeds
        │
part2_intent_eval.py   ──>  PART 1: encoder reads first-k prey steps -> belief over corners;
                             reports accuracy + entropy + the capture ladder
        │
part2_planner.py       ──>  PART 2: belief-conditioned Monte-Carlo planner.
                             enc_proba[k]: positions -> belief  (the drop-in interface)
```

Three facts about the wiring that an advisor may probe:

1. **The intent lives in the environment's `get_obs`.** The prey always gets its
   intent one-hot; the predators get it (as a possibly-noised belief) only in the
   oracle/belief conditions. So the *only* difference between conditions is what's
   appended to the predator observation — clean and auditable.
   (`simple_tag_intent.py`, `get_obs`.)
2. **The belief is a drop-in function.** Everything downstream takes
   `enc_proba[k]: (positions) -> belief`. The supervised encoder, the JEPA label-free
   belief, and a flat belief all satisfy the same signature, so swapping them is a
   one-line change. That's how the ablations stay honest — same planner, different
   belief source.
3. **The world model shares the simulator's interface.** The learned-model planner's
   `predict(state, action) -> next_state` has the *same signature* as the simulator,
   so Part 3/4 drop straight into the Part 2 planner. This is why the Part-4 JEPA
   negative is an apples-to-apples comparison.

Key files: `src/simple_tag_intent.py` (env), `src/mappo_teams_mlp.py` (trainer),
`src/part2_intent_eval.py` (Part 1), `src/part2_planner.py` (Part 2),
`src/jepa_vs_vae_encoder.py` / `src/jepa_inference_speed.py` /
`src/part2_jepa_planner.py` / `src/part4_jepa_world_planner.py` (JEPA), and
`src/jepa_demo.py` / `src/intent_demo.py` (the GIFs).

---

## 8. Where this sits vs the published SOTA

The lab's framing is **model-based multi-agent RL**, and the baseline question (raised
by Ellen) is "multi-agent extensions of MuZero." Full annotated notes are in
[`docs/literature_review.md`](docs/literature_review.md); the short version:

| baseline (real paper) | what it does | how we relate |
|-----------------------|--------------|---------------|
| **ToMnet** (Rabinowitz 2018) | meta-learns to infer agents' goals from behavior | inference itself is *established here* — not our novelty |
| **VAE opponent model** (Papoudakis 2019) | unsupervised VAE opponent encoder | **the baseline we beat head-to-head** (JEPA 0.88 vs 0.54) |
| **SPR** (Schwarzer 2021) | self-predictive reps for sample-efficient RL | the exact machinery of our JEPA encoder, applied to *self* not *opponent* |
| **MBOM** (Yu 2021) | model-based OM; imagines opponent *policies* by recursion | closest paper to us; we infer a low-D *intent* instead |
| **MuZero / MAZero** (Schrittwieser 2020; Liu 2024) | plan with a learned world model | we plan against an inferred *belief*, not a full learned model |
| **MAMBA** (Egorov 2022) | Dreamer-style multi-agent world model | a model-based-MARL baseline to run head-to-head |
| **Equivariant nets** (van der Pol 2021) | symmetry-invariant policies | the named fix for our "encodes position, not strategy" gap |

**Our two genuine, defensible deltas:** (1) a **predictive (JEPA) opponent encoder**
that beats the published VAE-for-OM baseline and matches a supervised classifier; (2)
a **label-free belief** that matches the supervised planner. The inference is
ToMnet-style and the planning is I-POMDP-style — the *contribution* is the predictive,
label-free opponent representation plus the clean uncertainty result.

---

## 9. The honest gaps and the next steps (ranked)

You will sound much stronger if you volunteer these before you're asked.

1. **The symmetry gap (highest leverage).** The lab already diagnosed (5/13) that the
   VAE encodes *map position*, not *strategy* — rotate/reflect the trajectories and
   the strategy is unchanged but the latent moves. **JEPA does not actually fix this**
   (it discards *noise*, but still predicts *absolute positions*). The published fix is
   **equivariance** (Multi-Agent MDP Homomorphic Networks, van der Pol 2021 — its
   abstract literally says "rotating the state globally permutes the optimal joint
   policy"). The novel, unclaimed kernel: a **D₄-equivariant JEPA encoder** (the 4
   corners form a dihedral group), tested on a *held-out rotation* of the arena.
2. **No model-based-MARL baseline is actually run.** Implement **MAZero / MAMBA / MBOM**
   in this env and measure **episodes-to-recover against a novel/adaptive prey**.
   Prediction: a learned world model must re-fit (slow); our belief re-infers a 4-way
   intent in ~12 steps (fast). That turns Ellen's hypothesis into a result.
3. **The opponent is static.** The prey draws a *fixed* intent per episode. The
   frontier is *adaptive* opponents — switching intent mid-episode (a changepoint /
   tracking problem; Everett 2018) or best-responding (few-shot meta-adaptation;
   Al-Shedivat 2018). This is also where recursive reasoning (PR2) becomes the right
   comparison.

Smaller: the Part-4 negative is now *characterized* (cite Terver 2025), and everything
is one environment — porting to a JaxMARL env (SMAX) would show it's not a simple_tag
artifact.

---

## 10. Q&A — questions you'll get, and your answers

- **"Isn't beating the oracle suspicious?"** The oracle is *reactive* with the true
  intent; we *plan*. Planning > reacting even with perfect info; I compare against the
  oracle *reactive* policy and note an oracle *planner* is the true ceiling.
- **"Why is the prey a good stand-in for a real adaptive opponent?"** It draws a fresh
  hidden strategy every episode (the "opponent uses varying strategies" condition),
  controlled so I can measure inference against ground truth. Making it *adapt* is the
  next step.
- **"The corners are known — is that cheating?"** The *intent* (which corner) is
  hidden; the corner *coordinates* are arena geometry, known to anyone. The label-free
  belief uses only geometry + a self-supervised readout — no strategy labels.
- **"Why did the JEPA world model fail — does that hurt the story?"** No, it sharpens
  it: it delineates where the predictive principle helps (encoding: discard noise) vs
  hurts (world model: needs fidelity). Reporting it straight is a strength.
- **"How is this different from ToMnet / LOLA / MuZero?"** ToMnet already infers goals
  (not our novelty); LOLA *shapes* a learning opponent (different problem, the
  co-training half we haven't built); MuZero plans with a learned *world* model whereas
  we plan against an inferred *intent belief*. Our deltas: JEPA > VAE-OM, and the
  label-free belief.
- **"What's the regime caveat?"** Predators are slowed so the prey can express its
  strategy — the operating point where opponent modeling is meaningful.
- **"What's next?"** A variational continuous-`z` encoder + a latent-conditioned
  opponent policy; a MuZero-style learned model with a strategy-conditioned value; and
  closing the co-training loop so `red` and `blue` adapt to each other.

---

## 11. Cheat sheet — numbers on the tip of your tongue

- **+61%** — planner (4.31) vs blind (2.68); and it beats the oracle (4.05).
- **+40%** — the belief's own contribution (planner 4.31 vs flat-belief 3.07).
- **−47%** — a confident wrong guess (1.42); *why uncertainty matters*.
- **+51%** — value of knowing the strategy (oracle 4.05 vs blind 2.68).
- **0.37 → 0.97** — inference accuracy over 3 → 25 observed steps.
- **0.88 vs 0.54** — JEPA vs VAE encoder probe; JEPA > supervised (0.85).
- **~2×** — JEPA reads the strategy from half the steps (anytime).
- **4.08, no labels** — JEPA-belief planner matches supervised (4.31).
- **0.57** — JEPA world model (the honest negative; needs fidelity, not compression).

**If you remember one thing:** the chain — *infer → exploit → uncertainty → plan* —
and that the −47% (confident-but-wrong) is *why* the whole thing has to be
uncertainty-aware.
