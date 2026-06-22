# Methods notes (for the 6/12 questions)

Three things the meeting asked to pin down: how the latent scatter is generated
and what its axes are, how the belief planner works and the kinds of belief it
uses, and what the bar plot on the self-supervised encoder figure means.

---

## 1. The latent scatter — what the axes are, how it is made

File: `mopa/experiments/latent_resources.py` (+ `mopa/encoders.py`, `mopa/data.py`).
Figure: `plots/mopa_latent_resources_mappo_prey_occ.png`.

Pipeline, end to end:

1. **Roll out the two specialists.** The circle-trained prey and the
   corners-trained prey are each run in their native environment against the
   weakened predators they were trained with (`PRED_MAX_SPEED=0.6`), 600
   episodes each (1,200 total), pooled over 3 checkpoint seeds. We keep the
   prey's absolute (x, y) positions per episode. Absolute, not
   init-conditioned: the placement signal *is* where the prey routes (circle
   ring r≈0.6 vs corner dashes r≈1.13), and the old exp4/exp5 init-conditioning
   removed exactly that.

2. **Featurize one vector per episode.** Two options:
   - *window*: the first k steps flattened (later steps zero-masked).
   - *occupancy*: an 8×8 normalized histogram of where the prey spent its time
     (this is the one that carries the signal — a single trajectory window is at
     chance even for a supervised probe). Each feature is then z-scored.

3. **Train two encoders, unsupervised (no placement labels ever).**
   - **VAE**: encode → sample a latent → *reconstruct* the input (ELBO). The
     plotted latent is the posterior mean μ.
   - **JEPA**: encode the context, then *predict the representation* of the full
     window produced by an EMA copy of the encoder — no decoder, with a VICReg
     variance term so the latent can't collapse. The plotted latent is the
     encoder output.
   Latent dimension is 2 (so the latent *is* the plot) or 4.

4. **The scatter.** One point per episode = that episode's latent, colored by its
   true placement (circle = teal, corners = orange). The color is for *us*; the
   encoder never sees it.

   **The axes.** If the latent is 2-D, the two axes are the latent coordinates
   directly. If the latent is higher-D, we project to its top-2 principal
   components (PCA), so the axes are **PC1 / PC2 of the latent** — a pure
   rotation that shows the two most-spread directions. (The figure labels now say
   which case it is.) There is no other transform: no t-SNE/UMAP, no per-axis
   scaling — distances on the plot are distances in the latent (up to that
   rotation).

   **What we see:** overlapping clouds, not two separated blobs. The placement is
   a linearly decodable *direction* in the latent, not a pair of *modes*.

---

## 2. The belief planner — how it works, and the kinds of belief

Files: `src/part2_planner.py` (planner) and `src/part2_intent_eval.py` (beliefs).

**The belief `b_t`.** A distribution over the K=4 corner intents the prey might be
pursuing, produced by an encoder from the prey's first-k observed positions and
*sharpening* as k grows within the episode (accuracy 0.37→0.97, entropy
1.35→0.03 nats over 3→25 steps).

**The kinds of belief in the project (this is the "different types"):**

| belief fed to the predator | what it is | captures/ep |
|---|---|---|
| **flat / uniform** (1/K) | no information — the ablation | planner 3.07 |
| **online soft posterior** | the encoder's calibrated, sharpening distribution — *the* belief | planner **4.31** |
| **hard-inferred** | the argmax of the posterior at a fixed step (k=8): a point estimate | 2.56 (−5%) |
| **oracle** | the true intent as a one-hot — upper bound on knowing | 4.05 |
| **confident-but-wrong** | a one-hot on the wrong corner | 1.42 (−47%) |

The last two rows are the thesis: a *wrong* point estimate is worse than no
model, so you must carry the whole distribution, not its argmax.

**The planner (Part 2), step by step.** At each real step the predators hold
`b_t` and choose a joint action by one-step lookahead through the *true
simulator*:

1. Enumerate all **125 = 5³** candidate joint predator actions.
2. Draw **8 intents** `g ~ b_t`. The same 8 are shared across all 125 candidates
   (a *paired* comparison → low variance with few rollouts).
3. For each candidate, roll the simulator forward **H=5** steps: predators play
   the candidate's first action then the belief-trained policy; the **prey's
   moves are sampled from its own policy conditioned on the drawn intent `g`** —
   i.e. we imagine the prey heading to a believed corner.
4. Score = discounted captures (γ=0.95) **minus** a leaf term: the distance from
   the nearest predator to the *imagined* prey position at the end of the horizon.
   That leaf term is what turns the plan into an **interception** — end positioned
   where the prey is believed to be going, rather than waiting for a rare
   in-horizon capture.
5. Take the arg-max candidate and play its first action. Repeat next step.

**Why it beats the oracle.** Marginalizing over `b_t` makes the plan
uncertainty-aware: while the belief is flat the predators hedge; as it sharpens
they commit. Planning lookahead lets them *pre-position* — so the online-belief
planner (4.31) tops even the oracle *reactive* policy (4.05): planning beats
reacting even with perfect information. The belief itself contributes +40% over
the flat-belief planner (3.07).

---

## 3. The bar plot on the self-supervised encoder figure

Figure: `plots/jepa_vs_vae_encoder.png` (and the bars in
`mopa_latent_resources_*`). The bar panel reports **two different ways** of asking
"does the latent recover the strategy, without labels?", for each encoder:

- **probe accuracy** — a logistic-regression classifier is trained on the latent
  to predict the strategy, scored by 5-fold cross-validation. This measures
  whether the strategy is **linearly decodable** from the latent (chance = 0.25
  for the 4-way intent task, 0.5 for circle-vs-corners).
- **GMM ARI** — fit a Gaussian mixture to the latent *with no labels*, then
  compare its clusters to the true strategy via the adjusted Rand index (0 =
  chance, 1 = perfect). This measures whether the latent **clusters** into the
  strategies on its own.

The two **dashed/dotted reference lines** are the *supervised ceiling* (a probe
trained on the raw standardized input — what's recoverable in principle) and
*chance*.

**How to read the gap.** On the hidden-intent task both are high (JEPA probe
0.89, ARI 0.65) — the four intents both decode *and* cluster. On circle-vs-corners
the probe is high (up to 0.81) but **ARI ≈ 0**: the placement is linearly
decodable but does *not* form separated clusters. probe ≫ ARI is precisely the
"signal yes, clusters no" finding, and it's why the next step is to *shape* the
latent (contrastive / equivariant / return-relevant MI) rather than add capacity.
