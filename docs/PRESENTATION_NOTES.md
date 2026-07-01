# Presentation — slide plan + speaker notes

Results-focused. Metrics are **captures/episode (performance)**, not action
accuracy. Each slide: what's on it, the figure, and what to say.

---

### 1. Title
**Adaptive Opponent Modeling for Adversarial Co-Training in MARL.** JAX / JaxMARL.
> "The idea: infer the opponent's hidden strategy with calibrated uncertainty, and plan against it. I'll show results, then the behaviour-cloning follow-up from last time."

### 2. The task — hidden strategy *(fig: demo_jepa.gif)*
> "Three predators chase one prey. Each episode the prey secretly picks one of four corners to haunt; the predators can't see which, they infer it from motion. Left is intent-blind and just chases; right infers the corner and intercepts. The strategy lives *inside* the opponent, not on the map."

### 3. Headline ladder — captures/episode *(fig: part2_intent_eval.png)*
Numbers: blind 2.68 · wrong-guess 1.42 (−47%) · hard-inferred 2.56 (−5%) · reactive belief 2.82 (+5%) · planner-flat 3.07 · oracle 4.05 (+51%) · **planner+belief 4.31 (+61%)**.
> "Each row isolates one thing. The key one: a confident *wrong* guess is worse than knowing nothing — minus 47 percent. So you can't act on a point estimate; you carry the whole belief. Planning on the belief gets 4.31 captures, +61% over blind."

### 4. Planning beats the oracle *(fig: part2_planner.png)*
> "The planner at 4.31 beats even the oracle that's *handed* the true strategy, 4.05 — because planning lets you pre-position, and lookahead beats reacting even with perfect info. The belief itself contributes +40% over a flat-belief planner."

### 5. The encoder — predict, don't reconstruct *(fig: jepa_vs_vae_encoder.png)*
> "How do we get the belief cheaply? A JEPA encoder that predicts the future representation instead of reconstructing. On this task it recovers the strategy at probe 0.89 versus a VAE's 0.53, matching a supervised classifier — with no labels. Its label-free belief drives the planner to 4.08."

### 6. Honest negative — JEPA as a world model *(fig: part4_jepa_world_planner.png)*
> "The same predict-don't-reconstruct idea *fails* as a planning world model — 0.57 captures. Encoders want to discard noise; a world model needs accurate dynamics. Opposite requirements. I report it straight."

### 7. Second axis — circle vs corners: "are circles just squares?" *(fig: occupancy_heatmaps.png)* ★ NEW
> "Someone asked whether circle and corners are even different behaviours — technically 4 resources on a circle *is* a square of points. So I looked at where each prey actually spends time. The corners prey is sharp — four crisp hot-spots. The circle prey is diffuse — a smeared ring, no crisp peaks. They're different, cosine 0.42, but asymmetric: the circle behaviour is weak and spread out. That diffuseness is exactly why it's hard to separate — and it motivates trying a genuinely distinct placement, like a snake path."

### 8. Circle vs corners — can an encoder recover it? *(fig: mopa_latent_resources_mappo_prey_occ.png)*
> "Same Part-1 question here. The axes are the 2-D latent (or PC1/PC2 when higher-D); each point is one episode's occupancy embedding, colored by placement — the encoder never sees the color. Result: the placement is *linearly decodable* — supervised probe 0.93, latents up to 0.81 — but it does **not cluster**, ARI near zero. Signal yes, separated modes no. That's the open problem."

### 9. Does BC even clone the expert? — CAPTURES *(fig: mopa_bc_vs_mappo.png)*
> "Before asking if a latent helps, is BC a faithful clone? Deployed in the game, vanilla BC gets 1.22 captures; the MAPPO expert it copies gets 1.35; random is 0.40. So BC recovers **86%** of the expert's edge — cloning isn't the bottleneck. The 14% gap is the room a strategy signal could fill."

### 10. Latent-conditioned BC — CAPTURES *(fig: mopa_bc_latent_deploy.png)* ★ [FILL]
> "Now condition a placement-blind clone on the prey-strategy latent and measure captures. [vanilla X · +latent Y · +oracle Z · MAPPO 1.35]. The latent recovers [N]% of the pooled-BC→expert gap." — *reframe to sweep if deploy unavailable.*

### 11. Honest mechanism — is the gain just averaging? ★ NEW
> "A fair question: the latent improves as it watches longer — is that the *strategy*, or just averaging over more episodes? Honest answer: here the placement is **fixed** across a specialist's episodes, so more observation gives a better *occupancy estimate* — it's estimation quality, statistical averaging, not the strategy unfolding. That's different from the hidden-intent task, where the intent genuinely *is* revealed over time. So I call it a scaling-with-observation result, and I don't overclaim."

### 12. Where vanilla BC struggles *(fig: demo_bc.gif)* ★
> "The concrete failure: on a *corners* episode, a placement-blind clone sees the prey drifting and can't tell if it's committing to that corner or just passing through, so it hedges and the prey reaches its spot. Knowing the placement turns hedging into interception."

### 13. How the plots are generated
> "Every result: 3 seeds, episode-level splits, leakage-checked. Occupancy = normalized 8×8 histogram of where the prey goes. The supervised probe is a cross-validated logistic regression *with* labels — the ceiling. The latent probe swaps the input for the encoder's 2–4-D output. Full procedures in METHODS.md."

### 14. Next steps
> "One: replace the diffuse circle with a genuinely distinct placement — a snake path — so the two behaviours actually form clusters. Two: shape the latent to cluster — contrastive identity grounding (CTR) or equivariance. Three: everything in captures, deployed, not action accuracy."

---

**Do NOT compare absolute heights across the BC figures** (bc_vs_mappo captures vs the accuracy figures) — different metrics, different experiments.
**If asked why VAE not JEPA on occupancy:** occupancy already removed the evasion noise JEPA exploits, so the generative VAE wins here; on raw-trajectory intent, JEPA wins 0.89 vs 0.53. Winner depends on the feature.
