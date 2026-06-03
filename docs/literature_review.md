# Literature review — model-based MARL, opponent modeling, and where we can improve

Scope: placing `marl-opp-aware` (hidden-intent predator–prey; uncertainty-aware
belief encoder + belief-conditioned planner; JEPA opponent encoder) next to the
SOTA the lab actually targets — **model-based multi-agent RL** (multi-agent
MuZero-style baselines), **adaptive opponents under limited interaction**, and the
**representation gap** the team diagnosed on 5/13 (the VAE encodes *where on the
map* the agent goes, not the rotation/reflection-invariant *strategy*).

Generated from a Semantic Scholar sweep (`src/lit_review_s2.py`, raw in
`logs/lit_review_s2.json`, 560 papers / 30 families). Cite counts are as of the
pull (2026-06-03) and are only a rough influence signal. Notes are mine; the
"→ for us" line is the gap / opportunity for this project.

---

## 1. Model-based MARL & the MuZero family (the lab's baseline question)

**Schrittwieser et al. 2019 — MuZero: Mastering Atari, Go, chess and shogi by planning with a learned model.** *Nature; 2527 cites; arXiv:1911.08265.*
Learns a model that predicts only what matters for planning — policy, value,
reward — in a latent space, with no access to the true dynamics, and plans with
MCTS over it. The anchor of the whole "learned-model planning" line.
→ for us: this is the *single-agent* archetype the lab wants the multi-agent
baseline compared against. Note MuZero's model predicts **reward/value/policy**,
not full next-state — our Part 3/4 world model predicts next *state*; that design
difference is exactly why our planner over-trusted off-distribution actions.

**Ye et al. 2021 — EfficientZero: Mastering Atari Games with Limited Data.** *NeurIPS; 325 cites; arXiv:2111.00210.*
MuZero + a self-supervised consistency loss (SimSiam-style) + value-prefix; first
to beat median human on Atari-100k (2 hours of data). The self-supervised
consistency term is essentially a self-predictive/JEPA-flavored auxiliary.
→ for us: the strongest evidence that *sample efficiency* in model-based RL comes
from a **self-predictive auxiliary** — the same principle as our JEPA encoder.
This is the citation that connects "predict-don't-reconstruct" to the lab's
sample-efficiency thesis. EfficientZeroV2 (Wang 2024) extends to continuous control.

**Hafner et al. 2019 — Dreamer: Learning Behaviors by Latent Imagination.** *ICLR 2020; 1955 cites; arXiv:1912.01603.*
Learns long-horizon behavior by backpropagating value gradients through
trajectories imagined in a compact learned latent space. DreamerV3 (2023)
generalized it across domains with fixed hyperparameters.
→ for us: the latent-imagination archetype that MAMBA/MACD lift to multi-agent.
Our Part 4 is a Dreamer-shaped idea (plan in latent space) that *failed* for the
planner — see Terver 2025 (§4) for why latent planning is not free.

**Egorov & Shpilman 2022 — MAMBA: Scalable Multi-Agent Model-Based RL.** *AAMAS; 44 cites; arXiv:2205.15023.*
A Dreamer-style world model for MARL: each agent imagines rollouts using a learned
model plus communication, drastically cutting environment interaction vs model-free
CTDE. One of the two clearest "multi-agent world model" baselines.
→ for us: **a direct baseline to implement.** MAMBA models cooperative dynamics +
communication; it does *not* maintain a belief over a discrete hidden opponent
intent. Our value-add is exactly that explicit latent-intent belief.

**Liu et al. 2024 — Efficient Multi-agent RL by Planning (MAZero).** *ICLR; 19 cites; arXiv:2405.11778.*
A genuine multi-agent MuZero: learned model + tree search, handling the blown-up
joint action space via a nearly-independent / optimistic search. The cleanest
answer to Ellen's "multi-agent extension of MuZero" question.
→ for us: **the headline baseline.** Test the lab hypothesis directly — under
*limited interaction with an adaptive/novel prey*, does MAZero's learned model
need re-fitting while our belief-conditioned planner adapts in a few episodes by
just re-inferring the intent? That comparison is a paper-worthy experiment.

**Chai et al. 2024 — MACD: Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination.** *AAMAS; 15 cites.*
Centralized-Imagination-Decentralized-Execution (CIDE): a centralized world model
generates higher-quality pseudo-data for decentralized policies; targets credit
assignment + sample efficiency.
→ for us: shows the field is converging on *centralized imagination* — the same
place our centralized-critic CTDE lives. Cite as the cooperative counterpart;
contrast that we target *mixed/competitive* hidden-intent inference, not credit.

**Yu et al. 2021 — MBOM: Model-Based Opponent Modeling.** *NeurIPS; 48 cites; arXiv:2108.01843.*
**The closest paper to our whole thesis.** Uses the environment model to *simulate
recursive reasoning* and imagine a set of improving opponent policies, so a single
agent adapts to opponents that are fixed, learning, *or* reasoning — all in one
method.
→ for us: **this is the baseline we must cite and beat/complement.** MBOM imagines
opponent *policies* via recursive rollouts; we infer a *low-dimensional hidden
intent* with calibrated uncertainty and plan against the belief. Hypothesis to
test: under limited interaction our explicit-belief approach is more
sample-efficient than MBOM's imagined-policy recursion when the opponent's
variation is a small discrete latent (which is the realistic structured case).

**Zhang et al. 2021 — AORPO: Model-based Multi-agent Policy Optimization with Adaptive Opponent-wise Rollouts.** *IJCAI; 45 cites; arXiv:2105.03363.*
Each agent builds its own environment model = dynamics model + **multiple opponent
models**, and the paper gives a return-discrepancy bound separating *dynamics
sample complexity* from *opponent sample complexity*.
→ for us: the theoretical scaffold we're missing. Their opponent-sample-complexity
bound is the right language to argue "inferring a discrete intent needs fewer
opponent interactions than learning an opponent dynamics model." Borrow the
analysis to justify the belief approach formally.

**Antonoglou et al. 2022 — Stochastic MuZero: Planning in Stochastic Environments with a Learned Model.** *ICLR; 83 cites.*
Extends MuZero to stochastic dynamics by learning a model with chance nodes /
afterstates.
→ for us: relevant because a hidden-intent opponent makes our environment
*stochastic from the predators' view* — the intent is exactly an unobserved chance
variable. Stochastic MuZero's chance-node machinery is one principled way to fold
the intent belief into a learned-model planner (an alternative to our explicit
softmax belief).

**Czechowski & Oliehoek 2020 — Decentralized MCTS via Learned Teammate Models.** *IJCAI; 22 cites; arXiv:2003.08727.*
Decentralized online planning where each agent plans with MCTS using *learned
models of teammates*; one-agent-at-a-time adaptation gives convergence guarantees.
→ for us: the cooperative-planning analogue of our predator coordination; the
"learned model of the other agent drives my search" pattern is exactly our
belief-conditioned planner, but for teammates rather than an adversary.

---

## 2. Opponent modeling & theory of mind

**He et al. 2016 — Opponent Modeling in Deep RL (DRON).** *ICML; 372 cites; arXiv:1609.05559.*
Encodes opponent observations into a DQN; a Mixture-of-Experts auto-discovers
opponent strategy patterns without extra supervision.
→ for us: the ancestor of belief-conditioning. We make the opponent representation
*explicit + uncertainty-calibrated* rather than an implicit DQN feature.

**Rabinowitz et al. 2018 — Machine Theory of Mind (ToMnet).** *ICML; 589 cites; arXiv:1802.07740.*
Meta-learns a prior over agents and infers their goals/mental state from behavior
alone; passes the Sally-Anne false-belief test in gridworlds.
→ for us: **inference-from-behavior is established here** — our 0.37→0.97 intent
accuracy is a ToMnet-style capability, not a novelty. Our novelty is what we *do*
with the belief (plan) and *how* we get it cheaply (JEPA, label-free). Don't
over-claim the inference itself.

**Albrecht & Stone 2018 — Autonomous agents modelling other agents: a comprehensive survey.** *AIJ; 536 cites; arXiv:1709.08071.*
The taxonomy: policy reconstruction, type-based reasoning, plan/goal recognition,
recursive reasoning, graphical models. The map of the whole field.
→ for us: cite for framing; we sit at *type-based reasoning* (discrete intent) +
*plan recognition* + *planning*.

**Wen et al. 2019 — PR2: Probabilistic Recursive Reasoning for MARL.** *ICLR; 169 cites; arXiv:1901.09207.*
Level-k recursion: each agent variationally approximates the opponent's conditional
policy and best-responds, modeling "what they think I'll do."
→ for us: the recursive-reasoning baseline. Ours is *non-recursive* (the prey's
intent is exogenous, not a best-response to us) — simpler, and honest about it.
If we later make the prey adaptive, PR2 becomes the right comparison.

**Shen & How 2019/2021 — Robust Opponent Modeling via Adversarial Ensemble RL.** *ICAPS; 19 cites; arXiv:1909.08735.*
Infers an uncertain opponent *type* (private info) via Bayes' rule and trains an
ensemble for robustness in asymmetric imperfect-info games.
→ for us: the nearest "uncertainty over opponent type" prior. Our belief-sharpness
sweep (confident-but-wrong is −47%, worse than blind) is a *cleaner demonstration*
of why the posterior, not a point estimate, must drive action. Cite as the robust
counterpart; differentiate on the calibrated-belief-into-planner mechanism.

**Davies et al. 2020 — Learning to Model Opponent Learning (LeMOL).** *AAAI student abstract; 7 cites; arXiv:2006.03923.*
Models not the opponent's policy but the opponent's *learning process* (how the
policy changes), to anticipate a moving target.
→ for us: points at the frontier we haven't touched — opponents that *adapt*. Our
intent is static per episode; modeling the opponent's adaptation is the natural
next escalation.

**Tian et al. 2022 — Multi-agent Actor-Critic with Time Dynamical Opponent Model (TDOM).** *Neurocomputing; 15 cites; arXiv:2204.05576.*
Encodes the prior that opponents *improve their return over time* into the opponent
model, easing non-stationary policy improvement.
→ for us: a cheap, structured prior on opponent dynamics; if we make the prey
learn, TDOM's "opponents tend to improve" prior is a baseline assumption to test
our belief tracker against.

---

## 3. Non-stationarity & adaptive opponents (the "limited interaction" frontier)

**Hernandez-Leal et al. 2017 — A Survey of Learning in Multiagent Environments: Dealing with Non-Stationarity.** *arXiv:1707.09183; 322 cites.*
The canonical taxonomy of opponent-induced non-stationarity (ignore / model /
track / respond) across game theory, RL, and bandits.
→ for us: the framing document for the "adaptive opponents" half of the proposal;
positions our belief tracker as a *model-and-respond* method.

**Papoudakis et al. 2019 — Dealing with Non-Stationarity in Multi-Agent Deep RL.** *arXiv:1906.04737; 237 cites.*
Survey specific to deep MARL: centralized training, opponent-policy representation,
meta-learning, communication.
→ for us: same authors as the VAE-OM baseline; cite both to show we're squarely in
their problem statement (opponent-policy representation under non-stationarity).

**Al-Shedivat et al. 2018 — Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments.** *ICLR; 373 cites; arXiv:1710.03641.*
Casts adaptation as meta-learning; introduces RoboSumo + iterated adaptation games;
shows meta-learning adapts in the *few-shot* regime far better than reactive
baselines against opponents that change.
→ for us: **the reference for the lab's exact hypothesis** ("better at adaptive
opponents under limited interaction"). The experiment to run: an *adaptive* prey,
and show belief-conditioned planning gives few-shot adaptation comparable to
meta-RL without the meta-training cost.

**Everett & Roberts 2018 — Learning Against Non-Stationary Agents with Opponent Modelling and Deep RL (SAM).** *AAAI Spring Symp; 31 cites.*
Online *switching* opponent detection — recognize which opponent type you face and
swap to the matching response policy, mid-interaction.
→ for us: directly motivates a **switching-intent** experiment (prey changes corner
mid-episode); tests whether our online belief is a true *tracker* (changepoint
detection) or just a converging *classifier*. This is a concrete, cheap upgrade.

---

## 4. Representation: the "strategy, not position" gap

This is the lab's own 5/13 diagnosis — the VAE latent organizes by *map location*,
not by *strategy* (rotate/reflect the trajectories and the strategy is unchanged).
The literature has two clean families of fixes: **equivariance** (architectural
invariance) and **self-predictive / contrastive** objectives (learned invariance).

**van der Pol et al. 2020 — MDP Homomorphic Networks: Group Symmetries in RL.** *NeurIPS; 191 cites.*
Policy/value networks made *equivariant* to symmetries of the joint state-action
space, shrinking the solution space and improving sample efficiency.
→ for us: the foundational equivariance-in-RL paper; the principle behind a
symmetry-aware opponent encoder.

**van der Pol et al. 2021 — Multi-Agent MDP Homomorphic Networks.** *ICLR; 37 cites.*
The multi-agent version. Its abstract's own example: *"rotating the state globally
results in a permutation of the optimal joint policy."*
→ for us: **the single most on-point paper for our gap.** Our arena's 4 corners
have a dihedral-D4 symmetry; a D4-equivariant encoder would make the strategy
latent *invariant to where the corner is*, which is exactly what the VAE failed to
do. **Top model-improvement candidate: a D4-equivariant JEPA opponent encoder.**

**Chen & Zhang 2023 — E(3)-Equivariant Actor-Critic Methods for Cooperative MARL.** *ICML; 10 cites; arXiv:2308.11842.*
Characterizes a subclass of Markov games admitting symmetric optimal policies and
builds Euclidean-equivariant actor-critics.
→ for us: theoretical license that the *optimal* predator policy is symmetric under
arena symmetries — so baking in equivariance loses nothing and gains sample
efficiency.

**Deac, Weber & Papamakarios 2023 — Equivariant MuZero.** *TMLR; 4 cites; arXiv:2302.04798.*
Bakes environment symmetries into MuZero's learned world model for better
generalization to unseen configurations.
→ for us: **the bridge between our two gaps** — symmetry + learned-model planning.
If we ever revisit the Part-4 latent world model, making it equivariant is the
principled fix for its poor OOD-action generalization.

**Schwarzer et al. 2021 — SPR: Data-Efficient RL with Self-Predictive Representations.** *ICLR; 408 cites; arXiv:2007.05929.*
Predict your *own* latent state k-steps ahead via an EMA target encoder + a learned
transition model; large Atari-100k sample-efficiency gains.
→ for us: **the exact machinery of our JEPA encoder**, applied to *self* in
single-agent RL. Our novelty = transplanting it to *opponent* encoding. This is
the strongest methodological citation we have.

**Guo et al. 2020 — PBL: Bootstrap Latent-Predictive Representations for Multitask RL.** *ICML; 155 cites; arXiv:2004.14646.*
Predict future latent observations and use them to bootstrap representations across
tasks.
→ for us: precedent that predictive (not reconstructive) latents transfer; supports
the label-free belief generalizing across opponents.

**Ni et al. 2024 — Bridging State and History Representations: Understanding Self-Predictive RL.** *ICLR; 53 cites; arXiv:2401.08898.*
Shows many representation methods reduce to one *self-predictive abstraction*
idea; explains the stop-gradient/EMA trick theoretically; gives a minimalist
algorithm.
→ for us: the theory that justifies our stop-gradient + EMA target design choices —
cite to defend "why predict-don't-reconstruct" beyond the empirical win.

**Assran et al. 2023 — I-JEPA: Self-Supervised Learning from Images with a JEPA.** *CVPR; 867 cites; arXiv:2301.08243.*
Predict representations of masked target blocks from a context block; non-generative
SSL; semantic features without hand-crafted augmentation.
→ for us: the JEPA origin and the masked-prediction design our anytime encoder
mirrors (observe first-k, predict the masked future).

**Li et al. 2024 — T-JEPA: A JEPA for Trajectory Similarity.** *SIGSPATIAL; 12 cites; arXiv:2406.12913.*
Applies JEPA to *trajectories* — sample-and-predict in representation space to infer
high-level trajectory semantics without augmentation.
→ for us: **direct precedent for "JEPA on trajectories"** (though for GPS movement,
not opponents/RL). Validates the modality choice; we extend it to *adversarial
intent* + downstream planning, which they don't do.

**Eysenbach et al. 2018 — DIAYN: Diversity is All You Need.** *ICLR; 1281 cites; arXiv:1802.06070.*
Learns diverse skills with no reward by maximizing mutual information between a
latent skill code and visited states.
→ for us: the generative direction of "strategy = latent code." Relevant if we want
to *generate* a diverse prey population (varied opponents) to stress-test the
encoder, rather than relying on the 4 hand-set intents.

**He et al. 2020 — MASD: Skill Discovery of Coordination in MARL.** *arXiv; 9 cites.*
Multi-agent skill discovery: maximize MI between a skill code and the *joint* state
while suppressing its influence on any single agent (adversarial), to get
*coordination* skills.
→ for us: a recipe for a diverse opponent *population* with genuinely distinct
joint strategies — better than 4 fixed corners for testing generalization.

**Hu et al. 2024 — ACORM: Attention-Guided Contrastive Role Representations for MARL.** *ICLR; 28 cites; arXiv:2312.04819.*
MI-maximization + contrastive learning to represent emergent *roles*; attention
lets the global state attend to role embeddings.
→ for us: the *contrastive* alternative to equivariance for the strategy-vs-position
gap — treat rotated/reflected versions of the same strategy as positives. Cheaper
to bolt onto our existing encoder than re-architecting for equivariance.

**Li et al. 2024 — Learning Distinguishable Trajectory Representation with Contrastive Loss.** *NeurIPS; 3 cites.*
Contrastive trajectory representations to keep parameter-shared MARL agents
behaviorally distinct (vs MI-with-identity methods that explore poorly).
→ for us: a recent, on-point contrastive-trajectory objective; a concrete baseline
to compare our JEPA encoder against on the "separate the strategies" metric.

---

## 5. JEPA / self-predictive planning — when latent planning helps

**Terver et al. 2025 — What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?** *arXiv:2512.24497; 16 cites.*
Characterizes *when* planning in a JEPA world model's latent space beats planning in
input space — abstracting "irrelevant detail" helps only if the discarded detail is
truly irrelevant to control.
→ for us: **the explanation for our Part-4 honest negative.** JEPA's
variance-regularized latent throws away dynamics detail the planner's OOD-action
queries actually need. Reframe the negative as a *characterized* phenomenon
(encoder wants to discard opponent noise; world model must keep dynamics) and cite
Terver as the general account.

**Maes et al. 2026 — LeWorldModel: Stable End-to-End JEPA world model.** *arXiv:2603.19312; 40 cites.*
Recent recipe for stable end-to-end JEPA world-model training (the instability we
hit in Part 4 directly).
→ for us: if we revisit the latent world model, this is the stabilization reference
(our fix was ad hoc: EMA 0.997, low JEPA weight, low LR).

**NextLat 2025 — Next-Latent Prediction Transformers Learn Compact World Models.** *arXiv:2511.05963; 9 cites.*
Adds a self-supervised next-latent prediction objective to a transformer so it
compresses history into consistent latent states.
→ for us: a sequence-model route to the same self-predictive latent; an alternative
encoder backbone if we move from MLP to a transformer over the trajectory.

---

## 6. Belief-space planning & tree search (Part 2 / Part 3 lineage)

**Han et al. 2019 — IPOMDP-Net: Deep Net for Partially Observable Multi-Agent Planning via I-POMDPs.** *AAAI; 19 cites.*
Embeds an I-POMDP model + QMDP planner in a differentiable network; end-to-end
trained; generalizes to unseen env variants.
→ for us: the deep belief-space-planning baseline; more integrated than our
explicit MC planner. A target to compare planning quality/transfer against.

**Han et al. 2017 — Learning Others' Intentional Models in Multi-Agent Settings Using I-POMDPs.** *MAICS; 32 cites.*
Learns models of *others' intentions* within the I-POMDP framework.
→ for us: the classical statement of "infer the other's intent and plan" — our work
is the deep, JEPA-encoder, JaxMARL instantiation of this idea.

**Schwarting et al. 2019 — Stochastic Dynamic Games in Belief Space.** *T-RO; 75 cites; arXiv:1909.06963.*
Game-theoretic continuous POMDP solved with iLQG in Gaussian belief space; agents
gather information and reason about what their actions reveal.
→ for us: the continuous-control, information-gathering analogue. Their "what does
my action reveal" idea is a richer objective than our greedy interception — a
direction for active belief-sharpening (move to *disambiguate* the intent).

**Huang et al. 2024 — Learning Online Belief Prediction for Efficient POMDP Planning in Autonomous Driving.** *RA-L; 19 cites; arXiv:2401.15315.*
Transformer + recurrent memory online-updates a latent belief over other agents'
intentions; feeds an MCTS planner with macro-actions and multimodal trajectory
prediction.
→ for us: **the closest full-system analogue to our pipeline** (infer belief →
plan), in driving. We differ by the JEPA/label-free encoder and the small,
controlled testbed. Good "same shape, different domain + cheaper encoder" cite.

---

## Synthesis — the real gaps and where the model can improve

Ranked by leverage for this project.

### Gap A (highest leverage): the strategy latent still encodes *position*, not strategy
The lab diagnosed this for the VAE on 5/13. **Our JEPA encoder does not actually fix
it** — JEPA discards *unpredictable evasion noise*, but it still predicts *absolute
future positions*, so a rotation/reflection of the arena would move the latent. The
published fix is **equivariance** (Multi-Agent MDP Homomorphic Networks, van der Pol
2021; the abstract is our exact example). 
- **Concrete improvement:** a **D4-equivariant JEPA opponent encoder** — encode the
  prey trajectory with a network equivariant to the dihedral group of the 4 corners,
  so "strategy" is provably invariant to which corner / orientation. This combination
  (equivariance × JEPA × opponent modeling) is unclaimed in the literature.
- **Cheaper alternative:** a **contrastive** objective (ACORM / Li 2024 style) that
  makes rotated/reflected trajectories of the same intent positives — learned, not
  architectural, invariance. Faster to try; a good ablation against the equivariant
  version.
- **Test:** probe accuracy on a *held-out rotation* of the arena. The VAE and our
  current JEPA should drop; the equivariant/contrastive encoder should hold. That
  delta is the cleanest possible evidence we closed the gap.

### Gap B: no model-based-MARL baseline is actually run
We claim model-based relevance but have implemented none of the baselines the lab
asked about. **Implement MAZero (Liu 2024) and/or MAMBA (Egorov 2022), and MBOM (Yu
2021) / AORPO (Zhang 2021) as the opponent-aware model-based baselines** in
`simple_tag_intent`.
- **Test (this is Ellen's hypothesis turned into a result):** under *limited
  interaction with a novel/adaptive prey*, measure episodes-to-recover. Prediction:
  MAZero/MAMBA must re-fit a world model (slow); our belief-conditioned planner
  re-infers a 4-way intent in ~12 steps (fast). If that holds, it's the paper's core
  claim, *quantified against real baselines* rather than asserted.

### Gap C: the opponent is static; the frontier is *adaptive* opponents
Our prey draws a fixed intent per episode. The interesting SOTA problem (Al-Shedivat
2018; Everett 2018; LeMOL; MBOM) is opponents that *change*.
- **Cheap upgrade:** **switching intent mid-episode** → tests whether our online
  belief is a real *tracker* (changepoint detection, Everett 2018) or just a
  converging classifier.
- **Bigger upgrade:** an **adaptive (best-responding) prey** → tests few-shot
  adaptation vs meta-RL (Al-Shedivat). This is also where recursive reasoning (PR2)
  becomes the right comparison.

### Gap D: the Part-4 negative is uncharacterized
Our JEPA world model lost to the state-space model. Terver 2025 explains exactly
this (latent planning only helps if discarded detail is control-irrelevant; dynamics
detail is not). 
- **Improvement:** either (i) reframe as a *characterized* result — "JEPA latents
  help the encoder (discard opponent noise) and hurt the world model (discard
  dynamics)," citing Terver — or (ii) fix it with an **equivariant / consistency-
  regularized** latent model (Equivariant MuZero; EfficientZero's consistency loss;
  LeWorldModel for stability).

### Gap E: no sample-complexity / theory framing
AORPO (Zhang 2021) separates *dynamics* vs *opponent* sample complexity with a
return-discrepancy bound. We have none.
- **Improvement:** argue (even informally) that inferring a K-way discrete intent
  costs O(log K) bits / few interactions, vs learning an opponent *dynamics* model —
  the formal version of "belief is cheaper than a world model." Directly supports the
  limited-interaction thesis.

### Gap F: single environment (MPE simple_tag)
Everything is one 2D particle task. The lab already linked **JaxMARL / SMAX**
(Foerster lab, our JAX stack) in Slack.
- **Improvement:** port the belief-conditioned planner + (equivariant) JEPA encoder
  to one JaxMARL env (SMAX or MPE-spread) to show the method isn't a simple_tag
  artifact. Low marginal cost given JaxMARL is JAX-native.

### One-line positioning for the meeting
*We're a controlled study of **belief-conditioned planning against a hidden-intent
opponent**, with two real deltas over published work — a **predictive (JEPA)
opponent encoder** that beats the VAE-for-OM baseline (Papoudakis 2019) and a
**label-free belief** that matches supervised — but we have **not yet** run the
model-based-MARL baselines (MAZero/MAMBA/MBOM), made the opponent **adaptive**, or
closed the **symmetry gap** (equivariance). Those three are the highest-leverage next
steps, and the literature gives us the exact tools for each.*
