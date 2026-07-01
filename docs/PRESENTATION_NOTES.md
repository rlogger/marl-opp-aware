# Presentation Notes

Deck: `marl_opp_aware_results.pptx`

The deck is intentionally minimal: eight audience-facing slides, with the
technical defense in speaker notes and repository docs. The core message is that
the project is strongest when framed as a controlled study of calibrated
opponent belief plus planning, not as a broad solved-model-based-MARL result.

## One-Sentence Positioning

In a controlled hidden-intent predator-prey game, maintaining a calibrated
belief over the opponent's strategy and sampling that belief in a planner is
materially stronger than reacting to no model or to a hard guessed intent.

## Slide Flow

1. **Adaptive Opponent Modeling for MARL**
   - Anchor the audience on the three numbers: 4.31 captures for inferred-belief
     planning, -47% for a wrong hard intent, and zero labels for the JEPA belief.

2. **The experiment isolates one question**
   - The prey sees a hidden corner intent; predators infer it from early motion.
   - State the boundary early: static intent within each episode.

3. **The belief becomes useful evidence quickly**
   - Accuracy rises from 0.37 to 0.97 as more prey steps are observed.
   - Wrong confident intent is worse than being blind.

4. **Planning over the belief is the core result**
   - Flat-belief planner: 3.07 captures/episode.
   - Reactive oracle: 4.05.
   - Inferred-belief planner: 4.31.
   - Say "reactive oracle," not "oracle planner."

5. **JEPA makes the belief label-free**
   - JEPA beats VAE on the same trajectory signal without intent labels.
   - This is the bridge to self-supervised opponent representation learning.

6. **The main caveat is dynamics, not representation**
   - The learned latent world planner is weak.
   - This narrows the paper claim to belief and planning with simulator
     dynamics, while identifying the next engineering risk.

7. **The BC follow-up says strategy information has value**
   - BC is a credible clone, so imitation is not the bottleneck.
   - Oracle strategy information helps; today's unsupervised resource latent is
     still modest and noisy.

8. **What to say, and what not to overclaim**
   - Claim: calibrated opponent belief matters for planning.
   - Caveat: static per-episode opponents, simulator planner, reactive oracle.
   - Next: adaptive opponents and baselines from HOP, MAZero/MAMBA/MARIE/MATWM,
     MBOM, and AORPO.

## Reviewer Questions

**Why does the planner beat the oracle?**
The oracle condition is reactive with true intent. The inferred-belief condition
has lookahead. An oracle planner is the real ceiling and should be run before
submission.

**Is the opponent adaptive?**
Not yet. The current prey draws a static intent per episode. That isolates the
belief/planning mechanism, but switching or adaptive opponents are the next
scientific upgrade.

**Is the resource-axis latent a paper-level win?**
Not alone. It shows that strategy information has downstream value, but the
unsupervised latent is not yet as clean as the hidden-intent JEPA result.

**Where does this sit in the literature?**
HOP is the closest full-system comparison because it couples goal belief with
MCTS. MAZero, MAMBA, MARIE, and MATWM are the model-based MARL line. MBOM and
AORPO are the opponent-modeling line.
