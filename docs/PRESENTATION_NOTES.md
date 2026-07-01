# Presentation Notes

Deck: `marl_opp_aware_results.pptx`

This deck is a fuller paper talk. The first slide stays blank except for the
title. The rest of the deck explains the setup, what each ablation measured,
where the result fits in the literature, and which experiments should run before
submission.

## One-Sentence Positioning

In a controlled hidden-intent predator-prey game, a calibrated belief over the
opponent's strategy improves planning more than a hard intent guess.

## Slide Flow

1. **Adaptive Opponent Modeling for MARL**
   - Blank title slide.
   - Say the whole talk studies one question: whether belief improves control.

2. **The paper claim is narrow enough to defend**
   - Main result: inferred-belief planning reaches 4.31 captures per episode.
   - Stress test: a wrong hard intent reaches 1.42.
   - Label-free path: JEPA belief planning reaches 4.08.

3. **The task hides strategy inside the prey**
   - The prey draws a corner intent at reset.
   - Predators infer intent from early motion.
   - The fixed map prevents a perception-only explanation.

4. **The belief signal becomes measurable**
   - Intent accuracy rises from 0.37 to 0.97.
   - Posterior entropy falls from 1.35 to 0.03 nats.
   - This validates the Part 1 encoder.

5. **Hard intent labels can damage control**
   - Opponent-blind baseline: 2.68.
   - Hard intent inferred at k=8: 2.56.
   - Confident wrong intent: 1.42.
   - The point estimate loses useful uncertainty.

6. **The planner gets value from the whole belief**
   - Flat-belief planner: 3.07.
   - Reactive oracle: 4.05.
   - Inferred-belief planner: 4.31.
   - The flat-belief ablation isolates the value of inference.

7. **The oracle comparison needs precise wording**
   - The oracle policy reacts with true intent.
   - The belief planner searches short futures.
   - Say "reactive oracle." An oracle planner remains the real ceiling.

8. **JEPA recovers the belief without intent labels**
   - Same trajectory data and latent size as VAE.
   - JEPA probe accuracy: 0.89.
   - VAE probe accuracy: 0.53.
   - JEPA belief planner reaches 4.08.

9. **Learned dynamics failed the control test**
   - Simulator belief planning works.
   - The latent world planner reaches 0.57 captures per episode.
   - Current bottleneck: dynamics fidelity.

10. **BC is strong enough to test strategy inputs**
    - Random policy: 0.40.
    - Vanilla BC: 1.22.
    - MAPPO expert: 1.35.
    - The clone is good enough for strategy-conditioning checks.

11. **Strategy information has value, even when the latent is weak**
    - Oracle placement closes most of the deployed capture gap.
    - The current unsupervised latent gives a small deployed lift.
    - Better probes improve BC accuracy in the observation sweep.

12. **The literature slot is belief plus planning**
    - HOP is the closest belief-search comparator.
    - MAZero, MAMBA, MARIE, and MATWM test model-based MARL claims.
    - MBOM and AORPO test the opponent-modeling claim.

13. **Before submission, run the missing tests**
    - Oracle planner.
    - Switching prey.
    - HOP baseline.
    - Raw artifact archive.

## Reviewer Questions

**Why can inferred belief planning exceed the oracle policy?**
The oracle policy receives true intent but reacts. The belief planner searches
short futures with simulator dynamics. An oracle planner would be the ceiling.

**Is the opponent adaptive?**
No. The current prey draws a static intent at reset. That is why switching prey
appears on the closing slide.

**Is the resource-axis latent a main result?**
No. The resource-axis follow-up shows that strategy information has downstream
value. The hidden-intent belief planner remains the main result.

**What is the closest literature comparison?**
HOP is the closest full-system baseline because it combines goal belief and
search. The other baseline groups test different parts of the claim.
