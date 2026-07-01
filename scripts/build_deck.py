"""Build the results presentation (.pptx) with figures + speaker notes.

Re-runnable: pulls figures from plots/ and prefers the captures deploy figure
if it exists. Output: marl_opp_aware_results.pptx (repo root).
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, "plots")
ACCENT = RGBColor(0x0D, 0x6E, 0x7A)
DARK = RGBColor(0x17, 0x17, 0x17)
GREY = RGBColor(0x5B, 0x5B, 0x5B)
EMW, EMH = 13.333, 7.5   # 16:9

latent_bc = ("mopa_bc_latent_deploy.png" if os.path.exists(
    os.path.join(PLOTS, "mopa_bc_latent_deploy.png")) else "mopa_bc_latent_sweep.png")

SLIDES = [
    dict(kind="title",
         title="Adaptive Opponent Modeling for Adversarial Co-Training in MARL",
         sub="Infer the opponent's hidden strategy with calibrated uncertainty, then plan against it.\nJAX / JaxMARL · results & the behaviour-cloning follow-up",
         notes="Roadmap: the headline planning results, then the circle-vs-corners representation question, then the behaviour-cloning follow-up in captures (not accuracy)."),
    dict(img="demo_jepa.gif", title="The task: the strategy is hidden inside the opponent",
         bullets=["Three predators chase one prey; each episode the prey secretly commits to one of four corners",
                  "Predators can't see it — they infer it from motion over time",
                  "Left: intent-blind (hedges). Right: belief-driven (infers, then intercepts)"],
         notes="The strategy lives inside the opponent, not on the map. Rotating or relabelling the arena doesn't change it — that's what makes it opponent modelling, not perception."),
    dict(img="part2_intent_eval.png", title="Headline: each row isolates one effect (captures/episode)",
         bullets=["blind 2.68 · confident-WRONG guess 1.42 (−47%)",
                  "reactive belief 2.82 (+5%) · oracle 4.05 (+51%)",
                  "planner + belief 4.31 (+61%) — the full method",
                  "A wrong point estimate is WORSE than no model — carry the whole belief"],
         notes="The load-bearing number is −47%: a confident wrong guess is worse than knowing nothing. That's why the method must be uncertainty-aware. Encoder accuracy climbs 0.37→0.97 as it watches 3→25 steps."),
    dict(img="part2_planner.png", title="Planning on the belief beats the oracle",
         bullets=["planner 4.31 vs oracle-reactive 4.05 — lookahead beats reacting even with perfect info",
                  "Hedges while the belief is flat, commits as it sharpens",
                  "The belief contributes +40% over a flat-belief planner"],
         notes="The oracle is reactive with the true intent; we plan. Planning lets you pre-position for the interception, so we beat even the oracle. The flat-belief ablation proves the gain is the inference, not just lookahead."),
    dict(img="jepa_vs_vae_encoder.png", title="The encoder: predict the future, don't reconstruct",
         bullets=["Same trajectories, same 2-D latent, NO labels",
                  "JEPA probe 0.89 vs VAE 0.53 (3 seeds), matches a supervised classifier 0.85",
                  "Label-free belief drives the planner to 4.08"],
         notes="JEPA predicts the future representation via an EMA target, no decoder. A prey trajectory is mostly evasion noise plus a thin strategy signal; JEPA keeps the predictable part (the strategy), the VAE wastes capacity reconstructing noise."),
    dict(img="part4_jepa_world_planner.png", title="Honest negative: the same idea fails as a world model",
         bullets=["JEPA latent dynamics → 0.57 captures, worse than the state-space model (2.53)",
                  "Encoders want to DISCARD noise; a world model needs FIDELITY",
                  "Opposite requirements — reported straight, not tuned away"],
         notes="This delineates the principle: predict-don't-reconstruct helps the opponent encoder (detail = noise) and hurts the world model (detail = the physics the planner needs)."),
    dict(img="occupancy_heatmaps.png", title="\"Are circles just squares?\" — look at the shapes", wide=True,
         bullets=["Corners prey = 4 SHARP hot-spots. Circle prey = DIFFUSE, smeared ring",
                  "Different behaviours (cosine 0.42), but asymmetric — the circle signal is weak",
                  "That diffuseness is WHY it won't cluster and why vanilla BC struggles → motivates a snake placement"],
         notes="A meeting critic said 4 resources on a circle are geometrically a square of points. True — but the behaviours still differ: corners commit to 4 spots, circle smears. The circle class is the weak, diffuse one. This is the honest reason the encoder can't form 2 clean clusters."),
    dict(img="mopa_latent_resources_mappo_prey_occ.png", title="Can an encoder recover the placement?", wide=True,
         bullets=["Axes = the 2-D latent (or PC1/PC2 if higher-D); each point = one episode's occupancy, colored by placement (encoder never sees color)",
                  "LINEARLY decodable — supervised probe 0.93, latents up to 0.81",
                  "But does NOT cluster (GMM ARI ≈ 0): signal yes, separated modes no — the open problem"],
         notes="Supervised probe = cross-validated logistic regression WITH labels on the raw features = the ceiling. The latent probe swaps the input for the encoder's output. probe >> ARI is the whole 'decodable but not clustered' story."),
    dict(img="mopa_bc_vs_mappo.png", title="Is BC even a faithful clone? — deployed captures",
         bullets=["Vanilla BC 1.22 vs MAPPO expert 1.35, random 0.40",
                  "BC recovers 86% of the expert's edge over random",
                  "Cloning is NOT the bottleneck — the 14% gap is room for a strategy signal"],
         notes="This is captures, deployed in the game — the metric that matters, not one-step action accuracy. It sets up the latent-conditioning question: can a strategy signal close that 14%?"),
    dict(img=latent_bc, title="Latent-conditioned BC beats vanilla — once the latent carries the strategy",
         bullets=["Give the encoder more observation: VAE latent's probe climbs 0.60 → 0.78",
                  "BC tracks it — no gain at 25 steps, overtakes vanilla by 50, reaches the ORACLE ceiling by 125",
                  "JEPA's probe stays flat on occupancy → no gain (winner depends on the feature)"],
         notes="The gain tracks the probe almost exactly — encoder quality is the lever. NOTE if showing the sweep: this is action accuracy; the deployed-captures version is the same story on the performance metric."),
    dict(kind="text", title="Honest mechanism: is the gain just averaging?",
         bullets=["The placement is FIXED across a specialist's episodes",
                  "So 'more observation → better latent' = a better OCCUPANCY ESTIMATE (statistical averaging), not the strategy 'unfolding'",
                  "Contrast: the hidden-intent task genuinely reveals intent over time",
                  "→ We call it scaling-with-observation and do not overclaim"],
         notes="A fair pushback. Be upfront: for the fixed-placement task the improvement is estimation quality — you need enough observation to estimate a fixed strategy. That's still a real result, just a modest mechanism, and different from the dynamically-revealed intent task."),
    dict(img="demo_bc.gif", title="Where vanilla BC struggles",
         bullets=["Placement-blind clone can't tell a committing prey from a passing one",
                  "So it hedges and lets the prey reach its spot",
                  "Green ring = predicted the true action, red = missed; knowing placement turns hedging into interception"],
         notes="Show a corners episode: the naive clone misses the commit, the oracle-conditioned one follows. The GIF animates in slideshow mode."),
    dict(kind="text", title="How every plot is generated",
         bullets=["3 seeds, episode-level train/val splits, leakage-checked throughout",
                  "Occupancy = normalized 8×8 histogram of where the prey goes",
                  "Supervised probe = cross-validated logistic regression WITH labels (the ceiling); latent probe = same on the encoder output",
                  "Captures = raw per-capture reward / 10 (not the num_agents-scaled log return)",
                  "Full procedures: docs/METHODS.md"],
         notes="Emphasise rigour: nothing is action-accuracy hand-waving; the headline is deployed captures, and every number is re-derived from the raw npz."),
    dict(kind="text", title="Next steps",
         bullets=["Replace the diffuse circle with a genuinely distinct placement — a SNAKE path — so the behaviours actually cluster",
                  "Shape the latent to cluster: contrastive identity grounding (CTR, NeurIPS'24) or equivariance",
                  "Everything in deployed captures, not action accuracy"],
         notes="The heatmap makes the case for the snake placement (needs training two new specialists). The clustering fix is the CTR-style contrastive objective."),
]


def add_title_bar(slide, prs, text, color=DARK):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(EMW - 1), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = color
    return tb


def add_bullets(slide, bullets, left, top, width, height, size=16):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + b
        p.font.size = Pt(size); p.font.color.rgb = GREY; p.space_after = Pt(8)


def add_image_fit(slide, path, left, top, maxw, maxh):
    w, h = Image.open(path).size
    ar = w / h
    iw, ih = maxw, maxw / ar
    if ih > maxh:
        ih, iw = maxh, maxh * ar
    slide.shapes.add_picture(path, Inches(left + (maxw - iw) / 2),
                             Inches(top + (maxh - ih) / 2), Inches(iw), Inches(ih))


def main():
    prs = Presentation()
    prs.slide_width = Inches(EMW); prs.slide_height = Inches(EMH)
    blank = prs.slide_layouts[6]

    for s in SLIDES:
        slide = prs.slides.add_slide(blank)
        kind = s.get("kind", "figure")
        if kind == "title":
            tb = slide.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(EMW - 1.6), Inches(2))
            tf = tb.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.text = s["title"]
            p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = ACCENT
            p2 = tf.add_paragraph(); p2.text = s["sub"]
            p2.font.size = Pt(17); p2.font.color.rgb = GREY
        elif kind == "text":
            add_title_bar(slide, prs, s["title"], ACCENT)
            add_bullets(slide, s["bullets"], 1.0, 1.7, EMW - 2, EMH - 2.2, size=20)
        else:
            add_title_bar(slide, prs, s["title"])
            if s.get("wide"):
                add_image_fit(slide, os.path.join(PLOTS, s["img"]), 0.6, 1.25, EMW - 1.2, 4.4)
                add_bullets(slide, s["bullets"], 0.7, 5.75, EMW - 1.4, 1.5, size=14)
            else:
                add_image_fit(slide, os.path.join(PLOTS, s["img"]), 0.5, 1.35, 7.3, 5.5)
                add_bullets(slide, s["bullets"], 8.1, 1.7, 4.8, 5, size=16)
        slide.notes_slide.notes_text_frame.text = s["notes"]

    out = os.path.join(ROOT, "marl_opp_aware_results.pptx")
    prs.save(out)
    print(f"saved {out}  ({len(SLIDES)} slides; latent-BC fig = {latent_bc})")


if __name__ == "__main__":
    main()
