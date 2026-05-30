"""Demo renderer for Experiment 5: specialist routes, brittleness, and an
annotated explainer video.

Produces (in plots/):
  demo_routes_<alg>.gif        circle- vs corners-specialist rollout, side by side
  demo_brittleness_mappo.gif   same prey vs its in-dist predator and an OOD predator
  exp5_explainer.mp4           narrated multi-scene walkthrough of D1/D2/D3

Live frames come from rollout_one_checkpoint (numpy position arrays); result
cards embed the static analysis PNGs. Text is overlaid on every scene.

Usage:
    python src/exp5_demo.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import rollout_one_checkpoint

PLOTDIR = "plots"
PRED_COLOR = np.array([255, 75, 75]) / 255
PREY_COLOR = np.array([75, 75, 255]) / 255
LAND_COLOR = np.array([60, 60, 60]) / 255
RES_LIVE = np.array([50, 205, 50]) / 255
RES_DEAD = np.array([200, 200, 200]) / 255
LIM = 1.6
OBST = [[0.5, 0.5], [-0.5, -0.5]]
BG = "#f8f7f4"

# weakened-predator regime used for the Exp 5 specialists
WP = dict(pred_max_speed=0.6, pred_accel=1.5, collect_reward=10.0)


def roll(algorithm, placement, seed=0, steps=50, n_eps=24, eval_seed=3):
    """One specialist rollout. Returns dict of numpy arrays for episode `pick`."""
    G.NUM_STEPS = steps
    G.PRED_MAX_SPEED = WP["pred_max_speed"]
    G.PRED_ACCEL = WP["pred_accel"]
    G.COLLECT_REWARD = WP["collect_reward"]
    rng = jax.random.PRNGKey(eval_seed)
    return rollout_one_checkpoint(algorithm, placement + "_wp", seed, n_eps,
                                  placement, rng)


def cross_roll(algorithm, pred_placement, prey_placement, seed=0, steps=50,
               n_eps=24, eval_seed=5):
    """Rollout with predator and prey loaded from different specialists."""
    G.NUM_STEPS = steps
    G.PRED_MAX_SPEED = WP["pred_max_speed"]
    G.PRED_ACCEL = WP["pred_accel"]
    G.COLLECT_REWARD = WP["collect_reward"]
    # monkeypatch params loader to mix predator/prey origins
    orig = G._params_path
    def mixed(alg, placement, team, s):
        which = pred_placement if team == "pred" else prey_placement
        return orig(alg, which + "_wp", team, s)
    G._params_path = mixed
    try:
        rng = jax.random.PRNGKey(eval_seed)
        d = rollout_one_checkpoint(algorithm, prey_placement + "_wp", seed,
                                   n_eps, prey_placement, rng)
    finally:
        G._params_path = orig
    return d


# --------------------------------------------------------------------------- #
# low-level frame drawing
# --------------------------------------------------------------------------- #

def episode_stats(d, steps):
    """Mean captures/25-step-episode over all parallel episodes, and the index
    of the episode closest to that mean (a representative one to animate)."""
    caps = d["pred_rewards"][:, :, 0].sum(1) / 10.0 / (steps / 25.0)
    mean = float(caps.mean())
    rep = int(np.argmin(np.abs(caps - mean)))
    return mean, rep


def setup_axis(ax, title=None):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal"); ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for ox, oy in OBST:
        ax.add_patch(Circle((ox, oy), 0.1, color=LAND_COLOR, alpha=0.45, zorder=5))
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)


class Scene:
    """Renders a single episode of a rollout dict into an axis, frame by frame."""
    def __init__(self, ax, d, ep, trail_decay=18):
        self.ax = ax
        self.prey = d["positions"][ep]            # (T+1, 2)
        self.preds = d["pred_positions"][ep]      # (T+1, 3, 2)
        self.res = d["resource_pos"][ep]          # (T+1, 4, 2)
        self.coll = d["collected"][ep]            # (T+1, 4)
        self.pred_r = d["pred_rewards"][ep]       # (T, 3)
        self.T = self.prey.shape[0]
        self.trail_decay = trail_decay
        self.captures = 0.0

        self.pred_circ = [ax.add_patch(Circle(self.preds[0, i], 0.075,
                          color=PRED_COLOR, ec="white", lw=1.1, zorder=10))
                          for i in range(3)]
        self.prey_circ = ax.add_patch(Circle(self.prey[0], 0.05, color=PREY_COLOR,
                                      ec="white", lw=1.1, zorder=10))
        self.res_mark = [ax.add_patch(RegularPolygon(self.res[0, j], numVertices=4,
                         radius=0.06, orientation=np.pi/4, color=RES_LIVE,
                         ec="white", lw=0.8, zorder=8)) for j in range(4)]
        self.trail, = ax.plot([], [], "-", color=PREY_COLOR, alpha=0.35, lw=2, zorder=3)

    def update(self, t):
        t = min(t, self.T - 1)
        for i, c in enumerate(self.pred_circ):
            c.center = self.preds[t, i]
        self.prey_circ.center = self.prey[t]
        for j, m in enumerate(self.res_mark):
            dead = bool(self.coll[t, j])
            m.xy = self.res[t, j]
            m.set_color(RES_DEAD if dead else RES_LIVE)
            m.set_alpha(0.3 if dead else 1.0)
        lo = max(0, t - self.trail_decay)
        seg = self.prey[lo:t + 1].copy()
        if len(seg) > 1:                      # break the line at auto-reset jumps
            jumps = np.linalg.norm(np.diff(seg, axis=0), axis=1) > 0.5
            seg[1:][jumps] = np.nan
        self.trail.set_data(seg[:, 0], seg[:, 1])


def save_anim(fig, update, frames, outpath, fps=10, dpi=120):
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if outpath.endswith(".mp4"):
        ani.save(outpath, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        ani.save(outpath, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"  saved {outpath} ({os.path.getsize(outpath)/1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# standalone GIFs
# --------------------------------------------------------------------------- #

def routes_gif(algorithm="iql"):
    dc = roll(algorithm, "circle", steps=50)
    dk = roll(algorithm, "corners", steps=50)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    fig.patch.set_facecolor(BG)
    setup_axis(axes[0], "circle layout — prey loops the ring")
    setup_axis(axes[1], "corners layout — prey patrols the perimeter")
    sc = [Scene(axes[0], dc, 0, trail_decay=50), Scene(axes[1], dk, 0, trail_decay=50)]
    fig.suptitle(f"{algorithm.upper()} specialist prey learns a distinct route per layout",
                 fontweight="bold", fontsize=13)
    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PRED_COLOR, ms=10, label='predator'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor=PREY_COLOR, ms=10, label='prey'),
              Line2D([0], [0], marker='D', color='w', markerfacecolor=RES_LIVE, ms=8, label='resource')]
    axes[1].legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.85)
    T = sc[0].T

    def update(t):
        for s in sc:
            s.update(t)
    save_anim(fig, update, T, os.path.join(PLOTDIR, f"demo_routes_{algorithm}.gif"))


def brittleness_gif():
    # same corners-prey, left: its co-trained corners predators; right: OOD circle predators
    steps = 75
    d_in = cross_roll("mappo", "corners", "corners", steps=steps)
    d_ood = cross_roll("mappo", "circle", "corners", steps=steps)
    mean_in, rep_in = episode_stats(d_in, steps)
    mean_ood, rep_ood = episode_stats(d_ood, steps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.7))
    fig.patch.set_facecolor(BG)
    setup_axis(axes[0], "corners-prey vs its co-trained predators")
    setup_axis(axes[1], "same prey vs an UNSEEN (circle) predator")
    s_in = Scene(axes[0], d_in, rep_in, trail_decay=14)
    s_ood = Scene(axes[1], d_ood, rep_ood, trail_decay=14)
    # aggregate result over all 24 parallel episodes (a single episode is noisy)
    axes[0].text(-LIM + 0.1, LIM - 0.12, f"avg {mean_in:.1f} captures/ep", fontsize=12,
                 va="top", fontfamily="monospace", color="#1f7a4e", fontweight="bold")
    axes[1].text(-LIM + 0.1, LIM - 0.12, f"avg {mean_ood:.1f} captures/ep", fontsize=12,
                 va="top", fontfamily="monospace", color="#b44233", fontweight="bold")
    fig.suptitle("Deliverable 3 — predators are brittle to an opponent they never co-trained with",
                 fontweight="bold", fontsize=12.5)
    T = s_in.T

    def update(t):
        s_in.update(t); s_ood.update(t)
    save_anim(fig, update, T, os.path.join(PLOTDIR, "demo_brittleness_mappo.gif"))


# --------------------------------------------------------------------------- #
# annotated explainer MP4
# --------------------------------------------------------------------------- #

def text_card(ax, lines, sizes=None, colors=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(lines)
    ys = np.linspace(0.74, 0.26, n) if n > 1 else [0.5]
    for i, ln in enumerate(lines):
        ax.text(0.5, ys[i], ln, ha="center", va="center",
                fontsize=(sizes[i] if sizes else 18),
                color=(colors[i] if colors else "#171717"),
                fontweight=("bold" if i == 0 else "normal"), wrap=True)


def explainer():
    fps = 10
    # pre-roll the live scenes we need
    dc = roll("iql", "circle", steps=50)
    dk = roll("iql", "corners", steps=50)
    d_in = cross_roll("mappo", "corners", "corners", steps=75)
    d_ood = cross_roll("mappo", "circle", "corners", steps=75)
    mean_in, rep_in = episode_stats(d_in, 75)
    mean_ood, rep_ood = episode_stats(d_ood, 75)

    # load result PNGs
    img_d1 = mpimg.imread(os.path.join(PLOTDIR, "exp5_specialist_iql_wp.png"))
    img_d2 = mpimg.imread(os.path.join(PLOTDIR, "exp5b_encoder_length_iql_wp.png"))
    img_d3 = mpimg.imread(os.path.join(PLOTDIR, "brittleness_mappo_wp.png"))

    # scene script: (kind, duration_seconds, payload)
    script = [
        ("card", 3.0, dict(lines=[
            "Adaptive Opponent Modeling for Adversarial Co-Training",
            "Three preconditions for the proposal — demonstrated",
            "predator–prey + resources · JAX · MPE simple_tag"],
            sizes=[22, 16, 13], colors=["#0d6e7a", "#3b3b3b", "#6b6b6b"])),
        ("card", 4.0, dict(lines=[
            "The task",
            "3 predators chase 1 prey. 4 resources only the prey can collect.",
            "Resources appear on a CIRCLE or at the CORNERS.",
            "Question: does the layout induce a different prey strategy?"],
            sizes=[20, 15, 15, 15],
            colors=["#171717", "#3b3b3b", "#3b3b3b", "#b85c10"])),
        ("live2", 5.0, dict(d=[dc, dk], eps=[0, 0],
            titles=["circle layout", "corners layout"],
            caption="Deliverable 1 — the prey learns a route that follows the layout")),
        ("img", 4.0, dict(img=img_d1,
            caption="Pooled occupancy: a diamond vs a square — separable at 0.98 (chance 0.50)")),
        ("img", 4.5, dict(img=img_d2,
            caption="Deliverable 2 — an unsupervised VAE latent recovers the strategy (ARI 0.87)")),
        ("live2", 5.0, dict(d=[d_in, d_ood], eps=[rep_in, rep_ood],
            titles=["vs co-trained predator", "vs UNSEEN predator"],
            caption="Deliverable 3 — same prey, swap the predator it faces",
            cap_labels=[f"avg {mean_in:.1f} captures/ep", f"avg {mean_ood:.1f} captures/ep"])),
        ("img", 4.0, dict(img=img_d3,
            caption="The co-trained predator catches 43% more than the unfamiliar one")),
        ("card", 4.0, dict(lines=[
            "Putting it together",
            "(1) distinct strategies   (2) an encodable latent z   (3) a brittle opponent",
            "→ Part 1: condition an opponent model on z",
            "→ Part 2: sample it inside a model-based planner"],
            sizes=[20, 15, 15, 15],
            colors=["#0d6e7a", "#171717", "#1f7a4e", "#1f7a4e"])),
    ]

    # build a flat frame plan
    plan = []   # list of (kind, payload, local_frame, total_local)
    for kind, dur, payload in script:
        nf = int(dur * fps)
        for lf in range(nf):
            plan.append((kind, payload, lf, nf))
    total = len(plan)

    fig = plt.figure(figsize=(11, 6.2))
    fig.patch.set_facecolor("white")

    state = {"kind": None, "axes": None, "scenes": None, "cap": None}

    def clear():
        fig.clear()

    def update(fr):
        kind, payload, lf, nf = plan[fr]
        # (re)build the figure layout when the scene changes
        new_scene = (fr == 0) or (plan[fr - 1][1] is not payload)
        if new_scene:
            clear()
            if kind == "card":
                ax = fig.add_axes([0, 0, 1, 1])
                text_card(ax, payload["lines"], payload.get("sizes"), payload.get("colors"))
                state["kind"] = "card"
            elif kind == "img":
                ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
                ax.imshow(payload["img"]); ax.axis("off")
                fig.text(0.5, 0.95, payload["caption"], ha="center", va="top",
                         fontsize=15, fontweight="bold", color="#0d6e7a")
                state["kind"] = "img"
            elif kind == "live2":
                axL = fig.add_axes([0.04, 0.08, 0.43, 0.78])
                axR = fig.add_axes([0.53, 0.08, 0.43, 0.78])
                setup_axis(axL, payload["titles"][0])
                setup_axis(axR, payload["titles"][1])
                eps = payload.get("eps", [0, 0])
                scenes = [Scene(axL, payload["d"][0], eps[0], trail_decay=16),
                          Scene(axR, payload["d"][1], eps[1], trail_decay=16)]
                fig.text(0.5, 0.955, payload["caption"], ha="center", va="top",
                         fontsize=15, fontweight="bold", color="#0d6e7a")
                if payload.get("cap_labels"):
                    for ax, lab, col in zip([axL, axR], payload["cap_labels"],
                                            ["#1f7a4e", "#b44233"]):
                        ax.text(-LIM + 0.1, LIM - 0.12, lab, fontsize=12, va="top",
                                fontfamily="monospace", color=col, fontweight="bold")
                state.update(kind="live2", scenes=scenes)
        # per-frame animation for live scenes
        if state["kind"] == "live2":
            for sc in state["scenes"]:
                t = int(lf / nf * (sc.T - 1))
                sc.update(t)

    save_anim(fig, update, total, os.path.join(PLOTDIR, "exp5_explainer.mp4"),
              fps=fps, dpi=120)


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    print("routes GIF (iql)…");        routes_gif("iql")
    print("brittleness GIF (mappo)…"); brittleness_gif()
    print("explainer MP4…");           explainer()


if __name__ == "__main__":
    main()
