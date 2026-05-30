"""Demo: belief predators infer the prey's hidden corner and converge on it.

Side by side, same prey and same hidden intent:
  left  - intent-blind (unaware) predators: hedge, spread out
  right - uncertainty-aware (belief) predators fed the online intent posterior:
          the inferred corner sharpens and the pack commits to it.

A bar strip under the right panel shows the predator's belief over the 4 corners
updating as it observes the prey. Output: plots/demo_intent.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_tag_intent import SimpleTagIntentMPE
from generate_trajectory_dataset_resources import ActorLogits
from jaxmarl.wrappers.baselines import load_params
import part2_intent_eval as PE

PLOTDIR = "plots"
PRED = np.array([255, 75, 75]) / 255
PREY = np.array([75, 75, 255]) / 255
LAND = np.array([60, 60, 60]) / 255
CORNER_COLS = ["#0d6e7a", "#b85c10", "#1f7a4e", "#b44233"]
CORNERS = [(-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8), (0.8, 0.8)]
LIM = 1.4
BG = "#f8f7f4"


def logged_rollout(env, prey_p, pred_p, n_eps, rng, mode, enc_ks=None, enc_proba=None):
    """Like PE.rollout but also logs predator positions and the belief series."""
    advs = env.adversaries; prey = env.good_agents[0]
    sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    max_obs = max(sizes.values()); pred_base = sizes[advs[0]]
    isl = (pred_base - PE.K, pred_base)
    net = ActorLogits(action_dim=5, hidden_dim=128)
    reset = jax.vmap(env.reset); step = jax.vmap(env.step_env)
    rng, kr = jax.random.split(rng)
    obs, state = reset(jax.random.split(kr, n_eps))
    intent = np.asarray(state.intent)
    uniform = jnp.full((n_eps, PE.K), 1.0 / PE.K)
    prey_xy = [np.asarray(state.p_pos[:, env.num_adversaries])]
    pred_xy = [np.asarray(state.p_pos[:, :env.num_adversaries])]
    beliefs = []
    cache = {}

    def signal(t):
        if mode == "belief":
            usable = [k for k in enc_ks if k <= len(prey_xy)]
            if not usable:
                return uniform
            k = usable[-1]
            if k not in cache:
                cache[k] = jnp.asarray(enc_proba[k](np.stack(prey_xy[:k], 1)))
            return cache[k]
        return None

    for t in range(env.max_steps):
        sig = signal(t)
        beliefs.append(np.asarray(sig) if sig is not None else np.asarray(uniform))
        a_prey = net.apply(prey_p, PE._pad(obs[prey], max_obs)).argmax(-1).astype(jnp.int32)
        acts = {prey: a_prey}
        for a in advs:
            pin = PE._pad(obs[a], max_obs)
            if sig is not None and env.reveal_to_pred:
                pin = pin.at[:, isl[0]:isl[1]].set(sig)
            acts[a] = net.apply(pred_p, pin).argmax(-1).astype(jnp.int32)
        rng, ks = jax.random.split(rng)
        obs, state, rew, dones, info = step(jax.random.split(ks, n_eps), state, acts)
        prey_xy.append(np.asarray(state.p_pos[:, env.num_adversaries]))
        pred_xy.append(np.asarray(state.p_pos[:, :env.num_adversaries]))
    return (np.stack(prey_xy, 1), np.stack(pred_xy, 1), intent,
            np.stack(beliefs, 1) if beliefs else None)


def setup(ax, title):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for (cx, cy), col in zip(CORNERS, CORNER_COLS):
        ax.add_patch(Circle((cx, cy), 0.35, fill=False, ls="--", ec=col, lw=1, alpha=0.5))
    for ox, oy in [(0.5, 0.5), (-0.5, -0.5)]:
        ax.add_patch(Circle((ox, oy), 0.1, color=LAND, alpha=0.4))
    ax.set_title(title, fontsize=12, fontweight="bold")


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env = PE.make_env(reveal=True)
    # encoders (soft) for the online belief
    pos, intent = [], []
    for i in PE.SEEDS:
        P, g = PE.gather_prey_traj(env,
                                   load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                   load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                   120, jax.random.PRNGKey(i))
        pos.append(P); intent.append(g)
    pos = np.concatenate(pos); intent = np.concatenate(intent)
    ks = [3, 5, 8, 12, 18, 25]; enc_proba = {}
    for k in ks:
        _, _, _, soft = PE.train_encoder_at_k(pos, intent, k)
        enc_proba[k] = soft

    prey_be = load_params(PE.apath("mappo_intent_belief", "prey", 0))
    pred_be = load_params(PE.apath("mappo_intent_belief", "pred", 0))
    prey_un = load_params(PE.apath("mappo_intent_unaware", "prey", 0))
    pred_un = load_params(PE.apath("mappo_intent_unaware", "pred", 0))
    env_un = PE.make_env(reveal=False)

    # find an episode where the belief predator catches the prey (good demo)
    pxy_b, dxy_b, intent_b, bel_b = logged_rollout(
        env, prey_be, pred_be, 64, jax.random.PRNGKey(7), "belief",
        enc_ks=ks, enc_proba=enc_proba)
    pxy_u, dxy_u, intent_u, _ = logged_rollout(
        env_un, prey_un, pred_un, 64, jax.random.PRNGKey(7), "none")
    # choose an episode whose final belief is confident and correct
    final_conf = bel_b[:, -1].max(1)
    correct = bel_b[:, -1].argmax(1) == intent_b
    cand = np.where(correct)[0]
    e = cand[np.argmax(final_conf[cand])] if len(cand) else 0

    fig = plt.figure(figsize=(11.5, 6.2)); fig.patch.set_facecolor(BG)
    axL = fig.add_axes([0.03, 0.16, 0.44, 0.76])
    axR = fig.add_axes([0.50, 0.16, 0.44, 0.76])
    axb = fig.add_axes([0.50, 0.04, 0.44, 0.08])
    setup(axL, "intent-blind predators (hedge)")
    setup(axR, "belief predators (infer + commit)")
    gi = int(intent_b[e]); star = CORNERS[gi]
    for ax in (axL, axR):
        ax.scatter(*star, s=240, marker="*", c=CORNER_COLS[gi], edgecolors="k", zorder=6)

    def make(ax, pxy, dxy):
        pc = ax.add_patch(Circle(pxy[e, 0], 0.05, color=PREY, ec="white", lw=1, zorder=10))
        dc = [ax.add_patch(Circle(dxy[e, 0, i], 0.075, color=PRED, ec="white", lw=1, zorder=10))
              for i in range(3)]
        tr, = ax.plot([], [], "-", color=PREY, alpha=0.4, lw=2)
        return pc, dc, tr
    pcL, dcL, trL = make(axL, pxy_u, dxy_u)
    pcR, dcR, trR = make(axR, pxy_b, dxy_b)

    axb.set_xlim(-0.5, 3.5); axb.set_ylim(0, 1); axb.set_xticks(range(4))
    axb.set_xticklabels(["SW", "NW", "SE", "NE"], fontsize=8)
    axb.set_yticks([]); axb.set_title("predator's belief over corners", fontsize=9)
    bars = axb.bar(range(4), [0.25] * 4, color=CORNER_COLS)

    T = pxy_b.shape[1]

    def upd(t):
        for (pc, dc, tr, pxy, dxy) in [(pcL, dcL, trL, pxy_u, dxy_u),
                                       (pcR, dcR, trR, pxy_b, dxy_b)]:
            pc.center = pxy[e, t]
            for i in range(3):
                dc[i].center = dxy[e, t, i]
            lo = max(0, t - 10)
            tr.set_data(pxy[e, lo:t + 1, 0], pxy[e, lo:t + 1, 1])
        b = bel_b[e, min(t, T - 2)]
        for j, bar in enumerate(bars):
            bar.set_height(b[j])
    ani = animation.FuncAnimation(fig, upd, frames=T, interval=180)
    out = os.path.join(PLOTDIR, "demo_intent.gif")
    ani.save(out, writer="pillow", fps=6, dpi=110)
    plt.close(fig)
    print(f"saved {out} ({os.path.getsize(out)/1024:.0f} KB)  intent={gi}")


if __name__ == "__main__":
    main()
