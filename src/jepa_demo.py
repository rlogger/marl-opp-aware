"""Demo: the JEPA opponent model running in predator-prey.

The predators are fed the LABEL-FREE JEPA belief (encoder -> readout to a
predicted arrival position -> softmax over the four known corners). Side by side:
  left  - intent-blind predators (hedge, spread out)
  right - JEPA-belief predators: a hollow marker shows where the JEPA model
          predicts the prey is heading; it locks onto the prey's hidden corner as
          the pack closes in. A bar strip shows the JEPA belief sharpening.

Output: plots/demo_jepa.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_trajectory_dataset_resources import ActorLogits
from jaxmarl.wrappers.baselines import load_params
import part2_intent_eval as PE
from part2_jepa_planner import train_jepa_belief, KMAX, ARRIVE, SIGMA

PLOTDIR = "plots"
PRED = np.array([255, 75, 75]) / 255
PREY = np.array([75, 75, 255]) / 255
LAND = np.array([60, 60, 60]) / 255
CORNER_COLS = ["#0d6e7a", "#b85c10", "#1f7a4e", "#b44233"]
CORNERS = [(-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8), (0.8, 0.8)]
LIM = 1.4; BG = "#f8f7f4"


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
    env = PE.make_env(reveal=True); env_un = PE.make_env(reveal=False)
    corners = np.asarray(env.corners)

    # 1. train the label-free JEPA belief (encoder + arrival readout)
    pos, intent = [], []
    for i in PE.SEEDS:
        P, g = PE.gather_prey_traj(env, load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                   load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                   120, jax.random.PRNGKey(i))
        pos.append(P); intent.append(g)
    pos = np.concatenate(pos).astype(np.float32); intent = np.concatenate(intent).astype(int)
    full = pos[:, :KMAX].reshape(len(pos), -1)
    mu, sd = full.mean(0), full.std(0) + 1e-6
    full_std = ((full - mu) / sd).reshape(len(pos), KMAX, 2)
    arrive = pos[:, ARRIVE[0]:ARRIVE[1]].mean(1)
    print("training JEPA belief...")
    enc, rd, p = train_jepa_belief(full_std, arrive, jax.random.PRNGKey(0))

    def belief_pred(P):                                   # P:(n,t,2) -> belief(n,4), pred(n,2)
        n, t, _ = P.shape; tt = min(t, KMAX)
        buf = np.zeros((n, KMAX, 2), np.float32); buf[:, :tt] = P[:, :tt]
        std = ((buf.reshape(n, -1) - mu) / sd).reshape(n, KMAX, 2)
        m = (np.arange(KMAX)[None, :] < tt)[:, :, None]
        z = enc.apply(p["e"], jnp.asarray(np.where(m, std, 0.0).reshape(n, -1)))
        pr = np.asarray(rd.apply(p["r"], z))
        d2 = ((pr[:, None, :] - corners[None]) ** 2).sum(-1)
        return np.asarray(jax.nn.softmax(jnp.asarray(-d2 / SIGMA), -1)), pr

    # 2. rollouts: blind predators, and JEPA-belief predators (logging belief + prediction)
    net = ActorLogits(action_dim=5, hidden_dim=128)

    def rollout(env_, prey_p, pred_p, n, rng, use_jepa):
        sizes = {a: env_.observation_space(a).shape[0] for a in env_.agents}
        max_obs = max(sizes.values()); advs = env_.adversaries; prey = env_.good_agents[0]
        pb = sizes[advs[0]]; isl = (pb - PE.K, pb)
        reset = jax.vmap(env_.reset); step = jax.vmap(env_.step_env)
        rng, kr = jax.random.split(rng); obs, st = reset(jax.random.split(kr, n))
        intent = np.asarray(st.intent)
        pxy = [np.asarray(st.p_pos[:, env_.num_adversaries])]
        dxy = [np.asarray(st.p_pos[:, :env_.num_adversaries])]
        bels, preds, caps = [], [], np.zeros(n)
        for t in range(env_.max_steps):
            if use_jepa:
                b, pr = belief_pred(np.stack(pxy, 1)); bels.append(b); preds.append(pr)
            a_prey = net.apply(prey_p, PE._pad(obs[prey], max_obs)).argmax(-1).astype(jnp.int32)
            acts = {prey: a_prey}
            for a in advs:
                pin = PE._pad(obs[a], max_obs)
                if use_jepa:
                    pin = pin.at[:, isl[0]:isl[1]].set(jnp.asarray(b))
                acts[a] = net.apply(pred_p, pin).argmax(-1).astype(jnp.int32)
            rng, kss = jax.random.split(rng)
            obs, st, rew, _, _ = step(jax.random.split(kss, n), st, acts)
            caps += np.asarray(rew[advs[0]]) / 10.0
            pxy.append(np.asarray(st.p_pos[:, env_.num_adversaries]))
            dxy.append(np.asarray(st.p_pos[:, :env_.num_adversaries]))
        return (np.stack(pxy, 1), np.stack(dxy, 1), intent,
                (np.stack(bels, 1) if bels else None),
                (np.stack(preds, 1) if preds else None), caps)

    pxy_b, dxy_b, int_b, bel_b, pred_b, cap_b = rollout(
        env, load_params(PE.apath("mappo_intent_belief", "prey", 0)),
        load_params(PE.apath("mappo_intent_belief", "pred", 0)), 96, jax.random.PRNGKey(11), True)
    pxy_u, dxy_u, _, _, _, _ = rollout(
        env_un, load_params(PE.apath("mappo_intent_unaware", "prey", 0)),
        load_params(PE.apath("mappo_intent_unaware", "pred", 0)), 96, jax.random.PRNGKey(11), False)

    # lengthened, detailed demo: one strong episode PER corner, played back to back,
    # so all four intents and the belief sharpening are visible.
    correct = bel_b[:, -1].argmax(1) == int_b
    score = correct.astype(float) * (bel_b[:, -1].max(1) + 0.3 * cap_b)
    eps = []
    for g in range(4):
        cand = np.where(int_b == g)[0]
        if len(cand):
            eps.append(int(cand[np.argmax(score[cand])]))
    if not eps:
        eps = [int(np.argmax(score))]
    print(f"  demo episodes {eps}  intents {[int(int_b[e]) for e in eps]}")
    CNAMES = ["SW", "NW", "SE", "NE"]

    fig = plt.figure(figsize=(11.5, 6.7)); fig.patch.set_facecolor(BG)
    axL = fig.add_axes([0.03, 0.15, 0.44, 0.74]); axR = fig.add_axes([0.50, 0.15, 0.44, 0.74])
    axb = fig.add_axes([0.50, 0.045, 0.44, 0.065])
    setup(axL, "intent-blind predators (hedge)")
    setup(axR, "JEPA-belief predators (infer + intercept)")
    starL = axL.scatter([0], [0], s=280, marker="*", c="#111", zorder=6)
    starR = axR.scatter([0], [0], s=280, marker="*", c="#111", zorder=6)
    e0 = eps[0]

    def make(ax, pxy, dxy):
        pc = ax.add_patch(Circle(pxy[e0, 0], 0.05, color=PREY, ec="white", lw=1, zorder=10))
        dc = [ax.add_patch(Circle(dxy[e0, 0, i], 0.075, color=PRED, ec="white", lw=1, zorder=10))
              for i in range(3)]
        tr, = ax.plot([], [], "-", color=PREY, alpha=0.4, lw=2)
        return pc, dc, tr
    pcL, dcL, trL = make(axL, pxy_u, dxy_u)
    pcR, dcR, trR = make(axR, pxy_b, dxy_b)
    jx = axR.scatter([pred_b[e0, 0, 0]], [pred_b[e0, 0, 1]], s=170, marker="X",
                     facecolors="none", edgecolors="#111", linewidths=2, zorder=12,
                     label="JEPA's predicted destination")
    axR.legend(loc="upper left", fontsize=8, framealpha=0.85)
    caption = fig.text(0.5, 0.955, "", ha="center", fontsize=12, fontweight="bold")

    axb.set_xlim(-0.5, 3.5); axb.set_ylim(0, 1); axb.set_xticks(range(4))
    axb.set_xticklabels(CNAMES, fontsize=8); axb.set_yticks([])
    axb.set_title("JEPA belief over corners (label-free)", fontsize=9)
    bars = axb.bar(range(4), [0.25] * 4, color=CORNER_COLS)

    T = pxy_b.shape[1]
    HOLD = 6                                   # linger at the capture
    per = T + HOLD
    frames = per * len(eps)

    def upd(f):
        k = min(f // per, len(eps) - 1); e = eps[k]
        t = min(f - k * per, T - 1)
        gi = int(int_b[e])
        starL.set_offsets([CORNERS[gi]]); starR.set_offsets([CORNERS[gi]])
        for pc, dc, tr, pxy, dxy in [(pcL, dcL, trL, pxy_u, dxy_u), (pcR, dcR, trR, pxy_b, dxy_b)]:
            pc.center = pxy[e, t]
            for i in range(3):
                dc[i].center = dxy[e, t, i]
            lo = max(0, t - 12); tr.set_data(pxy[e, lo:t + 1, 0], pxy[e, lo:t + 1, 1])
        ti = min(t, bel_b.shape[1] - 1)
        jx.set_offsets([pred_b[e, ti]])
        for j, bar in enumerate(bars):
            bar.set_height(bel_b[e, ti, j])
        conf = bel_b[e, ti].max()
        caption.set_text(f"episode {k+1}/{len(eps)}   ·   true corner {CNAMES[gi]}   "
                         f"·   step {t+1}/{T}   ·   JEPA belief {conf:.0%} confident")
    ani = animation.FuncAnimation(fig, upd, frames=frames, interval=200)
    out = os.path.join(PLOTDIR, "demo_jepa.gif")
    ani.save(out, writer="pillow", fps=5, dpi=118); plt.close(fig)
    print(f"saved {out} ({os.path.getsize(out)/1024:.0f} KB, {frames} frames)")


if __name__ == "__main__":
    main()
