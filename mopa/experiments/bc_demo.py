"""Demo GIF for the BC result: what 'held-out action accuracy' looks like.

A held-out episode replays (true positions, true actions); at every step each
BC variant predicts each predator's action. Predators flash GREEN when the
variant predicted the action they actually took and RED when it missed, with a
running accuracy counter. Side by side: naive pi(a|s) vs oracle
pi(a|s, placement) -- the two ends of the paper's BC table -- one circle
episode (top row) and one corners episode (bottom row), chosen where the
conditioning matters most.

Output: plots/demo_bc.gif
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation

import jax
import jax.numpy as jnp
import optax

from mopa import legacy
from mopa.bc import BCNet, build_samples, BC_BATCH, BC_STEPS
from mopa.data import specialist_dataset

PRED = np.array([255, 75, 75]) / 255
PREY = np.array([75, 75, 255]) / 255
LAND = np.array([60, 60, 60]) / 255
OK, BAD = "#1f7a4e", "#b44233"
LIM = 1.4
BG = "#f8f7f4"
CTX = 26                       # episode 2 starts here (reset at t=25)


def train_bc(S, A, seed=0, steps=BC_STEPS):
    """Train a BC net on standardized S; return (net, params, mu, sd)."""
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Sn = (S - mu) / sd
    net = BCNet()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    params = net.init(ki, Sn[:1])
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Sj, Aj = jnp.asarray(Sn), jnp.asarray(A)
    n = len(Sn)

    def loss_fn(p, s, a):
        return optax.softmax_cross_entropy_with_integer_labels(
            net.apply(p, s), a).mean()

    @jax.jit
    def upd(params, opt, idx):
        g = jax.grad(loss_fn)(params, Sj[idx], Aj[idx])
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for _ in range(steps):
        key, bk = jax.random.split(key)
        idx = jax.random.choice(bk, n, (min(BC_BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx)
    return net, params, mu, sd


def setup(ax, title, placement):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if placement == 0:                                  # circle, r = 0.6
        ax.add_patch(Circle((0, 0), 0.6, fill=False, ls="--",
                            ec="#0d6e7a", lw=1.2, alpha=0.6))
    else:                                               # corners, offset 0.8
        for cx in (-0.8, 0.8):
            for cy in (-0.8, 0.8):
                ax.add_patch(Rectangle((cx - 0.07, cy - 0.07), 0.14, 0.14,
                                       fill=False, ec="#b85c10", lw=1.2,
                                       alpha=0.7))
    for ox, oy in [(0.5, 0.5), (-0.5, -0.5)]:
        ax.add_patch(Circle((ox, oy), 0.1, color=LAND, alpha=0.4))
    ax.set_title(title, fontsize=10.5, fontweight="bold")


def main():
    os.makedirs(legacy.PLOTDIR, exist_ok=True)
    print("rolling out specialists...")
    ds = specialist_dataset(n_eps=64, ckpt_seeds=(0,))
    y = ds["label"]
    S0, A, ep = build_samples(ds, CTX)
    oracle_feat = np.eye(2, dtype=np.float32)[y]

    # episode-level split: train both variants on the same train episodes
    rng = np.random.RandomState(0)
    eps = np.unique(ep)
    val_eps = rng.choice(eps, int(len(eps) * 0.25), replace=False)
    tr = ~np.isin(ep, val_eps)

    print("training naive BC...")
    netN, pN, muN, sdN = train_bc(S0[tr], A[tr])
    print("training oracle BC...")
    So = np.concatenate([S0, oracle_feat[ep]], -1).astype(np.float32)
    netO, pO, muO, sdO = train_bc(So[tr], A[tr])

    def preds(net, p, mu, sd, S):
        return np.asarray(net.apply(p, jnp.asarray((S - mu) / sd))).argmax(-1)

    pn, po = preds(netN, pN, muN, sdN, S0), preds(netO, pO, muO, sdO, So)
    okN, okO = (pn == A), (po == A)

    # per-episode accuracy gap on held-out episodes; pick one per placement
    picks = {}
    for lab in (0, 1):
        best, gap = None, -1
        for e in val_eps:
            if y[e] != lab:
                continue
            m = ep == e
            g = okO[m].mean() - okN[m].mean()
            if g > gap:
                best, gap = e, g
        picks[lab] = (int(best),
                      float(okN[ep == best].mean()),
                      float(okO[ep == best].mean()))
        print(f"  {'circle' if lab == 0 else 'corners'} demo ep {best}: "
              f"naive {picks[lab][1]:.2f}, oracle {picks[lab][2]:.2f}")

    # per-(episode, step, predator) correctness lookup
    steps = sorted({t for t in range(CTX, 50)})
    n_eps_total = len(y)

    def grid(ok):
        g = np.zeros((n_eps_total, 50, 3), bool)
        i = 0
        for t in steps:                       # build_samples ordering: t outer,
            for pidx in range(3):             # predator inner, episodes vector
                g[:, t, pidx] = ok[i * n_eps_total:(i + 1) * n_eps_total]
                i += 1
        return g
    gN, gO = grid(okN), grid(okO)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.2))
    fig.patch.set_facecolor(BG)
    artists = []
    for r, lab in enumerate((0, 1)):
        e, accN, accO = picks[lab]
        pname = "circle" if lab == 0 else "corners"
        for c, (g, nm, acc) in enumerate(
                [(gN, "naive  π(a|s)", accN),
                 (gO, "oracle  π(a|s, placement)", accO)]):
            ax = axes[r, c]
            setup(ax, f"{pname} episode — {nm}\nepisode accuracy {acc:.2f}",
                  lab)
            pc = ax.add_patch(Circle(ds["prey_pos"][e, CTX], 0.05, color=PREY,
                                     ec="white", lw=1, zorder=10))
            dc = [ax.add_patch(Circle(ds["pred_pos"][e, CTX, i], 0.075,
                                      color=PRED, ec="white", lw=2.2,
                                      zorder=10)) for i in range(3)]
            tr_, = ax.plot([], [], "-", color=PREY, alpha=0.4, lw=2)
            txt = ax.text(0.03, 0.03, "", transform=ax.transAxes, fontsize=9,
                          fontfamily="monospace")
            artists.append((ax, e, g, pc, dc, tr_, txt))

    T0, T1 = CTX, 50

    def upd(fr):
        t = T0 + fr
        for ax, e, g, pc, dc, tr_, txt in artists:
            pc.center = ds["prey_pos"][e, t]
            for i in range(3):
                dc[i].center = ds["pred_pos"][e, t, i]
                dc[i].set_edgecolor(OK if g[e, t, i] else BAD)
            lo = max(T0, t - 8)
            tr_.set_data(ds["prey_pos"][e, lo:t + 1, 0],
                         ds["prey_pos"][e, lo:t + 1, 1])
            window = g[e, T0:t + 1].reshape(-1)
            txt.set_text(f"step {t-T0+1:2d}/24   running acc "
                         f"{window.mean():.2f}")
    fig.suptitle("What the BC table measures: per-step action prediction on a "
                 "held-out episode\n(green ring = predicted the predator's "
                 "actual action, red = missed)", fontweight="bold", fontsize=12)
    ani = animation.FuncAnimation(fig, upd, frames=T1 - T0, interval=240)
    out = os.path.join(legacy.PLOTDIR, "demo_bc.gif")
    ani.save(out, writer="pillow", fps=4, dpi=105)
    plt.close(fig)
    print(f"saved {out} ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
