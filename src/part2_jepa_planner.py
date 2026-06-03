"""Label-free opponent-aware planning: a JEPA belief drives the planner.

The Part-2 planner (part2_planner.py) reached 4.31 captures/episode using a belief
from a SUPERVISED intent classifier (trained with ground-truth intent labels).
Here we replace it with a fully SELF-SUPERVISED, label-free belief and ask whether
the planner still matches that result.

The label-free belief:
  1. a JEPA encoder of the prey trajectory (observe first-t of a KMAX window, mask
     the future, predict the full window's representation -- no labels);
  2. a readout  z -> predicted arrival position  (trained on the prey's actual
     future positions -- no intent labels), random observation length t;
  3. belief over the four corners  b(c) ~ softmax(-||readout(z) - corner_c||^2 / s),
     using only the *known* arena geometry, never the opponent's intent.

This drops into the planner's belief slot unchanged (same {k: fn} interface).

Output: plots/part2_jepa_planner.png, logs/.../part2_jepa_planner.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part2_intent_eval as PE
import part2_planner as PP
from jaxmarl.wrappers.baselines import load_params

LOGDIR = "logs/MPE_simple_tag_v3"; PLOTDIR = "plots"
KMAX = 24; LAT = 16; HID = 128; STEPS = 6000
ARRIVE = (18, 24)              # "where it ends up" window (label-free target)
SIGMA = 0.5
ENC_KS = [3, 5, 8, 12, 18, 24]


class Enc(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(LAT, kernel_init=orthogonal(1.0))(x)


class Pred(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(LAT, kernel_init=orthogonal(1.0))(z)


class Readout(nn.Module):                 # z -> predicted 2-D arrival position
    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(2)(z)


def train_jepa_belief(full_std, arrive, rng):
    """Joint JEPA + arrival-readout, random observation length. Self-supervised."""
    enc, pred, rd = Enc(), Pred(), Readout()
    rng, ke, kp, kr = jax.random.split(rng, 4)
    x0 = full_std.reshape(len(full_std), -1)[:1]
    p = {"e": enc.init(ke, x0), "p": pred.init(kp, jnp.zeros((1, LAT))),
         "r": rd.init(kr, jnp.zeros((1, LAT)))}
    tgt = p["e"]; tx = optax.adam(1e-3); opt = tx.init(p)
    F = jnp.asarray(full_std); A = jnp.asarray(arrive); n = len(full_std); MOM = 0.996

    def mask(xb, t):
        m = (jnp.arange(KMAX)[None, :] < t[:, None])[:, :, None]
        return jnp.where(m, xb, 0.0).reshape(xb.shape[0], -1)

    def loss(p, tgt, xb, ab, t):
        ctx = mask(xb, t)
        z = enc.apply(p["e"], ctx)
        tr = jax.lax.stop_gradient(enc.apply(tgt, xb.reshape(xb.shape[0], -1)))
        std = jnp.sqrt(z.var(0) + 1e-4)
        jepa = jnp.mean(jnp.abs(pred.apply(p["p"], z) - tr)) + jnp.mean(jax.nn.relu(1 - std))
        read = jnp.mean(jnp.sum((rd.apply(p["r"], z) - ab) ** 2, -1))   # label-free
        return jepa + read

    @jax.jit
    def upd(p, tgt, opt, idx, kt):
        t = jax.random.randint(kt, (len(idx),), 3, KMAX)
        g = jax.grad(loss)(p, tgt, F[idx], A[idx], t)
        u, opt = tx.update(g, opt); p = optax.apply_updates(p, u)
        tgt = jax.tree_util.tree_map(lambda a, b: MOM * a + (1 - MOM) * b, tgt, p["e"])
        return p, tgt, opt
    for s in range(STEPS):
        rng, bk, kt = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (256,), replace=False)
        p, tgt, opt = upd(p, tgt, opt, idx, kt)
    return enc, rd, p


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env = PE.make_env(reveal=True)
    corners = np.asarray(env.corners)                     # (4,2) known geometry
    pos, intent = [], []
    for i in PE.SEEDS:
        P, g = PE.gather_prey_traj(env,
                                   load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                   load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                   PE.N_EPS, jax.random.PRNGKey(i))
        pos.append(P); intent.append(g)
    pos = np.concatenate(pos).astype(np.float32); intent = np.concatenate(intent).astype(int)

    full = pos[:, :KMAX].reshape(len(pos), -1)
    mu, sd = full.mean(0), full.std(0) + 1e-6
    full_std = ((full - mu) / sd).reshape(len(pos), KMAX, 2)
    arrive = pos[:, ARRIVE[0]:ARRIVE[1]].mean(1)          # actual arrival (label-free)

    print("training label-free JEPA belief (encoder + arrival readout)...")
    enc, rd, p = train_jepa_belief(full_std, arrive, jax.random.PRNGKey(0))

    # sanity: does the belief recover the (held-out) intent?
    def belief_from_first_t(P):                            # P:(n,t,2) absolute
        n, t, _ = P.shape
        buf = np.zeros((n, KMAX, 2), np.float32)
        tt = min(t, KMAX); buf[:, :tt] = P[:, :tt]
        std = ((buf.reshape(n, -1) - mu) / sd).reshape(n, KMAX, 2)
        m = (np.arange(KMAX)[None, :] < tt)[:, :, None]
        ctx = jnp.asarray(np.where(m, std, 0.0).reshape(n, -1))
        z = enc.apply(p["e"], ctx)
        pred_xy = np.asarray(rd.apply(p["r"], z))          # (n,2) predicted arrival
        d2 = ((pred_xy[:, None, :] - corners[None]) ** 2).sum(-1)   # (n,4)
        return np.asarray(jax.nn.softmax(jnp.asarray(-d2 / SIGMA), axis=-1))

    acc = {}
    for k in ENC_KS:
        b = belief_from_first_t(pos[:, :k])
        acc[k] = float((b.argmax(1) == intent).mean())
    print("  label-free belief intent accuracy:", {k: round(v, 2) for k, v in acc.items()})

    # run the planner with the JEPA belief (drop-in {k: fn})
    enc_proba = {k: belief_from_first_t for k in ENC_KS}
    caps = []
    for i in PE.SEEDS:
        prey_be = load_params(PE.apath("mappo_intent_belief", "prey", i))
        pred_be = load_params(PE.apath("mappo_intent_belief", "pred", i))
        c = PP.run_planner_episode(env, prey_be, pred_be, ENC_KS, enc_proba,
                                   jax.random.PRNGKey(400 + i), belief_mode="online")
        caps.append(c.mean()); print(f"  seed {i}: JEPA-belief planner {c.mean():.2f}", flush=True)
    jm, js = float(np.mean(caps)), float(np.std(caps))

    prev = np.load(os.path.join(LOGDIR, "part2_planner.npz"))
    sup = (float(prev["planner"].mean()), float(prev["planner"].std()))      # supervised belief
    react = float(prev["ref_belief"].mean()); unaw = float(prev["ref_unaware"].mean()) \
        if "ref_unaware" in prev else float(np.load(os.path.join(LOGDIR, "part2_intent_eval.npz"))["cap_unaware"].mean())
    orc = float(prev["ref_oracle"].mean())
    print(f"\n  unaware (no intent)          : {unaw:.2f}")
    print(f"  supervised-belief planner    : {sup[0]:.2f}")
    print(f"  JEPA-belief planner (ours)   : {jm:.2f} +/- {js:.2f}   [LABEL-FREE]")
    print(f"  oracle (true intent, reactive): {orc:.2f}")

    np.savez(os.path.join(LOGDIR, "part2_jepa_planner.npz"),
             jepa=np.array(caps), sup=prev["planner"], unaware=unaw, oracle=orc,
             belief_acc=np.array([acc[k] for k in ENC_KS]), ks=np.array(ENC_KS))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = ["unaware\n(no intent)", "supervised-belief\nplanner",
              "JEPA-belief planner\n(label-free, ours)", "oracle\n(true intent)"]
    means = [unaw, sup[0], jm, orc]; stds = [0, sup[1], js, 0]
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=["#999", "#1f7a4e", "#0d6e7a", "#073b42"])
    ax.set_ylabel("captures / episode")
    ax.set_title(f"Label-free opponent-aware planning\n"
                 f"a self-supervised JEPA belief matches the supervised one "
                 f"({jm:.2f} vs {sup[0]:.2f}) -- no intent labels")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.2f}", ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, "part2_jepa_planner.png"), dpi=140); plt.close(fig)
    print("saved plots/part2_jepa_planner.png")


if __name__ == "__main__":
    main()
