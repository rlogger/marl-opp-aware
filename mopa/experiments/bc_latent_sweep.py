"""6/12 subsequent step: latent-conditioned BC vs vanilla BC -- and WHEN it wins.

The earlier one-episode result showed no gain because a latent from 25 steps of
observation barely carries the placement (~0.51 probe). The placement is fixed
for a specialist across all its episodes, so we can give the encoder MORE
observation and ask how the BC gain tracks the latent's quality.

Protocol: roll each specialist for 6 episodes (150 steps). For an observation
length L, compute the unsupervised latent z from the prey's occupancy over the
first L steps, then predict the predator's actions in the LAST episode
(steps 125-150) from pi(a|s) vs pi(a|s, z). z is from steps strictly before the
predicted episode, so there is no leakage. Episode-level split, 3 BC seeds.
Variants: none (vanilla), z_VAE, z_JEPA, oracle (true placement one-hot).

Claim under test: latent-conditioned BC overtakes vanilla BC once the latent
actually carries the strategy -- i.e. the BC gain rises with the latent's probe.

Outputs: plots/mopa_bc_latent_sweep.png
         logs/MPE_simple_tag_v3/mopa_bc_latent_sweep.npz
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax

from mopa import legacy
from mopa.data import specialist_dataset, standardize, occupancy
from mopa.encoders import train_vae, train_jepa, probe_acc
from mopa.bc import train_eval_bc

EP_LEN = 25
PREDICT = (125, 150)        # the held-out episode we clone in
LSWEEP = (25, 50, 100, 125)
LAT = 4


def build_range(ds, t0, t1):
    """(state, action, rollout-id) for every predator at steps [t0, t1),
    skipping reset boundaries (velocity invalid there)."""
    prey, preds, acts = ds["prey_pos"], ds["pred_pos"], ds["pred_act"]
    N = len(prey)
    steps = [t for t in range(t0, t1) if t % EP_LEN != 0]
    S, A, EP = [], [], []
    for t in steps:
        pos = np.concatenate([preds[:, t].reshape(N, 6), prey[:, t]], -1)
        vel = pos - np.concatenate([preds[:, t - 1].reshape(N, 6),
                                    prey[:, t - 1]], -1)
        for p in range(3):
            pid = np.zeros((N, 3), np.float32); pid[:, p] = 1.0
            S.append(np.concatenate([pos, vel, pid], -1))
            A.append(acts[:, t, p]); EP.append(np.arange(N))
    return (np.concatenate(S).astype(np.float32),
            np.concatenate(A).astype(np.int32),
            np.concatenate(EP).astype(np.int32))


def main():
    os.makedirs(legacy.PLOTDIR, exist_ok=True)
    print("rolling out specialists (6 episodes each)...")
    ds = specialist_dataset(n_eps=150, num_steps=150)
    y = ds["label"]
    S0, A, ep = build_range(ds, *PREDICT)
    oracle = np.eye(2, dtype=np.float32)[y]
    print(f"  rollouts {len(y)}, predict-window samples {len(S0)}")

    base = {}
    for s in (0, 1, 2):
        base.setdefault("v", []).append(train_eval_bc(S0, A, ep, s))
    vanilla = (float(np.mean(base["v"])), float(np.std(base["v"])))
    orc = np.concatenate([S0, oracle[ep]], -1).astype(np.float32)
    oracle_runs = [train_eval_bc(orc, A, ep, s) for s in (0, 1, 2)]
    oracle_acc = (float(np.mean(oracle_runs)), float(np.std(oracle_runs)))
    print(f"  vanilla pi(a|s): {vanilla[0]:.4f}   "
          f"oracle pi(a|s,placement): {oracle_acc[0]:.4f}")

    rows = {"L": [], "probe_vae": [], "probe_jepa": [],
            "vae": [], "vae_std": [], "jepa": [], "jepa_std": []}
    for L in LSWEEP:
        Xc, _, _ = standardize(occupancy(ds["prey_pos"], 0, L))
        # JEPA target capped at the predict-window start: never sees steps 125+
        Xt, _, _ = standardize(occupancy(ds["prey_pos"], 0, min(2 * L, PREDICT[0])))
        zv = train_vae(Xc, jax.random.PRNGKey(0), lat=LAT)
        zj = train_jepa(Xc, Xt, jax.random.PRNGKey(0), lat=LAT)
        rows["L"].append(L)
        rows["probe_vae"].append(probe_acc((zv - zv.mean(0)) / (zv.std(0) + 1e-6), y))
        rows["probe_jepa"].append(probe_acc((zj - zj.mean(0)) / (zj.std(0) + 1e-6), y))
        for nm, z in (("vae", zv), ("jepa", zj)):
            zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
            S = np.concatenate([S0, zs[ep]], -1).astype(np.float32)
            r = [train_eval_bc(S, A, ep, s) for s in (0, 1, 2)]
            rows[nm].append(float(np.mean(r))); rows[f"{nm}_std"].append(float(np.std(r)))
        print(f"  L={L:3d}: probe(jepa) {rows['probe_jepa'][-1]:.2f}  "
              f"BC vae {rows['vae'][-1]:.4f}  jepa {rows['jepa'][-1]:.4f}  "
              f"(vanilla {vanilla[0]:.4f})")

    np.savez(os.path.join(legacy.LOGDIR, "mopa_bc_latent_sweep.npz"),
             vanilla=np.array(vanilla), oracle=np.array(oracle_acc),
             **{k: np.array(v) for k, v in rows.items()})

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    L = rows["L"]
    ax.axhline(vanilla[0], ls="--", c="#444", lw=1.4, label=f"vanilla π(a|s) {vanilla[0]:.3f}")
    ax.axhline(oracle_acc[0], ls=":", c="#073b42", lw=1.4, label=f"oracle (placement) {oracle_acc[0]:.3f}")
    ax.errorbar(L, rows["vae"], yerr=rows["vae_std"], marker="o", c="#999", label="π(a|s, z_VAE)", capsize=3)
    ax.errorbar(L, rows["jepa"], yerr=rows["jepa_std"], marker="o", c="#0d6e7a", label="π(a|s, z_JEPA)", capsize=3)
    ax.set_xlabel("encoder observation length L (steps seen before the predicted episode)")
    ax.set_ylabel("held-out action accuracy")
    ax.set_title("Latent-conditioned BC overtakes vanilla BC once the latent carries the strategy")
    for i, l in enumerate(L):
        ax.annotate(f"probe {rows['probe_jepa'][i]:.2f}", (l, rows["jepa"][i]),
                    textcoords="offset points", xytext=(0, 8), fontsize=7.5, ha="center")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(legacy.PLOTDIR, "mopa_bc_latent_sweep.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
