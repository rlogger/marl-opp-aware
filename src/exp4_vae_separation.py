"""Is the opponent's strategy recoverable from its trajectory? (Part 1 precondition)

The co-training proposal's Part 1 needs an encoder that maps an opponent's
trajectory tau^red to a latent z carrying its *strategy*. That is only possible
if the strategy is actually present in the trajectory. This script measures, for
the prey treated as the opponent, whether the resource-placement strategy
(circle-loop vs corner-dash) is recoverable, and separates two failure modes:

  (a) signal absent from the data   -> even a supervised probe on the raw
      trajectory cannot predict placement (the prey never executed the strategy,
      e.g. because it was dominated by the predators);
  (b) signal present but unrecovered -> a raw probe succeeds but the unsupervised
      VAE latent does not (an encoder/objective limitation, the Exp 2 story).

For each algorithm's trajectory dataset we report three numbers:
  - raw linear probe   : supervised placement accuracy on the flat trajectory
  - latent linear probe: supervised placement accuracy on the VAE latent z
  - unsupervised ARI   : GMM(k=2) on z vs the true placement label

Placement is recovered from resource geometry (circle r=0.6, corners r~1.13)
and is never shown to the encoder.

Input  : logs/MPE_simple_tag_v3/trajectory_dataset_resources_{alg}_A.npz
Output : plots/exp4_vae_separation.png
         logs/MPE_simple_tag_v3/exp4_vae_separation.npz
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from train_traj_vae import TrajVAE, elbo

LOGDIR = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"

LATENT_DIM = 8
HIDDEN = 64
LR = 1e-3
BATCH = 64
N_STEPS = 8_000
KL_ANNEAL_STEPS = 2_000
GRAD_CLIP = 1.0
SEED = 0

# algorithms to compare, with their dataset suffixes.
# The first three share the figure's scatter panels; SCATTER controls that.
ALGOS = [("iql", "IQL"), ("oa_iql", "OA-IQL"), ("mappo", "MAPPO"),
         ("iql_hr", "IQL ×4 reward")]
SCATTER = {"iql", "oa_iql", "mappo"}
EXP2_ARI = 0.008  # prey VAE on Exp 2 shaping data (verify_vae_modes), for context


# Episodes are 25 steps; trajectories are T=50, so auto-reset re-draws the
# placement at the t=25 boundary. We slice the FIRST episode only (t=0..25),
# whose placement is the t=0 geometry, so the label is valid for the whole slice.
EP_LEN = 25


def placement_labels(d):
    """Recover circle (0) vs corners (1) for the first episode from t=0 geometry."""
    rp = d["resource_pos"][:, 0]                    # (N, 4, 2) at t=0
    r = np.linalg.norm(rp, axis=-1).mean(axis=-1)
    return (r > 0.9).astype(np.int32)


def load_prey(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    pos = d["positions"].astype(np.float32)[:, :EP_LEN + 1]   # first episode only
    rel = (pos - pos[:, :1])[:, 1:]                 # init-condition, drop t=0
    N, T, _ = rel.shape
    prey_ret = float(d["rewards"][:, :EP_LEN].sum(1).mean())
    return rel.reshape(N, T * 2), placement_labels(d), prey_ret


def train_vae(x_std, T):
    D = x_std.shape[1]
    rng = jax.random.PRNGKey(SEED)
    model = TrajVAE(hidden=HIDDEN, latent=LATENT_DIM, out_dim=D)
    rng, ik, ck = jax.random.split(rng, 3)
    params = model.init(ik, jnp.asarray(x_std[:1]), ck)
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def step(state, batch, rng, beta):
        def loss_fn(p):
            x_hat, mu, lv, _ = state.apply_fn(p, batch, rng)
            loss, aux = elbo(x_hat, batch, mu, lv, beta)
            return loss, aux
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss

    xt = jnp.asarray(x_std)
    n = xt.shape[0]
    for s in range(N_STEPS):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, shape=(BATCH,), replace=False)
        beta = float(min(1.0, s / max(KL_ANNEAL_STEPS, 1)))
        state, _ = step(state, xt[idx], sk, beta)

    mu, _ = model.apply(state.params, xt, method=model.encode)
    return np.asarray(mu)


def probe_acc(X, y):
    return float(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean())


def analyse(suffix):
    path = os.path.join(LOGDIR, f"trajectory_dataset_resources_{suffix}_A.npz")
    if not os.path.exists(path):
        return None
    x_raw, labels, prey_ret = load_prey(path)
    mean = x_raw.mean(0); std = x_raw.std(0) + 1e-6
    x_std = (x_raw - mean) / std

    raw_probe = probe_acc(x_std, labels)            # is strategy IN the data?
    z = train_vae(x_std, x_std.shape[1] // 2)
    lat_probe = probe_acc(z, labels)                # did the VAE keep it?
    gm = GaussianMixture(2, covariance_type="full", n_init=10,
                         random_state=SEED).fit(z)
    pred = gm.predict(z)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    maj = max((labels == 0).mean(), (labels == 1).mean())
    print(f"[{suffix}] prey_ret={prey_ret:+.1f}  majority={maj:.2f}  "
          f"raw_probe={raw_probe:.2f}  lat_probe={lat_probe:.2f}  "
          f"ARI={ari:.3f}  NMI={nmi:.3f}")
    return dict(suffix=suffix, prey_ret=prey_ret, majority=float(maj),
                raw_probe=raw_probe, lat_probe=lat_probe, ari=ari, nmi=nmi,
                z=z, labels=labels, pred=pred)


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    results = {}
    for suffix, _ in ALGOS:
        r = analyse(suffix)
        if r is not None:
            results[suffix] = r

    np.savez(os.path.join(LOGDIR, "exp4_vae_separation.npz"),
             **{f"{k}_{m}": np.asarray(v[m])
                for k, v in results.items()
                for m in ("raw_probe", "lat_probe", "ari", "nmi",
                          "prey_ret", "majority")})

    present = [(s, n) for s, n in ALGOS if s in results]
    scatters = [(s, n) for s, n in present if s in SCATTER]
    ncol = len(scatters) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4))

    # per-algorithm latent scatter, coloured by true placement
    for ax, (suffix, name) in zip(axes[:-1], scatters):
        r = results[suffix]
        z2 = PCA(2).fit_transform(r["z"])
        for lab, lname, col in [(0, "circle", "#0d6e7a"), (1, "corners", "#b85c10")]:
            m = r["labels"] == lab
            ax.scatter(z2[m, 0], z2[m, 1], s=12, alpha=0.6, c=col, label=lname)
        ax.set_title(f"{name}  (prey ret {r['prey_ret']:+.0f})\n"
                     f"raw probe {r['raw_probe']:.2f} → latent {r['lat_probe']:.2f}, "
                     f"ARI {r['ari']:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(); ax.grid(alpha=0.3)

    # summary bars: raw vs latent probe per algorithm + chance (all conditions)
    ax = axes[-1]
    names = [n for _, n in present]
    xs = np.arange(len(names))
    w = 0.38
    raw = [results[s]["raw_probe"] for s, _ in present]
    lat = [results[s]["lat_probe"] for s, _ in present]
    chance = [results[s]["majority"] for s, _ in present]
    ax.bar(xs - w/2, raw, w, label="raw-traj probe", color="#0d6e7a")
    ax.bar(xs + w/2, lat, w, label="VAE-latent probe", color="#b85c10")
    ax.plot(xs, chance, "k--", marker="o", label="majority class")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("supervised placement accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Is strategy in the data, and in z?")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Exp 4 — recovering prey strategy from its trajectory "
                 "(Part 1 precondition)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, "exp4_vae_separation.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
