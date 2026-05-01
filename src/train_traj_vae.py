"""Train an MLP-VAE on init-conditioned trajectories (Dataset A).

Step 2 of the proposal sketch: learn a latent representation of an agent's
trajectory distribution. Run once per agent (--agent prey | pred_0 | pred_1
| pred_2). The dataset stores the prey trajectory as `positions` and the
three predator trajectories as `pred_positions[:, :, i, :]`.

Inputs
    logs/MPE_simple_tag_v3/trajectory_dataset_A.npz

Outputs (suffix = agent name; default agent is "prey")
    logs/MPE_simple_tag_v3/traj_vae_<agent>.safetensors
    logs/MPE_simple_tag_v3/traj_vae_<agent>_stats.npz
    logs/MPE_simple_tag_v3/traj_vae_<agent>_metrics.npz
    plots/traj_vae_<agent>_loss.png

Architecture
    Encoder MLP  : (T*2)  -> hidden -> hidden -> (mu, logvar) of dim LATENT
    Decoder MLP  : LATENT -> hidden -> hidden -> (T*2)
    Loss         : sum-MSE recon + beta * KL(q(z|x) || N(0, I))
                   beta linearly annealed from 0 -> 1 over KL_ANNEAL_STEPS.

Training is on init-conditioned positions (subtract t=0 from every step)
with the trivial first step dropped, so the input is dim T*2.
"""
import argparse
import os
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
import optax

from jaxmarl.wrappers.baselines import save_params


LOGDIR  = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"

LATENT_DIM       = 8
HIDDEN           = 64
LR               = 1e-3
BATCH            = 64
N_STEPS          = 8_000
KL_ANNEAL_STEPS  = 2_000
GRAD_CLIP        = 1.0
SEED             = 0


# --------------------------- model ------------------------------------------ #

class TrajVAE(nn.Module):
    hidden: int
    latent: int
    out_dim: int

    def setup(self):
        ki, bi = orthogonal(1.0), constant(0.0)
        self.enc1 = nn.Dense(self.hidden, kernel_init=ki, bias_init=bi)
        self.enc2 = nn.Dense(self.hidden, kernel_init=ki, bias_init=bi)
        self.enc_mu     = nn.Dense(self.latent, kernel_init=orthogonal(0.01), bias_init=bi)
        self.enc_logvar = nn.Dense(self.latent, kernel_init=orthogonal(0.01), bias_init=bi)
        self.dec1 = nn.Dense(self.hidden, kernel_init=ki, bias_init=bi)
        self.dec2 = nn.Dense(self.hidden, kernel_init=ki, bias_init=bi)
        self.dec_out = nn.Dense(self.out_dim, kernel_init=ki, bias_init=bi)

    def encode(self, x):
        h = nn.relu(self.enc1(x))
        h = nn.relu(self.enc2(h))
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z):
        h = nn.relu(self.dec1(z))
        h = nn.relu(self.dec2(h))
        return self.dec_out(h)

    def __call__(self, x, rng):
        mu, logvar = self.encode(x)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mu.shape)
        z = mu + std * eps
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def elbo(x_hat, x, mu, logvar, beta):
    recon = jnp.mean(jnp.sum((x_hat - x) ** 2, axis=-1))
    kl    = jnp.mean(-0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1))
    return recon + beta * kl, (recon, kl)


# --------------------------- data ------------------------------------------- #

def _agent_positions(d: np.lib.npyio.NpzFile, agent: str) -> np.ndarray:
    """Pull the (N, T+1, 2) trajectory for the named agent out of a dataset."""
    if agent == "prey":
        return d["positions"].astype(np.float32)
    if agent.startswith("pred_"):
        i = int(agent.split("_", 1)[1])
        return d["pred_positions"][:, :, i, :].astype(np.float32)
    raise ValueError(f"unknown agent {agent!r} (use prey | pred_0 | pred_1 | pred_2)")


def load_and_prepare(npz_path: str, agent: str = "prey"):
    """Init-condition, drop t=0, flatten — for the chosen agent's trajectory."""
    d = np.load(npz_path, allow_pickle=True)
    positions = _agent_positions(d, agent)               # (N, T+1, 2)
    rel = positions - positions[:, :1]                   # (N, T+1, 2)
    rel = rel[:, 1:]                                     # drop trivial t=0
    N, T, _ = rel.shape
    x = rel.reshape(N, T * 2)
    return x, T


# --------------------------- training --------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="prey",
                   choices=["prey", "pred_0", "pred_1", "pred_2"])
    args = p.parse_args()
    agent = args.agent

    os.makedirs(PLOTDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    npz = os.path.join(LOGDIR, "trajectory_dataset_A.npz")
    x_raw, T = load_and_prepare(npz, agent=agent)
    N, D = x_raw.shape
    print(f"agent={agent}  loaded {npz}: N={N}, T={T}, flat_dim={D}")

    # Train / val split (90 / 10).
    rng = jax.random.PRNGKey(SEED)
    rng, perm_key = jax.random.split(rng)
    perm = np.array(jax.random.permutation(perm_key, N))
    n_val = N // 10
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Per-dim standardisation, fit on train only.
    train_mean = x_raw[train_idx].mean(axis=0)
    train_std  = x_raw[train_idx].std(axis=0) + 1e-6
    x_std = (x_raw - train_mean) / train_std
    x_train = jnp.asarray(x_std[train_idx])
    x_val   = jnp.asarray(x_std[val_idx])
    print(f"train shape {x_train.shape}  val shape {x_val.shape}")

    # Init.
    model = TrajVAE(hidden=HIDDEN, latent=LATENT_DIM, out_dim=D)
    rng, init_key, init_call_key = jax.random.split(rng, 3)
    params = model.init(init_key, x_train[:1], init_call_key)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"params: {n_params:,}")

    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch, rng, beta):
        def loss_fn(p):
            x_hat, mu, logvar, _ = state.apply_fn(p, batch, rng)
            loss, (recon, kl) = elbo(x_hat, batch, mu, logvar, beta)
            return loss, (recon, kl)
        (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, recon, kl

    @jax.jit
    def eval_step(state, x, rng):
        x_hat, mu, logvar, _ = state.apply_fn(state.params, x, rng)
        loss, (recon, kl) = elbo(x_hat, x, mu, logvar, 1.0)
        return loss, recon, kl

    history: dict[str, list[Any]] = {
        "step": [], "loss": [], "recon": [], "kl": [], "beta": [],
        "val_loss": [], "val_recon": [], "val_kl": [],
    }

    n_train = x_train.shape[0]
    print(f"\ntraining for {N_STEPS} steps  batch={BATCH}  KL anneal over {KL_ANNEAL_STEPS}")
    for step in range(N_STEPS):
        rng, b_key, v_key, e_key = jax.random.split(rng, 4)
        idx = jax.random.choice(b_key, n_train, shape=(BATCH,), replace=False)
        batch = x_train[idx]
        beta = float(min(1.0, step / max(KL_ANNEAL_STEPS, 1)))
        state, loss, recon, kl = train_step(state, batch, v_key, beta)

        if step % 250 == 0 or step == N_STEPS - 1:
            val_loss, val_recon, val_kl = eval_step(state, x_val, e_key)
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["recon"].append(float(recon))
            history["kl"].append(float(kl))
            history["beta"].append(beta)
            history["val_loss"].append(float(val_loss))
            history["val_recon"].append(float(val_recon))
            history["val_kl"].append(float(val_kl))
            print(f"  step {step:>5d}  beta={beta:.2f}  "
                  f"loss={float(loss):>7.3f}  recon={float(recon):>7.3f}  kl={float(kl):>6.3f}  "
                  f"|  val_loss={float(val_loss):>7.3f}  val_recon={float(val_recon):>7.3f}  val_kl={float(val_kl):>6.3f}")

    # Save params + normalisation stats.
    suffix = agent
    params_path = os.path.join(LOGDIR, f"traj_vae_{suffix}.safetensors")
    save_params(state.params, params_path)
    print(f"\nsaved params -> {params_path}")

    stats_path = os.path.join(LOGDIR, f"traj_vae_{suffix}_stats.npz")
    np.savez(
        stats_path,
        train_mean=train_mean, train_std=train_std,
        T=np.int32(T), latent_dim=np.int32(LATENT_DIM), hidden=np.int32(HIDDEN),
    )
    print(f"saved stats  -> {stats_path}")

    metrics_path = os.path.join(LOGDIR, f"traj_vae_{suffix}_metrics.npz")
    np.savez(metrics_path, **{k: np.array(v) for k, v in history.items()})
    print(f"saved metrics-> {metrics_path}")

    # Loss plot.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(history["step"], history["loss"], label="train (annealed β)")
    axes[0].plot(history["step"], history["val_loss"], label="val (β=1)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("ELBO loss")
    axes[0].set_title("Total loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["step"], history["recon"], label="train")
    axes[1].plot(history["step"], history["val_recon"], label="val")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("MSE (sum over dim)")
    axes[1].set_title("Reconstruction"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["step"], history["kl"], label="train")
    axes[2].plot(history["step"], history["val_kl"], label="val")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("nats / sample")
    axes[2].set_title("KL(q(z|x) || N(0, I))"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle(f"Trajectory VAE — agent: {agent}")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, f"traj_vae_{suffix}_loss.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved loss plot -> {out}")


if __name__ == "__main__":
    main()
