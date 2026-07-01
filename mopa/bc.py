"""Latent-conditioned behaviour cloning for the predator team.

The meeting-notes protocol: naive BC predicts a predator's action from all
agent locations, pi(a | s); the claim under test is that conditioning on the
opponent-strategy latent z from the unsupervised encoder, pi(a | s, z),
improves held-out action prediction.

Protocol guarantees (the audit's standard):
  * EPISODE-level train/val split -- no pooled-timestep leakage; all steps of
    an episode are in exactly one fold.
  * z is computed from the prey's first CTX steps only, and BC samples start
    at t = CTX, so the conditioning never peeks at the future relative to the
    predicted step.
  * every number is mean +/- std over training seeds.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
import optax

from mopa.samples import build_predator_samples
from mopa.splits import episode_validation_mask

BC_HID = 128
BC_STEPS = 4000
BC_BATCH = 256


class BCNet(nn.Module):
    n_actions: int = 5

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(BC_HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(BC_HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))(x)


def build_samples(ds, ctx, ep_len=25, t_max=50):
    """(state, action, episode-id) samples for every predator at every valid
    step. State = absolute positions of all 4 agents + velocity proxy + the
    predator's id one-hot. Valid steps: t in [ctx, t_max) excluding the
    auto-reset boundary (velocity proxy invalid across the reset).

    Returns S (M, 19), A (M,), ep (M,)."""
    return build_predator_samples(ds, ctx, t_max, ep_len=ep_len)


def train_eval_bc(S, A, ep, rng_seed, val_frac=0.2, steps=BC_STEPS):
    """Episode-level split, train a BC net, return held-out accuracy."""
    vmask = episode_validation_mask(ep, rng_seed=rng_seed, val_frac=val_frac)
    Str, Atr, Sva, Ava = S[~vmask], A[~vmask], S[vmask], A[vmask]

    mu, sd = Str.mean(0), Str.std(0) + 1e-6
    Str, Sva = (Str - mu) / sd, (Sva - mu) / sd

    net = BCNet()
    key = jax.random.PRNGKey(rng_seed)
    key, ki = jax.random.split(key)
    params = net.init(ki, Str[:1])
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Sj, Aj = jnp.asarray(Str), jnp.asarray(Atr)
    n = len(Str)

    def loss_fn(p, s, a):
        logits = net.apply(p, s)
        return optax.softmax_cross_entropy_with_integer_labels(logits, a).mean()

    @jax.jit
    def upd(params, opt, idx):
        g = jax.grad(loss_fn)(params, Sj[idx], Aj[idx])
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for _ in range(steps):
        key, bk = jax.random.split(key)
        idx = jax.random.choice(bk, n, (min(BC_BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx)

    pred = np.asarray(net.apply(params, jnp.asarray(Sva))).argmax(-1)
    return float((pred == Ava).mean())


def bc_comparison(ds, z_dict, ctx, seeds=(0, 1, 2)):
    """Run BC for each conditioning variant; return {name: (mean, std, runs)}.

    z_dict maps variant name -> per-episode conditioning array (N, d) or None
    for the unconditioned baseline."""
    S0, A, ep = build_samples(ds, ctx)
    results = {}
    for name, z in z_dict.items():
        if z is None:
            S = S0
        else:
            zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
            S = np.concatenate([S0, zs[ep]], -1).astype(np.float32)
        runs = [train_eval_bc(S, A, ep, s) for s in seeds]
        v = np.asarray(runs)
        results[name] = (float(v.mean()), float(v.std()), v)
        print(f"  BC pi(a|s{',' + name if z is not None else ''}) "
              f": {v.mean():.4f} +/- {v.std():.4f}")
    return results
