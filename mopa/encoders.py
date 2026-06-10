"""Self-supervised opponent-trajectory encoders: VAE (reconstruct) vs JEPA
(predict the future's representation).

Same architectures/hyperparameters as the validated hidden-intent study
(src/jepa_vs_vae_encoder.py), generalised to any window sizes and label
cardinality, with the two protocol upgrades the audit called for:

  * multi-seed: every reported number is mean +/- std over ENCODER training
    seeds (default 3), not a single PRNGKey(0) point estimate;
  * the probe is cross-validated at the trajectory level (one sample per
    episode), so there is no pooled-timestep leakage.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
import optax

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score

HID = 64
STEPS = 5000
BATCH = 128
EMA = 0.996


class Enc(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(x)


class EncVAE(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.lat)(x), nn.Dense(self.lat)(x)   # mu, logvar


class Dec(nn.Module):
    out: int

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(self.out)(z)


class Pred(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(z)


def probe_acc(z, y, cv=5):
    return float(cross_val_score(LogisticRegression(max_iter=1000), z, y,
                                 cv=cv).mean())


def metrics(z, y, n_classes):
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    probe = probe_acc(zc, y)
    gm = GaussianMixture(n_classes, n_init=8, random_state=0).fit(zc)
    return probe, float(adjusted_rand_score(y, gm.predict(zc)))


def train_vae(Xc, rng, lat=2, steps=STEPS):
    """ELBO on the context window. Returns z (N, lat)."""
    enc, dec = EncVAE(lat=lat), Dec(out=Xc.shape[1])
    rng, ke, kd = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "d": dec.init(kd, jnp.zeros((1, lat)))}
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xj = jnp.asarray(Xc)
    n = len(Xc)

    def loss_fn(p, x, rng, beta):
        mu, lv = enc.apply(p["e"], x)
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(rng, mu.shape)
        rec = jnp.mean(jnp.sum((dec.apply(p["d"], z) - x) ** 2, -1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), -1))
        return rec + beta * kl

    @jax.jit
    def upd(params, opt, idx, rng, beta):
        g = jax.grad(loss_fn)(params, Xj[idx], rng, beta)
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for s in range(steps):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx, sk, float(min(1.0, s / 1500)))
    return np.asarray(enc.apply(params["e"], Xj)[0])


def train_jepa(Xc, Xt, rng, lat=2, steps=STEPS):
    """Predict the EMA-target representation of the target window from the
    context window; VICReg variance term against collapse. Returns z (N, lat)."""
    enc, pred = Enc(lat=lat), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "p": pred.init(kp, jnp.zeros((1, lat)))}
    target = params["e"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xt)
    n = len(Xc)

    def loss_fn(p, tgt, xc, xt):
        z = enc.apply(p["e"], xc)
        pz = pred.apply(p["p"], z)
        t = jax.lax.stop_gradient(enc.apply(tgt, xt))
        std = jnp.sqrt(z.var(0) + 1e-4)
        return jnp.mean(jnp.abs(pz - t)) + jnp.mean(jax.nn.relu(1.0 - std))

    @jax.jit
    def upd(params, target, opt, idx):
        g = jax.grad(loss_fn)(params, target, Xcj[idx], Xtj[idx])
        u, opt = tx.update(g, opt)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(lambda a, b: EMA * a + (1 - EMA) * b,
                                        target, params["e"])
        return params, target, opt

    for s in range(steps):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx)
    return np.asarray(enc.apply(params["e"], Xcj))


def evaluate_encoders(Xc, Xt, y, n_classes, seeds=(0, 1, 2), lat=2,
                      steps=STEPS, keep_z_seed=0):
    """Train VAE and JEPA over several encoder seeds; return mean/std metrics
    and one representative latent per encoder (from keep_z_seed) for scatter
    plots. Xc = standardized context windows, Xt = standardized target windows."""
    out = {"vae": {"probe": [], "ari": []}, "jepa": {"probe": [], "ari": []}}
    z_keep = {}
    for s in seeds:
        zv = train_vae(Xc, jax.random.PRNGKey(s), lat=lat, steps=steps)
        zj = train_jepa(Xc, Xt, jax.random.PRNGKey(s), lat=lat, steps=steps)
        for name, z in (("vae", zv), ("jepa", zj)):
            p, a = metrics(z, y, n_classes)
            out[name]["probe"].append(p)
            out[name]["ari"].append(a)
            if s == keep_z_seed:
                z_keep[name] = z
    summary = {}
    for name in out:
        for m in ("probe", "ari"):
            v = np.asarray(out[name][m])
            summary[f"{name}_{m}_mean"] = float(v.mean())
            summary[f"{name}_{m}_std"] = float(v.std())
            summary[f"{name}_{m}_all"] = v
    return summary, z_keep
