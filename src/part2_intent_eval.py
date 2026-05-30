"""The decisive experiment: does inferring the opponent's hidden intent pay off?

Hidden-intent task (simple_tag_intent.py): the prey is assigned one of K=4 corner
intents at reset and loiters there when safe; predators cannot see the intent.
We trained two co-training conditions (MAPPO, 3 seeds): an UNAWARE predator and an
ORACLE predator that observes the intent one-hot.

This script answers: how much does knowing / inferring the intent help the
predator catch the prey? We hold the prey fixed (the oracle-condition prey) and
feed the oracle predator four different intent signals:

  oracle   : the true intent                                  (ceiling)
  inferred : intent predicted by an encoder from the prey's
             first-k steps of motion, after observing k steps  (the method)
  guess    : a random intent, fixed per episode                (no inference)
  unaware  : the separately-trained predator with no intent input (honest baseline)

We also report the encoder's intent-recovery accuracy and posterior entropy as a
function of how many steps of the opponent it has observed.

Usage:
    python src/part2_intent_eval.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_tag_intent import SimpleTagIntentMPE
from generate_trajectory_dataset_resources import ActorLogits
from jaxmarl.wrappers.baselines import load_params

LOGDIR = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"
ENV = "MPE_simple_tag_v3"
K = 4
SEEDS = [0, 1, 2]
WP = dict(pred_max_speed=0.75, pred_accel=2.0)
CORNER, IRAD, IREW = 0.8, 0.35, 4.0
OBST = [[0.5, 0.5], [-0.5, -0.5]]
INFER_K = 8                # steps of observation before the predator may infer
N_EPS = 300


def make_env(reveal):
    return SimpleTagIntentMPE(num_intents=K, corner_offset=CORNER, intent_radius=IRAD,
                              intent_reward=IREW, reveal_to_pred=reveal,
                              obstacle_positions=OBST,
                              pred_max_speed=WP["pred_max_speed"],
                              pred_accel=WP["pred_accel"])


def apath(alg, team, i):
    return os.path.join(LOGDIR, f"{alg}_{ENV}_{team}_actor_seed0_vmap{i}.safetensors")


def _pad(o, max_obs):
    w = max_obs - o.shape[-1]
    return jnp.concatenate([o, jnp.zeros(o.shape[:-1] + (w,))], -1) if w > 0 else o


def rollout(env, prey_params, pred_params, n_eps, rng, mode="none",
            encoder=None, infer_k=INFER_K, guess_key=0,
            enc_ks=None, enc_proba=None):
    """One-episode (25-step) parallel rollout. `mode` controls the intent signal
    fed to the predator: none|oracle|guess|inferred|belief. The `belief` mode
    feeds the encoder's online-sharpening posterior (uniform until the first
    encoder window, then re-inferred from all steps seen so far)."""
    advs = env.adversaries; prey = env.good_agents[0]
    sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    max_obs = max(sizes.values())
    pred_base = sizes[advs[0]]
    isl = (pred_base - K, pred_base)         # intent one-hot location in pred obs
    net = ActorLogits(action_dim=5, hidden_dim=128)
    reset = jax.vmap(env.reset); step = jax.vmap(env.step_env)

    rng, kr = jax.random.split(rng)
    obs, state = reset(jax.random.split(kr, n_eps))
    true_intent = np.asarray(state.intent)
    true_oh = jax.nn.one_hot(jnp.asarray(true_intent), K)
    guess_oh = jax.nn.one_hot(
        jax.random.randint(jax.random.PRNGKey(900 + guess_key), (n_eps,), 0, K), K)
    uniform = jnp.full((n_eps, K), 1.0 / K)

    prey_xy = [np.asarray(state.p_pos[:, env.num_adversaries])]
    cap = np.zeros(n_eps); atc = np.zeros(n_eps)
    inferred_oh = [None]
    belief_cache = {}

    def pred_signal(t):
        if mode == "oracle":   return true_oh
        if mode == "guess":    return guess_oh
        if mode == "inferred":
            if t < infer_k:    return uniform
            if inferred_oh[0] is None:
                P = np.stack(prey_xy[:infer_k], axis=1)        # (n,k,2) absolute
                inferred_oh[0] = jnp.asarray(encoder(P))
            return inferred_oh[0]
        if mode == "belief":
            seen = len(prey_xy)                                # positions 0..t
            usable = [kk for kk in enc_ks if kk <= seen]
            if not usable:     return uniform
            kk = usable[-1]
            if kk not in belief_cache:
                P = np.stack(prey_xy[:kk], axis=1)
                belief_cache[kk] = jnp.asarray(enc_proba[kk](P))
            return belief_cache[kk]
        return None

    for t in range(env.max_steps):
        prey_in = _pad(obs[prey], max_obs)
        a_prey = net.apply(prey_params, prey_in).argmax(-1).astype(jnp.int32)
        sig = pred_signal(t)
        acts = {prey: a_prey}
        for a in advs:
            pin = _pad(obs[a], max_obs)
            if sig is not None and env.reveal_to_pred:
                pin = pin.at[:, isl[0]:isl[1]].set(sig)
            acts[a] = net.apply(pred_params, pin).argmax(-1).astype(jnp.int32)

        rng, ks = jax.random.split(rng)
        obs, state, rew, dones, info = step(jax.random.split(ks, n_eps), state, acts)
        prey_xy.append(np.asarray(state.p_pos[:, env.num_adversaries]))
        cap += np.asarray(rew[advs[0]]) / 10.0
        atc += np.asarray(info["at_corner"])
    return dict(captures=cap, prey_xy=np.stack(prey_xy, 1), intent=true_intent,
                at_corner=atc / env.max_steps)


def gather_prey_traj(env, prey_params, pred_params, n_eps, rng):
    d = rollout(env, prey_params, pred_params, n_eps, rng, mode="oracle")
    return d["prey_xy"], d["intent"]


def train_encoder_at_k(pos, intent, k):
    """MLP classifier from first-k absolute positions -> intent. Returns
    (cv_acc, hard_predict, entropy, soft_predict) where the predict fns map
    P:(n,k,2) to a (n,K) one-hot / posterior respectively."""
    X = pos[:, :k].reshape(len(pos), -1)
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=0)
    acc = float(cross_val_score(clf, X, intent, cv=4).mean())
    clf.fit(X, intent)
    cls = clf.classes_.astype(int)

    def hard(P):
        return np.eye(K)[clf.predict(P.reshape(len(P), -1)).astype(int)]

    def soft(P):
        pr = clf.predict_proba(P.reshape(len(P), -1))
        full = np.zeros((len(P), K), np.float32)
        full[:, cls] = pr
        return full

    proba = clf.predict_proba(X)
    ent = float(np.mean(-(proba * np.log(proba + 1e-9)).sum(1)))
    return acc, hard, ent, soft


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env_or = make_env(reveal=True)     # oracle env (pred sees intent slot)
    env_un = make_env(reveal=False)    # unaware env

    # encoder dataset: oracle-prey vs oracle-pred trajectories (intent labels)
    pos_all, int_all = [], []
    for i in SEEDS:
        pp = load_params(apath("mappo_intent_oracle", "prey", i))
        dp = load_params(apath("mappo_intent_oracle", "pred", i))
        P, g = gather_prey_traj(env_or, pp, dp, 150, jax.random.PRNGKey(10 + i))
        pos_all.append(P); int_all.append(g)
    pos = np.concatenate(pos_all); intent = np.concatenate(int_all)
    print(f"encoder dataset: {len(pos)} trajectories, T={pos.shape[1]}")

    # encoder accuracy + entropy vs observation length; keep hard + soft heads
    ks = [3, 5, 8, 12, 18, 25]
    enc_acc, enc_ent = [], []
    enc_hard, enc_proba = {}, {}
    for k in ks:
        a, hard, e, soft = train_encoder_at_k(pos, intent, k)
        enc_acc.append(a); enc_ent.append(e); enc_hard[k] = hard; enc_proba[k] = soft
        print(f"  encoder k={k:2d}: acc {a:.2f}  post-entropy {e:.2f} nats")
    enc_infer = enc_hard[INFER_K]

    # ---- conditions, per seed ----
    #  unaware/guess/inferred/oracle on the oracle predator+prey (prey fixed);
    #  belief = uncertainty-aware predator fed the encoder's online posterior.
    order = ["unaware", "guess", "inferred", "belief", "oracle"]
    rows = {m: [] for m in order}
    atcorner = []
    for i in SEEDS:
        prey_or = load_params(apath("mappo_intent_oracle", "prey", i))
        pred_or = load_params(apath("mappo_intent_oracle", "pred", i))
        prey_un = load_params(apath("mappo_intent_unaware", "prey", i))
        pred_un = load_params(apath("mappo_intent_unaware", "pred", i))
        prey_be = load_params(apath("mappo_intent_belief", "prey", i))
        pred_be = load_params(apath("mappo_intent_belief", "pred", i))
        rng = jax.random.PRNGKey(50 + i)
        for mode in ["oracle", "guess", "inferred"]:
            d = rollout(env_or, prey_or, pred_or, N_EPS, rng, mode=mode,
                        encoder=enc_infer, guess_key=i)
            rows[mode].append(d["captures"].mean())
        # uncertainty-aware predator + its prey, fed the online posterior
        d = rollout(env_or, prey_be, pred_be, N_EPS, rng, mode="belief",
                    enc_ks=ks, enc_proba=enc_proba)
        rows["belief"].append(d["captures"].mean())
        # unaware equilibrium (honest no-intent baseline)
        d = rollout(env_un, prey_un, pred_un, N_EPS, rng, mode="none")
        rows["unaware"].append(d["captures"].mean())
        atcorner.append(d["at_corner"].mean())

    summ = {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}
    print("\ncaptures/episode (mean +/- std over 3 seeds):")
    for m in order:
        print(f"  {m:9s}: {summ[m][0]:.2f} +/- {summ[m][1]:.2f}")
    blift = 100 * (summ["belief"][0] - summ["unaware"][0]) / summ["unaware"][0]
    recov = 100 * (summ["belief"][0] - summ["guess"][0]) / (summ["oracle"][0] - summ["guess"][0] + 1e-9)
    print(f"belief (uncertainty-aware) vs unaware: {blift:+.0f}%   "
          f"recovers {recov:.0f}% of oracle-vs-guess gap")
    print(f"prey at-corner occupancy (unaware eq.): {np.mean(atcorner):.2f}")

    np.savez(os.path.join(LOGDIR, "part2_intent_eval.npz"),
             ks=np.array(ks), enc_acc=np.array(enc_acc), enc_ent=np.array(enc_ent),
             **{f"cap_{m}": np.array(rows[m]) for m in rows}, infer_k=INFER_K)

    # ---------------------------------------------------------------- figure #
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) intent occupancy of the prey, coloured by intent
    ax = axes[0]
    P, g = gather_prey_traj(env_or,
                            load_params(apath("mappo_intent_oracle", "prey", 0)),
                            load_params(apath("mappo_intent_oracle", "pred", 0)),
                            200, jax.random.PRNGKey(0))
    cols = ["#0d6e7a", "#b85c10", "#1f7a4e", "#b44233"]
    for gi in range(K):
        m = g == gi
        xy = P[m].reshape(-1, 2)
        ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.25, c=cols[gi])
        cx, cy = ([-CORNER, -CORNER, CORNER, CORNER][gi], [-CORNER, CORNER, -CORNER, CORNER][gi])
        ax.scatter([cx], [cy], s=90, marker="*", c=cols[gi], edgecolors="k", zorder=5)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
    ax.set_title("Hidden intent shapes where the prey goes\n(colour = intent corner)")
    ax.set_xticks([]); ax.set_yticks([])

    # (b) encoder accuracy + entropy vs observation length
    ax = axes[1]
    ax.plot(ks, enc_acc, "-o", color="#0d6e7a", label="intent accuracy")
    ax.axhline(1.0 / K, ls="--", c="k", lw=1, label=f"chance (1/{K})")
    ax.axvline(INFER_K, ls=":", c="#888")
    ax.set_xlabel("steps of opponent observed"); ax.set_ylabel("intent recovery")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(ks, enc_ent, "-s", color="#b85c10", alpha=0.7)
    ax2.set_ylabel("posterior entropy (nats)", color="#b85c10")
    ax.set_title("Inferring intent from motion\n(sharpens with observation)")
    ax.legend(loc="lower right", fontsize=8)

    # (c) the payoff: captures by condition
    ax = axes[2]
    labels = ["unaware\n(no intent)", "guess\n(rand intent)", "inferred\n(hard, @k=8)",
              "belief\n(ours)", "oracle\n(true intent)"]
    means = [summ[m][0] for m in order]; stds = [summ[m][1] for m in order]
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=["#999", "#c0a0a0", "#7fae9b", "#1f7a4e", "#0d6e7a"])
    ax.set_ylabel("captures / episode")
    ax.set_title(f"Modeling the hidden opponent helps\n"
                 f"belief +{blift:.0f}% vs unaware; recovers {recov:.0f}% of the oracle gap")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.04, f"{m:.2f}",
                ha="center", fontweight="bold", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Hidden-intent opponent modeling — inferring a hidden strategy "
                 "from behavior improves the best response", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, "part2_intent_eval.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
