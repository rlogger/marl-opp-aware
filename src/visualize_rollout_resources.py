"""Render rollout GIFs from trained checkpoints on the resource environment.

Loads IQL, OA-IQL, or MAPPO checkpoints and runs greedy rollouts,
producing annotated GIFs that show agent movement, resource collection,
prey trails, and step-by-step reward accumulation.

Usage:
    python src/visualize_rollout_resources.py --algorithm mappo --placement random --seed 0
    python src/visualize_rollout_resources.py --algorithm iql   --placement random --seed 0
    python src/visualize_rollout_resources.py --all   # generates all combinations
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_tag_resources import SimpleTagResourcesMPE
from jaxmarl.wrappers.baselines import load_params

LOGDIR    = "logs/MPE_simple_tag_v3"
OBSTACLES = [[0.5, 0.5], [-0.5, -0.5]]
HIDDEN    = 128
STEPS     = 25

# Colors
PRED_COLOR = np.array([255, 75, 75]) / 255
PREY_COLOR = np.array([75, 75, 255]) / 255
LANDMARK_COLOR = np.array([60, 60, 60]) / 255
RES_LIVE   = np.array([50, 205, 50]) / 255
RES_DEAD   = np.array([200, 200, 200]) / 255

ALG_COLORS = {"iql": "#2196F3", "oa_iql": "#FF9800", "mappo": "#4CAF50"}
ALG_LABELS = {"iql": "IQL", "oa_iql": "OA-IQL", "mappo": "MAPPO"}

ARENA_LIM = 1.6


# ── Network architectures (inference only) ────────────────────────

class MLPQNetwork(nn.Module):
    action_dim: int; hidden_dim: int; init_scale: float = 1.0
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                        bias_init=constant(0.0))(x)


class MLPQOppNetwork(nn.Module):
    action_dim: int; hidden_dim: int; opp_n_agents: int; init_scale: float = 1.0
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        opp = nn.Dense(self.opp_n_agents * self.action_dim,
                       kernel_init=orthogonal(self.init_scale),
                       bias_init=constant(0.0))(x)
        opp = opp.reshape(*opp.shape[:-1], self.opp_n_agents, self.action_dim)
        return q, opp


class ActorLogits(nn.Module):
    action_dim: int; hidden_dim: int = 128
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)


# ── Helpers ───────────────────────────────────────────────────────

def split_teams(env):
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey  = [a for a in env.agents if a.startswith("agent")]
    return {"pred": preds, "prey": prey}


def _ckpt(algorithm, placement, team, seed_idx):
    if algorithm == "iql":
        alg = f"iql_teams_resources_{placement}"
    elif algorithm == "oa_iql":
        alg = f"iql_teams_oa_resources_{placement}"
    else:
        alg = f"mappo_teams_resources_{placement}"
    base = f"{LOGDIR}/{alg}_MPE_simple_tag_v3_{team}"
    if algorithm == "mappo":
        return f"{base}_actor_seed0_vmap{seed_idx}.safetensors"
    return f"{base}_seed0_vmap{seed_idx}.safetensors"


def _make_env(placement):
    return SimpleTagResourcesMPE(
        num_resources=4, placement=placement,
        collect_radius=0.15, collect_reward=5.0,
        circle_radius=0.6, corner_offset=0.8,
        obstacle_positions=OBSTACLES,
    )


# ── Rollout ───────────────────────────────────────────────────────

def rollout_trained(algorithm, placement, seed_idx, eval_seed=42):
    """Run one greedy episode using the base env directly (no CTRolloutManager)."""
    base_env = _make_env(placement)
    teams    = split_teams(base_env)
    action_dim = max(base_env.action_spaces[a].n for a in base_env.agents)
    obs_sizes  = {a: base_env.observation_space(a).shape[0] for a in base_env.agents}
    max_obs    = max(obs_sizes.values())  # 26

    # IQL/OA-IQL trained with CTRolloutManager obs (padded + one-hot, 30-d)
    # MAPPO trained with padded obs only (26-d, no one-hot)
    if algorithm in ("iql", "oa_iql"):
        num_agents = len(base_env.agents)
        net_obs_size = max_obs + num_agents  # 30
    else:
        net_obs_size = max_obs  # 26

    if algorithm == "iql":
        nets = {t: MLPQNetwork(action_dim=action_dim, hidden_dim=HIDDEN) for t in teams}
    elif algorithm == "oa_iql":
        nets = {
            "pred": MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                                   opp_n_agents=len(teams["prey"])),
            "prey": MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                                   opp_n_agents=len(teams["pred"])),
        }
    else:
        nets = {t: ActorLogits(action_dim=action_dim, hidden_dim=HIDDEN) for t in teams}

    params = {t: load_params(_ckpt(algorithm, placement, t, seed_idx)) for t in teams}

    rng = jax.random.PRNGKey(eval_seed)
    rng, k_reset = jax.random.split(rng)
    obs, state = base_env.reset(k_reset)

    states = [state]
    pred_rewards_total = 0.0
    prey_rewards_total = 0.0
    reward_trace = []

    def pad_obs(raw_obs, agent_name):
        """Pad obs to network input size (mimic CTRolloutManager)."""
        o = np.asarray(raw_obs)
        padded = np.zeros(max_obs, dtype=np.float32)
        padded[:len(o)] = o
        if algorithm in ("iql", "oa_iql"):
            # Append one-hot agent id
            agent_idx = base_env.agents.index(agent_name)
            onehot = np.zeros(len(base_env.agents), dtype=np.float32)
            onehot[agent_idx] = 1.0
            padded = np.concatenate([padded, onehot])
        return jnp.array(padded)

    for step in range(STEPS):
        rng, k_step = jax.random.split(rng)

        actions = {}
        for t in teams:
            ags = teams[t]
            _obs = jnp.stack([pad_obs(obs[a], a) for a in ags], axis=0)

            if algorithm == "oa_iql":
                out = jax.vmap(nets[t].apply, in_axes=(None, 0))(params[t], _obs)
                q = out[0]
            else:
                q = jax.vmap(nets[t].apply, in_axes=(None, 0))(params[t], _obs)

            a_team = jnp.argmax(q, axis=-1)
            for i, name in enumerate(ags):
                actions[name] = int(a_team[i])

        obs, state, rewards, dones, info = base_env.step_env(k_step, state, actions)
        states.append(state)

        pr = sum(float(rewards[a]) for a in teams["pred"])
        yr = float(rewards[teams["prey"][0]])
        pred_rewards_total += pr
        prey_rewards_total += yr
        reward_trace.append((pr, yr))

    return base_env, states, reward_trace, pred_rewards_total, prey_rewards_total


# ── Rendering ─────────────────────────────────────────────────────

def render_gif(env, states, outpath, title=None, reward_trace=None):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
    ax.set_xlim([-ARENA_LIM, ARENA_LIM])
    ax.set_ylim([-ARENA_LIM, ARENA_LIM])
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f7f4")
    fig.patch.set_facecolor("#f8f7f4")
    ax.grid(True, alpha=0.12, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    s0 = states[0]

    # Agents
    agent_circles = []
    agent_labels = []
    for i in range(env.num_agents):
        pos = np.asarray(s0.p_pos[i])
        color = PRED_COLOR if i < env.num_adversaries else PREY_COLOR
        r = float(env.rad[i])
        c = Circle(pos, r, color=color, ec="white", linewidth=1.2, zorder=10)
        ax.add_patch(c)
        agent_circles.append(c)
        lbl = f"P{i+1}" if i < env.num_adversaries else "prey"
        t = ax.text(pos[0], pos[1], lbl, color="white", fontsize=7,
                    fontweight="bold", ha="center", va="center", zorder=11)
        agent_labels.append(t)

    # Landmarks
    for i in range(env.num_landmarks):
        idx = env.num_agents + i
        pos = np.asarray(s0.p_pos[idx])
        c = Circle(pos, float(env.rad[idx]), color=LANDMARK_COLOR, alpha=0.5, zorder=5)
        ax.add_patch(c)

    # Resources
    resource_markers = []
    rp = np.asarray(s0.resource_pos)
    coll = np.asarray(s0.collected)
    for j in range(env.num_resources):
        color = RES_DEAD if coll[j] else RES_LIVE
        d = RegularPolygon(rp[j], numVertices=4, radius=0.07,
                           orientation=np.pi/4, color=color,
                           ec="white", linewidth=1.0, zorder=8)
        ax.add_patch(d)
        resource_markers.append(d)

    # HUD
    step_text = ax.text(-ARENA_LIM + 0.08, ARENA_LIM - 0.08, "t=0",
                        fontsize=9, va="top", fontfamily="monospace", color="#555")
    collect_text = ax.text(ARENA_LIM - 0.08, ARENA_LIM - 0.08, "0/4",
                           fontsize=9, va="top", ha="right",
                           fontfamily="monospace", color="#2a7a2a")
    reward_text = ax.text(-ARENA_LIM + 0.08, -ARENA_LIM + 0.08, "",
                          fontsize=8, va="bottom", fontfamily="monospace",
                          color="#555")

    # Prey trail
    prey_idx = env.num_adversaries
    trail_x = [float(s0.p_pos[prey_idx][0])]
    trail_y = [float(s0.p_pos[prey_idx][1])]
    trail_line, = ax.plot(trail_x, trail_y, '-', color=PREY_COLOR, alpha=0.25,
                          linewidth=1.5, zorder=3)

    # Pred trails (faint)
    pred_trails = []
    for i in range(env.num_adversaries):
        tx = [float(s0.p_pos[i][0])]
        ty = [float(s0.p_pos[i][1])]
        ln, = ax.plot(tx, ty, '-', color=PRED_COLOR, alpha=0.12,
                      linewidth=1.0, zorder=2)
        pred_trails.append((tx, ty, ln))

    # Legend
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PRED_COLOR,
               markersize=10, label='Predator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PREY_COLOR,
               markersize=10, label='Prey'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=RES_LIVE,
               markersize=8, label='Resource'),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7,
              framealpha=0.85, edgecolor="#ccc")

    cum_pred = 0.0
    cum_prey = 0.0

    def update(frame):
        nonlocal cum_pred, cum_prey
        s = states[frame]
        p_pos = np.asarray(s.p_pos)
        col = np.asarray(s.collected)

        for i, c in enumerate(agent_circles):
            c.center = p_pos[i]
            agent_labels[i].set_position(p_pos[i])

        for j, m in enumerate(resource_markers):
            m.set_color(RES_DEAD if col[j] else RES_LIVE)
            m.set_alpha(0.3 if col[j] else 1.0)

        trail_x.append(float(p_pos[prey_idx][0]))
        trail_y.append(float(p_pos[prey_idx][1]))
        trail_line.set_data(trail_x, trail_y)

        for i in range(env.num_adversaries):
            tx, ty, ln = pred_trails[i]
            tx.append(float(p_pos[i][0]))
            ty.append(float(p_pos[i][1]))
            ln.set_data(tx, ty)

        step_text.set_text(f"t={frame}")
        collect_text.set_text(f"{int(col.sum())}/4 collected")

        if reward_trace and frame > 0 and frame <= len(reward_trace):
            pr, yr = reward_trace[frame - 1]
            cum_pred += pr
            cum_prey += yr
            reward_text.set_text(
                f"pred: {cum_pred:+.0f}  prey: {cum_prey:+.0f}")

    ani = animation.FuncAnimation(fig, update, frames=len(states),
                                  blit=False, interval=200)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="pillow", dpi=120)
    plt.close(fig)
    print(f"  {outpath} ({os.path.getsize(outpath) / 1024:.0f} KB)")


def render_comparison_gif(envs_states, outpath, placement):
    """Side-by-side 3-panel GIF comparing IQL, OA-IQL, MAPPO."""
    n = len(envs_states)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f8f7f4")
    fig.suptitle(f"Algorithm Comparison — {placement} placement",
                 fontsize=14, fontweight="bold", y=0.98)

    panels = []
    for idx, (alg, env, sts, rtrace) in enumerate(envs_states):
        ax = axes[idx]
        ax.set_xlim([-ARENA_LIM, ARENA_LIM])
        ax.set_ylim([-ARENA_LIM, ARENA_LIM])
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f7f4")
        ax.grid(True, alpha=0.12, linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        c = ALG_COLORS.get(alg, "#999")
        ax.set_title(ALG_LABELS.get(alg, alg), fontsize=13,
                     fontweight="bold", color=c, pad=8)

        s0 = sts[0]
        prey_idx = env.num_adversaries

        # Agents
        agent_circles = []
        agent_labels_list = []
        for i in range(env.num_agents):
            pos = np.asarray(s0.p_pos[i])
            color = PRED_COLOR if i < env.num_adversaries else PREY_COLOR
            r = float(env.rad[i])
            circ = Circle(pos, r, color=color, ec="white", linewidth=1.0, zorder=10)
            ax.add_patch(circ)
            agent_circles.append(circ)
            lbl = f"P{i+1}" if i < env.num_adversaries else "prey"
            t = ax.text(pos[0], pos[1], lbl, color="white", fontsize=6,
                        fontweight="bold", ha="center", va="center", zorder=11)
            agent_labels_list.append(t)

        # Landmarks
        for i in range(env.num_landmarks):
            li = env.num_agents + i
            Circle(np.asarray(s0.p_pos[li]), float(env.rad[li]),
                   color=LANDMARK_COLOR, alpha=0.4, zorder=5)
            ax.add_patch(Circle(np.asarray(s0.p_pos[li]), float(env.rad[li]),
                                color=LANDMARK_COLOR, alpha=0.4, zorder=5))

        # Resources
        res_markers = []
        rp = np.asarray(s0.resource_pos)
        coll = np.asarray(s0.collected)
        for j in range(env.num_resources):
            d = RegularPolygon(rp[j], numVertices=4, radius=0.06,
                               orientation=np.pi/4,
                               color=RES_DEAD if coll[j] else RES_LIVE,
                               ec="white", linewidth=0.8, zorder=8)
            ax.add_patch(d)
            res_markers.append(d)

        # HUD
        step_t = ax.text(-ARENA_LIM + 0.06, ARENA_LIM - 0.06, "t=0",
                         fontsize=8, va="top", fontfamily="monospace", color="#555")
        coll_t = ax.text(ARENA_LIM - 0.06, ARENA_LIM - 0.06, "",
                         fontsize=8, va="top", ha="right",
                         fontfamily="monospace", color="#2a7a2a")
        rew_t = ax.text(-ARENA_LIM + 0.06, -ARENA_LIM + 0.06, "",
                        fontsize=7, va="bottom", fontfamily="monospace", color="#555")

        # Trail
        trail_x = [float(s0.p_pos[prey_idx][0])]
        trail_y = [float(s0.p_pos[prey_idx][1])]
        trail_ln, = ax.plot(trail_x, trail_y, '-', color=PREY_COLOR,
                            alpha=0.25, linewidth=1.5, zorder=3)

        panels.append(dict(
            env=env, sts=sts, rtrace=rtrace,
            agent_circles=agent_circles, agent_labels=agent_labels_list,
            res_markers=res_markers, step_t=step_t, coll_t=coll_t, rew_t=rew_t,
            trail_x=trail_x, trail_y=trail_y, trail_ln=trail_ln,
            prey_idx=prey_idx, cum_pred=0.0, cum_prey=0.0,
        ))

    max_frames = max(len(p["sts"]) for p in panels)

    def update(frame):
        for p in panels:
            if frame >= len(p["sts"]):
                continue
            s = p["sts"][frame]
            pos = np.asarray(s.p_pos)
            col = np.asarray(s.collected)
            for i, c in enumerate(p["agent_circles"]):
                c.center = pos[i]
                p["agent_labels"][i].set_position(pos[i])
            for j, m in enumerate(p["res_markers"]):
                m.set_color(RES_DEAD if col[j] else RES_LIVE)
                m.set_alpha(0.3 if col[j] else 1.0)
            pi = p["prey_idx"]
            p["trail_x"].append(float(pos[pi][0]))
            p["trail_y"].append(float(pos[pi][1]))
            p["trail_ln"].set_data(p["trail_x"], p["trail_y"])
            p["step_t"].set_text(f"t={frame}")
            p["coll_t"].set_text(f"{int(col.sum())}/4")
            if p["rtrace"] and frame > 0 and frame <= len(p["rtrace"]):
                pr, yr = p["rtrace"][frame - 1]
                p["cum_pred"] += pr
                p["cum_prey"] += yr
                p["rew_t"].set_text(
                    f"pred:{p['cum_pred']:+.0f} prey:{p['cum_prey']:+.0f}")

    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  blit=False, interval=200)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="pillow", dpi=100)
    plt.close(fig)
    print(f"  {outpath} ({os.path.getsize(outpath) / 1024:.0f} KB)")


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="mappo",
                    choices=["iql", "oa_iql", "mappo"])
    ap.add_argument("--placement", default="random",
                    choices=["circle", "corners", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Generate individual + comparison GIFs for all algorithms")
    args = ap.parse_args()

    if args.all:
        placement = args.placement
        comp_data = []
        for alg in ["iql", "oa_iql", "mappo"]:
            path = _ckpt(alg, placement, "pred", args.seed)
            if not os.path.exists(path):
                print(f"  skip {alg}: {path} not found")
                continue
            print(f"Rolling out {ALG_LABELS[alg]} ({placement}, seed {args.seed})...")
            env, sts, rtrace, ptot, ytot = rollout_trained(
                alg, placement, args.seed, args.eval_seed)
            out = f"plots/rollout_resource_{alg}_{placement}.gif"
            render_gif(env, sts, out,
                       title=f"{ALG_LABELS[alg]} — {placement}",
                       reward_trace=rtrace)
            print(f"  pred total: {ptot:+.0f}  prey total: {ytot:+.0f}")
            comp_data.append((alg, env, sts, rtrace))

        if len(comp_data) >= 2:
            print("Rendering comparison GIF...")
            render_comparison_gif(
                comp_data,
                f"plots/rollout_resource_compare_{placement}.gif",
                placement)
        return

    alg = args.algorithm
    print(f"Rolling out {ALG_LABELS[alg]} ({args.placement}, seed {args.seed})...")
    env, sts, rtrace, ptot, ytot = rollout_trained(
        alg, args.placement, args.seed, args.eval_seed)
    out = args.out or f"plots/rollout_resource_{alg}_{args.placement}.gif"
    render_gif(env, sts, out,
               title=f"{ALG_LABELS[alg]} — {args.placement}",
               reward_trace=rtrace)
    print(f"pred total: {ptot:+.0f}  prey total: {ytot:+.0f}")


if __name__ == "__main__":
    main()
