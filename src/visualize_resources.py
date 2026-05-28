"""Render rollout GIFs for the resource-collection environment.

No trained policy needed -- uses random actions to show the arena layout,
resource positions, and the collection mechanic. Produces one GIF per
placement mode (circle, corners) plus a side-by-side comparison diagram.
"""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch, RegularPolygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_tag_resources import SimpleTagResourcesMPE, ResourceState

OBS = [[0.5, 0.5], [-0.5, -0.5]]

PRED_COLOR = np.array([255, 75, 75]) / 255     # red predators
PREY_COLOR = np.array([75, 75, 255]) / 255      # blue prey
LANDMARK_COLOR = np.array([60, 60, 60]) / 255   # dark grey obstacles
RES_COLOR_LIVE = np.array([50, 205, 50]) / 255  # green resource (uncollected)
RES_COLOR_DEAD = np.array([200, 200, 200]) / 255  # grey resource (collected)

ARENA_LIM = 1.6
STEPS = 25


def make_env(placement):
    return SimpleTagResourcesMPE(
        num_resources=4,
        placement=placement,
        collect_radius=0.15,
        collect_reward=5.0,
        obstacle_positions=OBS,
    )


def rollout_random(env, seed=0):
    """Run one episode with random actions, collecting states."""
    rng = jax.random.PRNGKey(seed)
    rng, key_reset = jax.random.split(rng)
    obs, state = env.reset(key_reset)

    states = [state]
    for t in range(STEPS):
        rng, key_act, key_step = jax.random.split(rng, 3)
        actions = {}
        for a in env.agents:
            actions[a] = jax.random.randint(key_act, (), 0, env.action_spaces[a].n)
            rng, key_act = jax.random.split(rng)
        obs, state, rewards, dones, info = env.step_env(key_step, state, actions)
        states.append(state)
    return states


def render_gif(env, states, outpath, title=None):
    """Render a rollout as a GIF with resources shown."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_xlim([-ARENA_LIM, ARENA_LIM])
    ax.set_ylim([-ARENA_LIM, ARENA_LIM])
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f7f4")
    fig.patch.set_facecolor("#f8f7f4")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    s0 = states[0]
    # Agents (predators then prey)
    agent_circles = []
    for i in range(env.num_agents):
        color = PRED_COLOR if i < env.num_adversaries else PREY_COLOR
        c = Circle(np.asarray(s0.p_pos[i]), env.rad[i], color=color,
                   ec="white", linewidth=1.2, zorder=10)
        ax.add_patch(c)
        agent_circles.append(c)

    # Landmarks (obstacles)
    landmark_circles = []
    for i in range(env.num_landmarks):
        idx = env.num_agents + i
        c = Circle(np.asarray(s0.p_pos[idx]), env.rad[idx],
                   color=LANDMARK_COLOR, alpha=0.5, zorder=5)
        ax.add_patch(c)
        landmark_circles.append(c)

    # Resources (diamonds)
    resource_markers = []
    rp = np.asarray(s0.resource_pos)
    collected = np.asarray(s0.collected)
    for j in range(env.num_resources):
        color = RES_COLOR_DEAD if collected[j] else RES_COLOR_LIVE
        diamond = RegularPolygon(rp[j], numVertices=4, radius=0.07,
                                  orientation=np.pi/4, color=color,
                                  ec="white", linewidth=1.0, zorder=8)
        ax.add_patch(diamond)
        resource_markers.append(diamond)

    # Labels
    for i in range(env.num_adversaries):
        ax.annotate(f"P{i+1}", np.asarray(s0.p_pos[i]), color="white",
                    fontsize=7, fontweight="bold", ha="center", va="center", zorder=11)
    prey_label = ax.annotate("prey", np.asarray(s0.p_pos[env.num_adversaries]),
                             color="white", fontsize=7, fontweight="bold",
                             ha="center", va="center", zorder=11)

    step_text = ax.text(-ARENA_LIM + 0.08, ARENA_LIM - 0.08, "Step: 0",
                        fontsize=9, va="top", fontfamily="monospace",
                        color="#555")
    collect_text = ax.text(ARENA_LIM - 0.08, ARENA_LIM - 0.08, "Collected: 0/4",
                           fontsize=9, va="top", ha="right", fontfamily="monospace",
                           color="#2a7a2a")

    # Prey trail
    trail_x, trail_y = [float(s0.p_pos[env.num_adversaries][0])], [float(s0.p_pos[env.num_adversaries][1])]
    trail_line, = ax.plot(trail_x, trail_y, '-', color=PREY_COLOR, alpha=0.25,
                          linewidth=1.5, zorder=3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PRED_COLOR,
               markersize=10, label='Predator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PREY_COLOR,
               markersize=10, label='Prey'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=RES_COLOR_LIVE,
               markersize=8, label='Resource'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LANDMARK_COLOR,
               markersize=8, label='Obstacle', alpha=0.5),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7,
              framealpha=0.85, edgecolor="#ccc")

    def update(frame):
        s = states[frame]
        p_pos = np.asarray(s.p_pos)
        coll = np.asarray(s.collected)

        # Update agents
        for i, c in enumerate(agent_circles):
            c.center = p_pos[i]

        # Update predator labels
        for i in range(env.num_adversaries):
            pass  # static labels would need separate text objects

        prey_label.set_position(p_pos[env.num_adversaries])

        # Update resources
        for j, m in enumerate(resource_markers):
            m.set_color(RES_COLOR_DEAD if coll[j] else RES_COLOR_LIVE)
            m.set_alpha(0.3 if coll[j] else 1.0)

        # Update trail
        trail_x.append(float(p_pos[env.num_adversaries][0]))
        trail_y.append(float(p_pos[env.num_adversaries][1]))
        trail_line.set_data(trail_x, trail_y)

        step_text.set_text(f"Step: {int(s.step)}")
        collect_text.set_text(f"Collected: {int(coll.sum())}/4")

    ani = animation.FuncAnimation(fig, update, frames=len(states),
                                   blit=False, interval=200)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ani.save(outpath, writer="pillow", dpi=120)
    plt.close(fig)
    print(f"Wrote {outpath} ({os.path.getsize(outpath) / 1024:.0f} KB)")


def render_placement_diagram(outpath):
    """Static side-by-side diagram of circle vs corners placement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#f8f7f4")

    for ax, mode, title in [(ax1, "circle", "Circle Placement"),
                             (ax2, "corners", "Corners Placement")]:
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f7f4")
        ax.grid(True, alpha=0.12, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#ccc")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

        # Obstacles
        for pos in OBS:
            c = Circle(pos, 0.1, color=LANDMARK_COLOR, alpha=0.4, zorder=5)
            ax.add_patch(c)
            ax.annotate("obs", pos, color="white", fontsize=6,
                        ha="center", va="center", zorder=6)

        env = make_env(mode)
        if mode == "circle":
            rp = np.asarray(env._circle_positions)
            # Draw the circle path
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(0.6 * np.cos(theta), 0.6 * np.sin(theta),
                    '--', color=RES_COLOR_LIVE, alpha=0.3, linewidth=1.5)
            # Arrow showing collection loop
            for i in range(len(rp)):
                j = (i + 1) % len(rp)
                dx, dy = rp[j] - rp[i]
                ax.annotate("", xy=rp[j]*0.85, xytext=rp[i]*0.85,
                            arrowprops=dict(arrowstyle="->", color=PREY_COLOR,
                                            alpha=0.3, lw=1.2))
        else:
            rp = np.asarray(env._corner_positions)
            # Draw dashed path between corners
            order = [0, 2, 3, 1, 0]  # zigzag
            for k in range(len(order) - 1):
                ax.plot([rp[order[k], 0], rp[order[k+1], 0]],
                        [rp[order[k], 1], rp[order[k+1], 1]],
                        '--', color=PREY_COLOR, alpha=0.25, linewidth=1.2)

        # Resources
        for j in range(len(rp)):
            diamond = RegularPolygon(rp[j], numVertices=4, radius=0.08,
                                      orientation=np.pi/4, color=RES_COLOR_LIVE,
                                      ec="white", linewidth=1.2, zorder=8)
            ax.add_patch(diamond)
            ax.annotate(f"R{j+1}", rp[j], color="white", fontsize=6,
                        fontweight="bold", ha="center", va="center", zorder=9)

        # Origin marker
        ax.plot(0, 0, '+', color="#aaa", markersize=10, markeredgewidth=1)

        # Annotations
        if mode == "circle":
            ax.annotate("r = 0.6", xy=(0.42, 0.42), fontsize=8, color="#666",
                        style="italic")
        else:
            ax.annotate("offset = 0.8", xy=(0.15, -1.15), fontsize=8,
                        color="#666", style="italic")

    fig.tight_layout(pad=2.0)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {outpath} ({os.path.getsize(outpath) / 1024:.0f} KB)")


def render_obs_diagram(outpath):
    """Diagram comparing predator vs prey observation vectors."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#f8f7f4")
    ax.set_facecolor("#f8f7f4")
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 3.5])
    ax.axis("off")

    # Predator observation bar
    y_pred = 2.5
    pred_sections = [
        (0, 2, "#e06050", "vel (2)"),
        (2, 6, "#d4887a", "agent pos (4)"),
        (6, 10, "#cba0a0", "landmark (4)"),
        (10, 16, "#bfb0b0", "other agents (6)"),
        (16, 26, "#e8e8e8", ""),
    ]
    for x0, x1, color, label in pred_sections:
        ax.barh(y_pred, x1 - x0, left=x0, height=0.5, color=color,
                edgecolor="white", linewidth=1)
        if label:
            ax.text((x0 + x1) / 2, y_pred, label, ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white" if color[1] < 'c' else "#333")
    ax.text(-0.5, y_pred, "Predator (16-d)", ha="right", va="center",
            fontsize=9, fontweight="bold", color=PRED_COLOR)

    # Prey observation bar
    y_prey = 1.2
    prey_sections = [
        (0, 2, "#5060e0", "vel (2)"),
        (2, 6, "#7a88d4", "agent pos (4)"),
        (6, 10, "#a0a0cb", "landmark (4)"),
        (10, 14, "#b0b0bf", "other agents (4)"),
        (14, 22, "#32cd32", "resource pos (8)"),
        (22, 26, "#28a428", "collected (4)"),
    ]
    for x0, x1, color, label in prey_sections:
        ax.barh(y_prey, x1 - x0, left=x0, height=0.5, color=color,
                edgecolor="white", linewidth=1)
        if label:
            fc = "white" if x0 >= 14 or x0 < 6 else "#333"
            ax.text((x0 + x1) / 2, y_prey, label, ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color=fc)
    ax.text(-0.5, y_prey, "Prey (26-d)", ha="right", va="center",
            fontsize=9, fontweight="bold", color=PREY_COLOR)

    # Bracket for resource-only dims
    ax.annotate("", xy=(14, 0.7), xytext=(26, 0.7),
                arrowprops=dict(arrowstyle="<->", color="#2a7a2a", lw=1.5))
    ax.text(20, 0.35, "resource info (12 dims)\nprey-only", ha="center",
            fontsize=8, color="#2a7a2a", fontstyle="italic")

    ax.set_title("Observation Space Comparison", fontsize=13,
                 fontweight="bold", pad=12)

    fig.tight_layout(pad=1.5)
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {outpath} ({os.path.getsize(outpath) / 1024:.0f} KB)")


if __name__ == "__main__":
    print("Generating placement diagram...")
    render_placement_diagram("plots/resource_placement_modes.png")

    print("Generating observation diagram...")
    render_obs_diagram("plots/resource_obs_comparison.png")

    for mode in ("circle", "corners"):
        print(f"Generating {mode} rollout GIF...")
        env = make_env(mode)
        states = rollout_random(env, seed=42)
        render_gif(env, states, f"plots/rollout_resource_{mode}.gif",
                   title=f"Resource Environment ({mode.title()} Placement)")

    print("Done.")
