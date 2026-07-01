"""'Circles are just squares' — look at the shapes in the heatmap.

A meeting question: are the circle-specialist and corners-specialist prey
actually doing different things, or does a circle of resources end up looking
like the four corners? We answer it directly: the mean occupancy of each
specialist prey over the arena, side by side, with the resource geometry drawn
on top, plus their difference.

If the two heatmaps look alike, that IS the explanation for why the strategy
latent does not cluster (ARI~0): the behaviours are not that distinct. It also
motivates placements that are genuinely different (circle vs a snake path).

Output: plots/occupancy_heatmaps.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from mopa import legacy
from mopa.data import specialist_dataset

BINS = 24
RNG = (-1.3, 1.3)
CIRCLE_R = 0.6
CORNER = 0.8


def occ_grid(prey_pos, t0=0, t1=100):
    """Mean occupancy over episodes: (BINS, BINS) probability grid."""
    xs = prey_pos[:, t0:t1, 0].ravel()
    ys = prey_pos[:, t0:t1, 1].ravel()
    h, _, _ = np.histogram2d(xs, ys, bins=BINS, range=[RNG, RNG])
    return (h / h.sum()).T                      # .T so imshow x=cols


def draw_geom(ax, placement):
    if placement == "circle":
        ax.add_patch(Circle((0, 0), CIRCLE_R, fill=False, ec="w", ls="--", lw=1.6))
    else:
        for cx in (-CORNER, CORNER):
            for cy in (-CORNER, CORNER):
                ax.add_patch(Rectangle((cx - 0.07, cy - 0.07), 0.14, 0.14,
                                       fill=False, ec="w", lw=1.6))
    for ox, oy in [(0.5, 0.5), (-0.5, -0.5)]:                # obstacles
        ax.add_patch(Circle((ox, oy), 0.1, color="k", alpha=0.35))


def main():
    os.makedirs(legacy.PLOTDIR, exist_ok=True)
    print("rolling out specialists (100 steps)...")
    ds = specialist_dataset(n_eps=150, num_steps=100)
    y = ds["label"]
    gc = occ_grid(ds["prey_pos"][y == 0])       # circle
    gs = occ_grid(ds["prey_pos"][y == 1])       # corners
    diff = gc - gs

    # a scalar "how different" summary: total-variation distance between the
    # two occupancy distributions, and their cosine similarity.
    tv = 0.5 * np.abs(gc - gs).sum()
    cos = float((gc * gs).sum() / (np.linalg.norm(gc) * np.linalg.norm(gs)))
    print(f"  occupancy TV distance {tv:.3f}   cosine similarity {cos:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.9))
    vmax = max(gc.max(), gs.max())
    for a, (g, name, pl) in zip(ax[:2],
            [(gc, "circle specialist", "circle"), (gs, "corners specialist", "corners")]):
        im = a.imshow(g, origin="lower", extent=[RNG[0], RNG[1], RNG[0], RNG[1]],
                      cmap="magma", vmin=0, vmax=vmax)
        draw_geom(a, pl)
        a.set_title(f"prey occupancy — {name}", fontsize=11)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, shrink=0.85)
    d = ax[2]
    m = np.abs(diff).max()
    im = d.imshow(diff, origin="lower", extent=[RNG[0], RNG[1], RNG[0], RNG[1]],
                  cmap="RdBu_r", vmin=-m, vmax=m)
    d.set_title(f"circle − corners  (TV {tv:.2f}, cosine {cos:.2f})", fontsize=11)
    d.set_xticks([]); d.set_yticks([])
    fig.colorbar(im, ax=d, fraction=0.046, shrink=0.85)
    fig.suptitle("Are circle and corners genuinely different behaviours? "
                 "(mean prey occupancy, 150 episodes)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(legacy.PLOTDIR, "occupancy_heatmaps.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    np.savez(os.path.join(legacy.LOGDIR, "occupancy_heatmaps.npz"),
             circle=gc, corners=gs, tv=tv, cosine=cos)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
