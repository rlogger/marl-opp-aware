"""Generate a prey trajectory dataset from the random-placement checkpoints only.

Only the `random` placement was trained for all three algorithms, so the full
build_dataset_A (which expects circle/corners/random) can't run. This mirrors
how the MAPPO _A dataset was produced: 3 random-seed checkpoints x 100 eps,
evaluated in the random env so both circle and corners layouts appear.

Usage:
    python src/gen_traj_random_only.py --algorithm iql
    python src/gen_traj_random_only.py --algorithm oa_iql
"""
import argparse
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import (
    rollout_one_checkpoint, LOGDIR, NUM_EPS_A,
)

SEEDS = [0, 1, 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", default="iql", choices=["iql", "oa_iql", "mappo"])
    p.add_argument("--eval_placement", default="random")
    p.add_argument("--ckpt_suffix", default="",
                   help="extra checkpoint-name suffix, e.g. '_hr' for high-reward "
                        "variant. The env still uses --eval_placement.")
    p.add_argument("--out_tag", default="",
                   help="override output dataset suffix (default = algorithm)")
    args = p.parse_args()

    # Optionally retarget the checkpoint name (e.g. random -> random_hr) while
    # keeping the evaluation env at --eval_placement.
    if args.ckpt_suffix:
        _orig = G._params_path
        def _patched(algorithm, placement, team, seed_idx):
            return _orig(algorithm, placement + args.ckpt_suffix, team, seed_idx)
        G._params_path = _patched

    rng = jax.random.PRNGKey(0)
    parts, labels, checkpoints = [], [], []
    for label_id, seed in enumerate(SEEDS):
        rng, k = jax.random.split(rng)
        tag = f"random_seed{seed}"
        print(f"  [{args.algorithm}] {tag} ({label_id+1}/{len(SEEDS)}) N={NUM_EPS_A} ...",
              flush=True)
        d = rollout_one_checkpoint(args.algorithm, "random", seed,
                                   NUM_EPS_A, args.eval_placement, k)
        parts.append(d)
        labels.append(np.full(d["positions"].shape[0], label_id, np.int32))
        checkpoints.append(tag)

    out = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}
    out["labels"] = np.concatenate(labels, axis=0)
    out["checkpoints"] = np.array(checkpoints, dtype="<U12")

    suffix = args.out_tag or args.algorithm
    path = os.path.join(LOGDIR, f"trajectory_dataset_resources_{suffix}_A.npz")
    np.savez(path, **out)
    print(f"saved {path}  positions.shape={out['positions'].shape}")


if __name__ == "__main__":
    main()
