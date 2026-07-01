"""Publication-oriented checks for regenerated experiment artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from mopa.paths import LOG_DIR


@dataclass(frozen=True)
class Check:
    name: str
    observed: float
    threshold: float
    passed: bool
    detail: str


@dataclass(frozen=True)
class MissingArtifact:
    path: Path
    detail: str


def _mean(*arrays: np.ndarray) -> float:
    return float(np.concatenate([np.asarray(a).ravel() for a in arrays]).mean())


def _check_at_least(name: str, observed: float, threshold: float, detail: str) -> Check:
    return Check(name, observed, threshold, observed >= threshold, detail)


def check_bc_vs_mappo(path: str | Path) -> list[Check]:
    """Check that vanilla BC is a competitive clone of the MAPPO expert."""

    d = np.load(path)
    random = _mean(d["random_circle"], d["random_corners"])
    bc = _mean(d["bc_circle"], d["bc_corners"])
    mappo = _mean(d["mappo_circle"], d["mappo_corners"])
    recovery = (bc - random) / (mappo - random + 1e-9)
    match = float(np.asarray(d["match"]).mean())
    return [
        _check_at_least(
            "bc_vs_mappo/recovery",
            recovery,
            0.75,
            "BC should recover at least 75% of MAPPO's edge over random.",
        ),
        _check_at_least(
            "bc_vs_mappo/action_match",
            match,
            0.75,
            "Held-out episode-level action match should be high enough for BC to be a faithful clone.",
        ),
        _check_at_least(
            "bc_vs_mappo/expert_gap",
            mappo - random,
            0.50,
            "The expert must remain meaningfully above random.",
        ),
    ]


def check_bc_latent_sweep(path: str | Path) -> list[Check]:
    """Check that latent-conditioned BC improves when the latent is informative."""

    d = np.load(path)
    vanilla = float(np.asarray(d["vanilla"])[0])
    oracle = float(np.asarray(d["oracle"])[0])
    vae = np.asarray(d["vae"], dtype=float)
    jepa = np.asarray(d["jepa"], dtype=float)
    best_latent = float(max(vae.max(), jepa.max()))
    best_probe = float(max(np.asarray(d["probe_vae"]).max(), np.asarray(d["probe_jepa"]).max()))
    return [
        _check_at_least(
            "bc_latent_sweep/best_gain",
            best_latent - vanilla,
            0.01,
            "A useful latent should beat vanilla action accuracy by at least one point.",
        ),
        _check_at_least(
            "bc_latent_sweep/oracle_gap_closed",
            best_latent - (oracle - 0.01),
            0.0,
            "Best latent-conditioned BC should reach the oracle accuracy band.",
        ),
        _check_at_least(
            "bc_latent_sweep/latent_probe",
            best_probe,
            0.75,
            "The latent used for the win should linearly decode placement.",
        ),
    ]


def check_bc_latent_deploy(path: str | Path) -> list[Check]:
    """Check deployed-captures evidence for value of the strategy signal."""

    d = np.load(path)
    vanilla = float(np.asarray(d["vanilla"]).mean())
    latent = float(np.asarray(d["latent"]).mean())
    oracle = float(np.asarray(d["oracle"]).mean())
    mappo = float(np.asarray(d["mappo"]).mean())
    gap = mappo - vanilla
    oracle_recovery = (oracle - vanilla) / (gap + 1e-9)
    return [
        _check_at_least(
            "bc_latent_deploy/oracle_recovery",
            oracle_recovery,
            0.70,
            "True placement should recover most of the pooled-BC to MAPPO deployed gap.",
        ),
        _check_at_least(
            "bc_latent_deploy/latent_not_worse",
            latent - vanilla,
            -0.02,
            "The unsupervised latent should not materially hurt deployed captures.",
        ),
        _check_at_least(
            "bc_latent_deploy/expert_gap",
            gap,
            0.10,
            "The deployed setup should leave measurable headroom above pooled BC.",
        ),
    ]


ARTIFACT_CHECKS = {
    "mopa_bc_vs_mappo.npz": check_bc_vs_mappo,
    "mopa_bc_latent_sweep.npz": check_bc_latent_sweep,
    "mopa_bc_latent_deploy.npz": check_bc_latent_deploy,
}


def run_checks(
    log_dir: str | Path = LOG_DIR,
    allow_missing: bool = False,
) -> tuple[list[Check], list[MissingArtifact]]:
    """Run all available publication checks."""

    root = Path(log_dir)
    checks: list[Check] = []
    missing: list[MissingArtifact] = []
    for name, fn in ARTIFACT_CHECKS.items():
        path = root / name
        if not path.exists():
            item = MissingArtifact(path, "Regenerate the corresponding experiment before claiming this result.")
            missing.append(item)
            continue
        checks.extend(fn(path))
    if missing and not allow_missing:
        return checks, missing
    return checks, missing


def format_report(checks: Iterable[Check], missing: Iterable[MissingArtifact]) -> str:
    lines = ["Experiment Result Checks"]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(
            f"{status:4s} {check.name}: observed={check.observed:.3f}, "
            f"threshold={check.threshold:.3f} -- {check.detail}"
        )
    for item in missing:
        lines.append(f"MISS {item.path}: {item.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    checks, missing = run_checks(args.log_dir, allow_missing=args.allow_missing)
    print(format_report(checks, missing))
    failed = [c for c in checks if not c.passed]
    if failed or (missing and not args.allow_missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
