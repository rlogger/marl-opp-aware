import numpy as np

from mopa.results import (
    check_bc_latent_deploy,
    check_bc_latent_sweep,
    check_bc_vs_mappo,
    run_checks,
)


def assert_all_pass(checks):
    failed = [c for c in checks if not c.passed]
    assert failed == []


def test_bc_vs_mappo_thresholds_pass_for_verified_shape(tmp_path):
    path = tmp_path / "mopa_bc_vs_mappo.npz"
    np.savez(
        path,
        random_circle=np.array([0.40, 0.42]),
        random_corners=np.array([0.38, 0.40]),
        bc_circle=np.array([1.22, 1.20]),
        bc_corners=np.array([1.24, 1.21]),
        mappo_circle=np.array([1.35, 1.34]),
        mappo_corners=np.array([1.37, 1.33]),
        match=np.array([0.80, 0.81]),
    )

    assert_all_pass(check_bc_vs_mappo(path))


def test_bc_latent_sweep_thresholds_pass_for_scaling_result(tmp_path):
    path = tmp_path / "mopa_bc_latent_sweep.npz"
    np.savez(
        path,
        vanilla=np.array([0.766, 0.003]),
        oracle=np.array([0.792, 0.004]),
        vae=np.array([0.764, 0.777, 0.786, 0.795]),
        jepa=np.array([0.762, 0.763, 0.764, 0.764]),
        probe_vae=np.array([0.60, 0.73, 0.78, 0.78]),
        probe_jepa=np.array([0.55, 0.60, 0.62, 0.64]),
    )

    assert_all_pass(check_bc_latent_sweep(path))


def test_bc_latent_deploy_thresholds_keep_caveat_modest(tmp_path):
    path = tmp_path / "mopa_bc_latent_deploy.npz"
    np.savez(
        path,
        random=np.array([0.35, 0.40, 0.42]),
        vanilla=np.array([0.94, 0.96, 0.95]),
        latent=np.array([0.98, 0.99, 1.00]),
        oracle=np.array([1.10, 1.12, 1.11]),
        mappo=np.array([1.13, 1.15, 1.14]),
    )

    assert_all_pass(check_bc_latent_deploy(path))


def test_run_checks_reports_missing_artifacts(tmp_path):
    checks, missing = run_checks(tmp_path, allow_missing=False)

    assert checks == []
    assert {m.path.name for m in missing} == {
        "mopa_bc_vs_mappo.npz",
        "mopa_bc_latent_sweep.npz",
        "mopa_bc_latent_deploy.npz",
    }
