"""Repository paths and output-directory helpers.

Research scripts should write plots and raw metrics through this module instead
of hard-coding relative paths. That keeps command-line runs reproducible from
any working directory and avoids silent failures when ignored output folders do
not exist yet.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs" / "MPE_simple_tag_v3"
PLOT_DIR = REPO_ROOT / "plots"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_output_dirs() -> tuple[Path, Path]:
    """Ensure the standard raw-result and figure directories exist."""

    return ensure_dir(LOG_DIR), ensure_dir(PLOT_DIR)


def log_path(name: str) -> Path:
    """Path for a raw experiment artifact under ``logs/MPE_simple_tag_v3``."""

    ensure_dir(LOG_DIR)
    return LOG_DIR / name


def plot_path(name: str) -> Path:
    """Path for a figure under ``plots``."""

    ensure_dir(PLOT_DIR)
    return PLOT_DIR / name
