"""Bootstrap for the legacy src/ modules (checkpoint paths, env builders,
rollout machinery). Importing this module makes `import
generate_trajectory_dataset_resources` work from anywhere."""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

REPO = os.path.dirname(SRC)
LOGDIR = os.path.join(REPO, "logs", "MPE_simple_tag_v3")
PLOTDIR = os.path.join(REPO, "plots")

# JaxMARL lives in a sibling checkout (the repo installs it editable; fall back
# to a direct path so the package works even if the editable finder breaks).
try:
    import jaxmarl  # noqa: F401
except ModuleNotFoundError:
    _JM = os.path.join(os.path.dirname(REPO), "JaxMARL")
    if os.path.isdir(os.path.join(_JM, "jaxmarl")):
        sys.path.insert(0, _JM)
