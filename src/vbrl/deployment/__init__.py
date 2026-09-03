"""Run a trained policy on real Trossen hardware."""

from __future__ import annotations

from vbrl.deployment.config import DeploymentConfig, Motion, load_config

# `keyboard` is deliberately not re-exported here: it is runnable as
# `python -m vbrl.deployment.keyboard`, and importing it from the package
# __init__ makes runpy warn that the module was already in sys.modules.
__all__ = ["DeploymentConfig", "Motion", "load_config"]
