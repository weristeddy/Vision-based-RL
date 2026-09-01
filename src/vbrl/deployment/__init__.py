"""Run a trained policy on real Trossen hardware.

The task ID still owns the architecture, observations, and action scaling; this
package supplies only what the simulator provided for free -- sensor reads, a
lift target, forward kinematics, and a clock.
"""

from __future__ import annotations

from vbrl.deployment.config import DeploymentConfig, SafetyLimits, load_config

__all__ = ["DeploymentConfig", "SafetyLimits", "load_config"]
