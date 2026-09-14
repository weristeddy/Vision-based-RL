"""Push-T rewards."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.utils.lab_api.math import wrap_to_pi

from ..geometry import FOOTPRINT_PARTS, yaw_from_quat
from .commands import push_t_command


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")
_DISTANCE_SCALE = 5.0
_MAX_REWARD = 3.0
# The far end of each box's long axis, in the object's own frame: the two ends of
# the crossbar and the two of the stem. Derived from the same footprint the
# overlap rasteriser scores, so the reward cannot drift from the shape.
KEYPOINTS_XY = tuple(
  (
    part.center_xy[0] + sign * part.half_extents_xy[0] * (axis == 0),
    part.center_xy[1] + sign * part.half_extents_xy[1] * (axis == 1),
  )
  for part in FOOTPRINT_PARTS
  for axis in ((0,) if part.half_extents_xy[0] >= part.half_extents_xy[1] else (1,))
  for sign in (-1.0, +1.0)
)


def maniskill_dense_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Normalized ManiSkill3 Push-T dense reward."""
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  )
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  weight = float(command.cfg.orientation_weight)
  reward = (
    weight * ((torch.cos(yaw_error) + 1.0) / 2.0).square()
    + (1.0 - weight)
    * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt(
      (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
    )
    / 20.0
  )
  reward = torch.where(
    command.get_at_goal(),
    torch.full_like(reward, _MAX_REWARD),
    reward,
  )
  return reward / _MAX_REWARD


def quadratic_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """ManiSkill's dense reward with the orientation term's dead zone removed.

  Identical to :func:`maniskill_dense_reward` in every term, weight and cap; the
  only change is the shape of the orientation factor. Measured on the term as
  written, ManiSkill's gradient is 0.250 at 90 degrees and **0.000** at 180: any
  function of ``cos e`` is flat there. Under a goal-yaw curriculum that never
  matters, because episodes start at zero error; without one the initial error is
  uniform on [0, pi], so roughly a quarter of episodes begin past 138 degrees
  with almost no orientation signal at all.

  ``1 - (|e| / pi)**2`` inverts the profile: 0.000 at 0 degrees, so a clumsy
  first contact is not punished, rising to **0.318** at 180 where ManiSkill has
  none. Same [0, 0.5] range, so normalisation and the sparse at-goal bonus are
  untouched.
  """
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  ).abs()
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  weight = float(command.cfg.orientation_weight)
  reward = (
    weight * (1.0 - (yaw_error / torch.pi).square())
    + (1.0 - weight)
    * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt(
      (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
    )
    / 20.0
  )
  reward = torch.where(
    command.get_at_goal(),
    torch.full_like(reward, _MAX_REWARD),
    reward,
  )
  return reward / _MAX_REWARD


def vertical_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_scale: float = 10.0,
) -> torch.Tensor:
  """Force-weighted vertical contact; clean side pushes score zero."""
  if force_scale <= 0.0:
    raise ValueError("force_scale must be positive.")
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.force is None or data.normal is None:
    raise RuntimeError(
      f"Contact sensor {sensor_name!r} requires found, force, and normal."
    )
  verticality = torch.abs(data.normal[..., 2]).clamp(0.0, 1.0)
  force = torch.linalg.vector_norm(data.force, dim=-1)
  bounded_force = torch.tanh(
    torch.nan_to_num(force, nan=0.0, posinf=force_scale) / force_scale
  )
  return torch.amax(
    torch.where(
      data.found > 0,
      verticality * bounded_force,
      torch.zeros_like(verticality),
    ),
    dim=-1,
  )


def top_contact_share(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  verticality_threshold: float = 0.7,
) -> torch.Tensor:
  """1.0 while the gripper presses a horizontal face of the object, else 0.0.

  The behaviour measure for the drag-versus-push failure: a policy that scrapes
  the top of the T sits near 1.0 whenever it is touching, a clean side push near
  0.0, and not touching is also 0.0. Logged as a metric rather than shaped --
  ``vertical_contact_force`` already carries the gradient, and its weight moves
  over training, which makes the reward channel unreadable as behaviour.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.normal is None:
    raise RuntimeError(f"Contact sensor {sensor_name!r} requires found and normal.")
  vertical = data.normal[..., 2].abs() > verticality_threshold
  return ((data.found > 0) & vertical).any(dim=-1).float()


def over_object_exclusion(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  object_name: str,
  ceiling: float,
  radius: float,
) -> torch.Tensor:
  """Forbid the fingertips the volume directly above the object.

  Push-T is planar, and every published version enforces that in the action
  space -- IBC, Diffusion Policy and ManiSkill3 all fix the pusher's z, so
  descending onto the object is impossible rather than discouraged. With six
  joints the policy finds that strategy: measured, 33% of its contacts are on the
  top face, because arriving from above means never having to position on the
  correct *side* of the T first.

  This is a no-fly cylinder rather than a height penalty, and the difference is
  the direction of the gradient. A ceiling on absolute height pushes *down*, and
  over the object down means into it -- run avv1us6e took top-face contact from
  0.105 to 0.331 because the fingertip resting on a 24 mm object violated a
  24 mm ceiling by construction and the cheapest compliance was to press. Here
  the value grows with how far *inside* the footprint the fingertip is, so the
  gradient points sideways and the only way out is to move aside.

  Descending cannot relieve it either: to be inside the footprint below the
  object's top face is to be inside the object, which the contact model forbids.
  So the exclusion has no downward escape, which is the property every previous
  attempt lacked.

  Returns 0 when below ``ceiling`` or outside ``radius``, rising to 1 directly
  above the object's centre. Both bounds come from the object's own geometry.
  """
  if ceiling <= 0.0 or radius <= 0.0:
    raise ValueError("over_object_exclusion needs ceiling > 0 and radius > 0.")
  asset: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  pos = asset.data.geom_pos_w[:, asset_cfg.geom_ids, :]
  planar = torch.linalg.vector_norm(
    pos[..., :2] - obj.data.root_link_pos_w[:, None, :2], dim=-1
  )
  depth = (1.0 - planar / radius).clamp_min(0.0)
  return (depth * (pos[..., 2] > ceiling)).amax(dim=-1)


def action_path_length_l1(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize total commanded travel: the L1 norm of the raw policy action.

  L1 rather than L2 because the objective is total *path*, not speed. Under L1
  one 0.1 rad step costs exactly what two 0.05 rad steps cost, so splitting a
  motion is never cheaper and a forward/backward correction cycle -- which
  doubles the path -- costs double. L2 would make many tiny moves cheaper than
  one decisive move, which is the opposite of what is wanted.

  Operates on the raw policy output, as MJLab's own ``action_rate_l2`` and
  ``action_acc_l2`` do. The action term scales every arm joint by the same
  0.1 rad and clips the delta, so this is proportional to commanded radians.
  """
  return torch.sum(torch.abs(env.action_manager.action), dim=1)


def at_goal_action_l1(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """L1 of the action, charged only while the object is already at its goal.

  ManiSkill's dense reward replaces itself with the sparse maximum once overlap
  crosses the threshold -- position, orientation and the tcp term all go -- so
  from that moment the task gives the arm no gradient whatsoever. Nothing
  rewards holding still, staying near the object, or backing off, and relative
  joint-position control integrates the action, so even a zero-mean policy
  random-walks rather than holding a pose. Measured on the trained VisualSlow
  policy: after success the end-effector still travels 6.65 mm per step, 33 cm/s
  at the 50 Hz control rate, while the object moves 0.48 mm.

  Charging motion in that region alone is free in a way no other penalty here
  is: the task reward is *constant* there, so this cannot trade against task
  performance the way a penalty applied during the push does.
  """
  command = push_t_command(env, command_name)
  action = torch.sum(torch.abs(env.action_manager.action), dim=1)
  return action * command.get_at_goal().to(action.dtype)


def max_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Peak contact-force magnitude on one sensor, in newtons.

  Reads the substep history when the sensor keeps one, so a spike that resolves
  inside a policy step is not missed -- the same quantity MJLab's
  ``illegal_contact`` thresholds.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  force = data.force_history if data.force_history is not None else data.force
  if force is None:
    raise RuntimeError(f"Contact sensor {sensor_name!r} requires the force field.")
  magnitude = torch.linalg.vector_norm(force, dim=-1)
  return torch.nan_to_num(magnitude, nan=0.0).flatten(1).amax(dim=-1)


def contact_force_barrier(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  onset: float,
  scale: float,
  cap: float,
) -> torch.Tensor:
  """Exponential barrier on contact force: exactly zero below ``onset``.

  ``expm1((F - onset) / scale)``, for "stay under onset newtons, and the cost of
  exceeding it climbs fast". Measured on a trained policy's *productive* contacts
  -- the steps where the T actually moves -- force runs p10 0.23 N, p50 4.45,
  p90 21.65, max 77.8, while the T slides at 0.23 N. So the task needs about 1 N
  and the policy uses twenty times that.

  **Capped, and that is not optional.** Left unbounded this returns 5.2e16 at the
  77.8 N the policy already produces. PPO regresses a value function on returns
  containing that, so the critic loss explodes and one contact step dominates
  every advantage in the batch -- the run dies immediately rather than degrading.
  The exponent is clamped rather than the output so the large exp is never
  evaluated at all, which keeps the gradient finite instead of merely the value.

  The cap does reintroduce a flat region, which is the flaw that made
  ``vertical_contact_force`` blind. The difference is where it starts: tanh(F/10)
  goes flat at 20 N, inside normal operation, while this goes flat a few newtons
  above the target, where the only remaining question is how much too much.
  """
  if onset < 0.0 or scale <= 0.0 or cap <= 0.0:
    raise ValueError("contact_force_barrier needs onset >= 0, scale > 0, cap > 0.")
  excess = (max_contact_force(env, sensor_name) - onset).clamp_min(0.0)
  limit = math.log1p(cap)
  return torch.expm1((excess / scale).clamp_max(limit))


def contact_force_hinge(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  onset: float,
  scale: float,
) -> torch.Tensor:
  """Quadratic hinge on contact force: exactly zero below ``onset`` newtons.

  Replaces the binary ``illegal_contact`` reward, which paid the same penalty at
  10 N as at 100 N and nothing at all below, so it offered no gradient toward
  gentler contact. Quadratic above the hinge -- the shape MJLab's
  ``joint_velocity_hinge_penalty`` uses for velocity -- so a graze stays
  negligible while a genuinely unsafe press grows fast.
  """
  if onset < 0.0 or scale <= 0.0:
    raise ValueError("contact_force_hinge needs onset >= 0 and scale > 0.")
  excess = (max_contact_force(env, sensor_name) - onset).clamp_min(0.0)
  return (excess / scale).square()


def linear_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """ManiSkill's dense reward with an orientation factor that never goes flat.

  Both existing shapes have a dead zone, at opposite ends. ManiSkill's
  ``((cos e + 1) / 2)**2`` has zero gradient at 0 *and* at pi -- 0.302 at 45
  degrees but 0.004 at 162 and 0.000 at 180 -- and
  :func:`quadratic_orientation_reward` removes the far one only by introducing a
  near one, 0.318 at 180 degrees against 0.000 at 0, which takes away the fine
  alignment the 0.90 overlap threshold is made of. It scored 2 of 15 where the
  matched ManiSkill control scored 4.

  ``1 - |e| / pi`` is flat nowhere: gradient 0.159 at every angle. That matters
  because the two terms compete. Rotating the T toward the goal also shifts it,
  and the position factor's gradient peaks near the goal, so a push that
  correctly reduces yaw error is *punished* under ManiSkill's shape wherever the
  orientation gradient has decayed -- measured at -0.00101 at 15 cm and 162
  degrees, and still negative at 22 cm. Roughly a quarter of episodes with a
  uniform goal start past 135 degrees, inside that region. Under the linear
  shape the same push pays at every distance and every angle.

  Shares the ``orientation_weight`` split with the other two shapes, so the
  weight and the shape are independent knobs, and reduces to the same
  ``[0, 0.5]`` contribution at the registered 0.5 -- normalisation, the cap
  and the sparse at-goal bonus are untouched.
  """
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  ).abs()
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  weight = float(command.cfg.orientation_weight)
  reward = (
    weight * (1.0 - yaw_error / torch.pi)
    + (1.0 - weight)
    * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt(
      (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
    )
    / 20.0
  )
  reward = torch.where(
    command.get_at_goal(), torch.full_like(reward, _MAX_REWARD), reward
  )
  return reward / _MAX_REWARD


def keypoint_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """One SE(2) term over four points on the T, replacing the weighted split.

  The other three shapes score position and orientation separately and trade
  them off with ``orientation_weight``. That split is the failure every arm in
  the orientation batch ran into: the two factors have independent gradients, so
  there is a distance at which a push that correctly reduces yaw error still
  loses reward -- measured at -0.00101 at 15 cm and 162 degrees under ManiSkill's
  shape -- and no value of the weight removes the competition, it only moves the
  crossover.

  Tracking points removes the split rather than retuning it. Each keypoint is
  carried to where the goal pose would put it and the reward is the mean of
  ``(1 - tanh(5 d))**2`` over the four distances, the same shape the position
  factor already uses, applied four times. A translation moves all four points
  and a rotation moves them in opposing directions, so both are the same
  currency and there is nothing left to weigh: the relative worth of position
  and orientation is fixed by the T's own geometry -- 16.5 cm of stem-tip travel
  for a half turn -- instead of by a hyperparameter. ``orientation_weight`` is
  therefore unused, and ``--orientation-weight`` has no effect when this shape
  is selected.

  Precedent, since it matters for how much to trust this. Two PPO papers reach
  the same construction from the same complaint:

  * *Whole-body End-Effector Pose Tracking* (arXiv:2409.16048) tracks a pose
    through three vertices of a 0.3 m cube on the end-effector -- the minimum
    that defines a pose uniquely -- scored as ``exp(-||d|| / 0.05)`` summed over
    keypoints. Its stated reason for not decomposing is the one measured here:
    separate position and orientation terms need "a fixed trade-off ... which
    may not be optimal for all workspace poses" and mean "balancing two
    quantities with different units and magnitudes, often causing training to
    collapse", where "the keypoint-based pose representation required far less
    tuning due to its unified representation". Its ablation puts keypoints ahead
    of quaternion, Euler and 6D pose representations.
  * *Iterative Keypoint Rewards* (arXiv:2502.08643) trains PPO on rewards over
    keypoints placed "at the object's extremities along its axes" -- the same
    rule used here -- from RGB-D, and handles orientation through keypoint
    positions rather than any rotation term, on prehensile *and* non-prehensile
    tasks.

  The underlying quantity is older still: averaging ``||R1 p + t1 - (R2 p +
  t2)||`` over model points is the ADD metric of Hinterstoisser et al. (ACCV
  2012).

  What is not borrowed is the kernel. ``exp(-d / 0.05)`` was measured against
  ``(1 - tanh(5 d))**2`` on this footprint and is three times *flatter* in the
  far field -- gradient at 170 degrees is 7.4e-03 of peak against 2.4e-02 -- so
  the shape stays matched to the position factor it replaces rather than copied.

  **It is not flat nowhere, and the reason is geometric rather than a choice of
  shape.** Under a pure rotation about the object's centre every keypoint
  distance is ``2 r sin(e/2)``, which is stationary at ``e = pi``, and the same
  holds whenever the position offset lies along the T's mirror axis, where the
  two crossbar terms cancel. What differs from the decomposed shapes is the
  order. Measured as a fraction of each shape's own peak gradient, with the T on
  the goal:

  ==========  ==================  ==================
  yaw error   keypoint, on goal   maniskill
  ==========  ==================  ==================
  170 deg     2.4e-02             2.0e-03
  175 deg     1.2e-02             2.5e-04
  179 deg     2.3e-03             1.9e-06
  ==========  ==================  ==================

  ManiSkill's ``cos(e/2)**4`` vanishes to third order and is three decades
  flatter by 179 degrees; this vanishes to first order, and only in that one
  alignment -- 10 cm off across the mirror axis the ratio at 179 degrees is
  0.64, no dead zone at all. An episode is off-position nearly all of the time,
  so the flat spot is reached only once the T is already placed.

  The cost of merging the terms, stated plainly: off-position the reward is no
  longer monotone in yaw alone. At a 10 cm offset it turns at 43 and 133
  degrees, so there are configurations where rotating *away* from the goal
  orientation pays. That is the correct behaviour for a joint SE(2) error --
  the T has to travel as well as turn, and the metric prices both -- but it is
  a real difference from the decomposed shapes, which are monotone in yaw
  everywhere by construction.

  The tcp term, the normalisation and the sparse at-goal bonus are untouched.
  """
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  position = obj.data.root_link_pos_w
  keypoints = torch.tensor(
    KEYPOINTS_XY, dtype=position.dtype, device=position.device
  )  # (K, 2)
  object_yaw = yaw_from_quat(obj.data.root_link_quat_w)
  placed = _place(keypoints, object_yaw, position[:, :2])
  target = _place(keypoints, command.target_yaw, command.target_pos[:, :2])
  distances = torch.linalg.vector_norm(placed - target, dim=-1)  # (N, K)
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  reward = (1.0 - torch.tanh(_DISTANCE_SCALE * distances)).square().mean(
    dim=-1
  ) + torch.sqrt(
    (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
  ) / 20.0
  reward = torch.where(
    command.get_at_goal(), torch.full_like(reward, _MAX_REWARD), reward
  )
  return reward / _MAX_REWARD


def _place(
  keypoints: torch.Tensor, yaw: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
  """Carry ``(K, 2)`` body-frame points to world under ``(N,)`` yaw, ``(N, 2)``."""
  cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
  x, y = keypoints[:, 0][None], keypoints[:, 1][None]
  return torch.stack(
    (cos * x - sin * y + position[:, :1], sin * x + cos * y + position[:, 1:2]),
    dim=-1,
  )


__all__ = [
  "KEYPOINTS_XY",
  "action_path_length_l1",
  "at_goal_action_l1",
  "contact_force_barrier",
  "contact_force_hinge",
  "over_object_exclusion",
  "keypoint_reward",
  "linear_orientation_reward",
  "maniskill_dense_reward",
  "max_contact_force",
  "quadratic_orientation_reward",
  "top_contact_share",
  "vertical_contact_force",
]
