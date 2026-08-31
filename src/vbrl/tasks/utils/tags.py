"""How a registered task ID becomes its W&B tag."""

from __future__ import annotations


TASK_ID_PREFIX = "Mjlab-"
# W&B rejects tags longer than this, and the ID scheme already reaches 68
# characters (task + variant + encoder + adapter + robot).
WANDB_TAG_MAX_LENGTH = 64


def wandb_task_tag(task_id: str) -> str:
  """Return the discovery tag for ``task_id``.

  The shared ``Mjlab-`` prefix is dropped: it carries no information (every ID
  has it) and costs six of W&B's 64 characters, which the longest IDs need.
  """
  tag = task_id.removeprefix(TASK_ID_PREFIX)
  if len(tag) > WANDB_TAG_MAX_LENGTH:
    raise ValueError(
      f"W&B tag {tag!r} is {len(tag)} characters; the limit is "
      f"{WANDB_TAG_MAX_LENGTH}. Shorten the task, variant, or robot token."
    )
  return tag


__all__ = ["TASK_ID_PREFIX", "WANDB_TAG_MAX_LENGTH", "wandb_task_tag"]
