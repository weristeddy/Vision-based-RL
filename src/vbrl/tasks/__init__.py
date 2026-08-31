"""VBRL task packages discovered through MJLab's native registry.

MJLab registers its own ``Mjlab-*`` tasks into the same flat registry, so a
prefix match cannot tell ownership. Use :func:`vbrl_task_ids`, which captures
the registration delta.
"""

from mjlab.tasks.registry import list_tasks
from mjlab.utils.lab_api.tasks.importer import import_packages


_BLACKLIST_PKGS = ["utils", ".mdp"]

_BEFORE = frozenset(list_tasks())
import_packages(__name__, _BLACKLIST_PKGS)
TASK_IDS: tuple[str, ...] = tuple(sorted(frozenset(list_tasks()) - _BEFORE))


def vbrl_task_ids() -> tuple[str, ...]:
  """Return every task ID this package registers, sorted."""
  return TASK_IDS


__all__ = ["TASK_IDS", "vbrl_task_ids"]
