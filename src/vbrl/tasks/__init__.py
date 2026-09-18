from mjlab.tasks.registry import list_tasks
from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

_BEFORE = frozenset(list_tasks())
import_packages(__name__, _BLACKLIST_PKGS)
TASK_IDS: tuple[str, ...] = tuple(sorted(frozenset(list_tasks()) - _BEFORE))


def vbrl_task_ids() -> tuple[str, ...]:
  return TASK_IDS


__all__ = ["TASK_IDS", "vbrl_task_ids"]
