from __future__ import annotations

import os
import select
import sys
import termios
import tty
from typing import Any

import numpy as np

from vbrl.deployment.config import GOAL_RANGE

STEP = 0.02
# Left is +y, not -y: the base frame is right-handed with x forward and z up,
# so +y points to the robot's own left. x has no key and holds its manifest
# value.
AXES = {
  "up": (2, +STEP),
  "down": (2, -STEP),
  "left": (1, +STEP),
  "right": (1, -STEP),
}
BOUNDS = np.array([GOAL_RANGE[axis] for axis in "xyz"])
# Terminals send arrows as CSI or as SS3, depending on the terminal and on
# whatever ran before us, so both are read. Every sequence is three bytes.
SEQUENCES = {
  "\x1b[A": "up",
  "\x1b[B": "down",
  "\x1b[C": "right",
  "\x1b[D": "left",
  "\x1bOA": "up",
  "\x1bOB": "down",
  "\x1bOC": "right",
  "\x1bOD": "left",
}
HELP = (
  "  up    goal z +0.02 m       left   goal y +0.02 m\n"
  "  down  goal z -0.02 m       right  goal y -0.02 m\n"
  "  ctrl-c  stop"
)


def decode(buffer: str) -> tuple[tuple[str, ...], str]:
  """Arrow keys in ``buffer``, and the tail that may still complete one."""
  keys = []
  while buffer:
    if len(buffer) < 3:
      if any(sequence.startswith(buffer) for sequence in SEQUENCES):
        break
      buffer = buffer[1:]
      continue
    key = SEQUENCES.get(buffer[:3])
    if key is None:
      buffer = buffer[1:]
      continue
    keys.append(key)
    buffer = buffer[3:]
  return tuple(keys), buffer


def nudge_goal(goal: Any, keys: Any) -> tuple[Any, tuple[str, ...]]:
  """Apply keypresses to ``goal``, clamped to the range the policy trained on.

  Every press counts, so N presses move N steps. A press that would leave the
  range is clamped to its edge and reported, since one that silently does
  nothing looks like a dropped keystroke.
  """
  goal = np.array(goal, dtype=np.float64)
  refused = []
  for key in keys:
    axis, step = AXES[key]
    low, high = BOUNDS[axis]
    # Rounded: repeated additions of 0.02 drift, and 0.4000000000000001 would
    # put the top of the range out of reach.
    wanted = round(goal[axis] + step, 6)
    clamped = min(high, max(low, wanted))
    if clamped != wanted:
      refused.append(
        f"goal {'xyz'[axis]} held at {clamped:.3f}: {wanted:.3f} is outside "
        f"the range the policy trained on, [{low}, {high}]"
      )
    goal[axis] = clamped
  return goal, tuple(refused)


class ArrowKeys:
  """Non-blocking arrow-key reads from the terminal, as a context manager.

  cbreak rather than raw mode: raw clears ISIG, and ctrl-c must always be able
  to stop a moving arm. The mode is re-asserted on every read because the arm
  and camera drivers start after this one does and put the terminal back into
  canonical mode, where the keys echo instead of arriving. With no terminal on
  stdin this disables itself.
  """

  def __init__(self, stream: Any = None) -> None:
    self._stream = sys.stdin if stream is None else stream
    self._buffer = ""
    self._terminal: int | None = None
    self._original: Any = None
    self.reclaims = 0

  @property
  def enabled(self) -> bool:
    return self._terminal is not None

  def __enter__(self) -> ArrowKeys:
    try:
      terminal = self._stream.fileno()
      if not os.isatty(terminal):
        return self
      self._original = termios.tcgetattr(terminal)
    except (AttributeError, OSError, ValueError, termios.error):
      return self
    self._terminal = terminal
    tty.setcbreak(terminal)
    return self

  def __exit__(self, *exception: Any) -> None:
    if self._terminal is not None:
      termios.tcsetattr(self._terminal, termios.TCSADRAIN, self._original)
      self._terminal = None

  def pressed(self) -> tuple[str, ...]:
    """Every arrow key typed since the last call. Never blocks."""
    if self._terminal is None:
      return ()
    if termios.tcgetattr(self._terminal)[3] & (termios.ICANON | termios.ECHO):
      # TCSANOW, not setcbreak's default TCSAFLUSH, which would discard keys
      # typed while the terminal was out of our hands.
      tty.setcbreak(self._terminal, termios.TCSANOW)
      self.reclaims += 1
    while select.select([self._terminal], [], [], 0)[0]:
      chunk = os.read(self._terminal, 1024)
      if not chunk:
        break
      self._buffer += chunk.decode("utf-8", "replace")
    keys, self._buffer = decode(self._buffer)
    return keys


__all__ = ["AXES", "HELP", "STEP", "ArrowKeys", "decode", "nudge_goal"]
