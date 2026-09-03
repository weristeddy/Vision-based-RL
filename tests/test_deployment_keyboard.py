"""Arrow-key decoding, goal nudging, and the terminal handling in between."""

from __future__ import annotations

import os
import pty
import termios
import time

import numpy as np
import pytest


pytest.importorskip("mjlab")

from vbrl.deployment.config import GOAL_RANGE  # noqa: E402
from vbrl.deployment.keyboard import STEP, ArrowKeys, decode, nudge_goal  # noqa: E402


CSI = {"up": "\x1b[A", "down": "\x1b[B", "right": "\x1b[C", "left": "\x1b[D"}
SS3 = {"up": "\x1bOA", "down": "\x1bOB", "right": "\x1bOC", "left": "\x1bOD"}
CEILING = GOAL_RANGE["z"][1]
FLOOR = GOAL_RANGE["z"][0]


@pytest.mark.parametrize("sequences", [CSI, SS3])
def test_both_terminal_encodings_decode(sequences: dict[str, str]) -> None:
  for name, sequence in sequences.items():
    assert decode(sequence) == ((name,), "")


def test_several_presses_in_one_read_all_count() -> None:
  assert decode(CSI["up"] * 3 + CSI["left"]) == (("up", "up", "up", "left"), "")


def test_a_sequence_split_across_reads_survives() -> None:
  """A read can land mid-sequence, so the tail is carried to the next one."""
  keys, tail = decode("\x1b")
  assert (keys, tail) == ((), "\x1b")
  keys, tail = decode(tail + "[")
  assert (keys, tail) == ((), "\x1b[")
  assert decode(tail + "A") == (("up",), "")


def test_other_keys_do_not_stall_the_parse() -> None:
  assert decode("qwe" + CSI["down"] + "z") == (("down",), "")


def test_one_press_moves_one_step_on_its_own_axis() -> None:
  start = np.array([0.35, 0.0, 0.30])
  for key, axis, sign in (
    ("up", 2, +1),
    ("down", 2, -1),
    ("left", 1, +1),
    ("right", 1, -1),
  ):
    moved, refused = nudge_goal(start, [key])
    expected = start.copy()
    expected[axis] += sign * STEP
    assert refused == ()
    np.testing.assert_allclose(moved, expected)


def test_x_is_never_touched() -> None:
  moved, _ = nudge_goal([0.35, 0.0, 0.30], ["up", "down", "left", "right"] * 5)
  assert moved[0] == 0.35


def test_presses_accumulate() -> None:
  moved, refused = nudge_goal([0.35, 0.0, 0.24], ["up"] * 4)
  assert refused == ()
  assert moved[2] == pytest.approx(0.24 + 4 * STEP)


def test_repeated_presses_reach_the_edge_of_the_range() -> None:
  """Repeated additions of 0.02 drift, which once put the ceiling out of reach."""
  goal = np.array([0.35, 0.0, FLOOR])
  for _ in range(round((CEILING - FLOOR) / STEP)):
    goal, refused = nudge_goal(goal, ["up"])
    assert refused == (), f"stalled early at z={goal[2]!r}"
  assert goal[2] == CEILING


def test_a_press_past_the_edge_is_clamped_and_reported() -> None:
  goal, refused = nudge_goal([0.35, 0.0, CEILING - 0.005], ["up"])
  assert goal[2] == CEILING
  assert len(refused) == 1

  goal, refused = nudge_goal([0.35, 0.0, CEILING], ["up"])
  assert goal[2] == CEILING
  assert "outside the range the policy trained on" in refused[0]

  goal, refused = nudge_goal([0.35, 0.0, FLOOR], ["down"])
  assert goal[2] == FLOOR and len(refused) == 1


def test_a_clamped_press_does_not_block_the_others() -> None:
  goal, refused = nudge_goal([0.35, 0.0, CEILING], ["up", "left"])
  assert len(refused) == 1
  assert goal[1] == pytest.approx(+STEP)


def test_without_a_terminal_it_disables_itself() -> None:
  class NotATerminal:
    def fileno(self) -> int:
      raise OSError("no fd")

  with ArrowKeys(NotATerminal()) as keys:
    assert not keys.enabled
    assert keys.pressed() == ()


def test_a_regular_file_is_not_a_terminal(tmp_path) -> None:
  path = tmp_path / "stdin"
  path.write_text(CSI["up"])
  with path.open() as handle, ArrowKeys(handle) as keys:
    assert not keys.enabled
    assert keys.pressed() == ()


def test_on_a_terminal_it_reads_keys_and_hands_it_back() -> None:
  master, slave = pty.openpty()
  original = termios.tcgetattr(slave)

  with ArrowKeys(os.fdopen(slave, "r")) as keys:
    assert keys.enabled
    lflag = termios.tcgetattr(slave)[3]
    assert not lflag & termios.ICANON
    assert not lflag & termios.ECHO
    # cbreak, not raw: with ISIG cleared, ctrl-c would arrive as a byte here
    # instead of stopping the arm.
    assert lflag & termios.ISIG

    os.write(master, b"\x1b[A")
    time.sleep(0.05)
    assert keys.pressed() == ("up",)

  assert termios.tcgetattr(slave) == original


def test_it_takes_the_terminal_back_when_something_resets_it() -> None:
  """What broke on hardware: the drivers left the tty in canonical mode, and
  the arrows echoed as ^[[D into the step log instead of reaching the policy."""
  master, slave = pty.openpty()

  with ArrowKeys(os.fdopen(slave, "r")) as keys:
    assert keys.reclaims == 0

    canonical = termios.tcgetattr(slave)
    canonical[3] |= termios.ICANON | termios.ECHO
    termios.tcsetattr(slave, termios.TCSANOW, canonical)

    os.write(master, b"\x1b[B\x1b[D")
    time.sleep(0.05)
    assert keys.pressed() == ("down", "left"), "keys typed meanwhile must survive"
    assert keys.reclaims == 1

    os.write(master, b"\x1b[C")
    time.sleep(0.05)
    assert keys.pressed() == ("right",)
    assert keys.reclaims == 1, "a healthy terminal must not be reclaimed again"


def test_the_policy_goal_can_be_moved_mid_run() -> None:
  from vbrl.deployment.policy import Policy

  policy = Policy.__new__(Policy)
  policy._goal = np.array([0.35, 0.0, 0.30])

  policy.goal = [0.4, 0.1, 0.25]
  assert policy.goal.tolist() == [0.4, 0.1, 0.25]

  policy.goal[0] = 99.0
  assert policy.goal[0] == 0.4, "the getter returns a copy"

  with pytest.raises(ValueError, match="goal must be 3 values"):
    policy.goal = [0.4, 0.1]
