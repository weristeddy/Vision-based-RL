"""Combine sharded probe outputs into one file per figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

P = Path("artifacts/probe")


def merge_json(pattern, target, keys):
  parts = sorted(P.glob(pattern))
  if not parts:
    print(f"  no shards for {target}")
    return
  merged = json.loads(parts[0].read_text())
  for f in parts[1:]:
    other = json.loads(f.read_text())
    for key in keys:
      merged.setdefault(key, {}).update(other.get(key, {}))
  (P / target).write_text(json.dumps(merged))
  n = len(merged.get(keys[0], {}))
  print(f"  {len(parts)} shards -> {target} ({n} architectures)")


def merge_npz(pattern, target):
  parts = sorted(P.glob(pattern))
  if not parts:
    print(f"  no shards for {target}")
    return
  store = {}
  for f in parts:
    store.update(dict(np.load(f)))
  np.savez_compressed(P / target, **store)
  n = len({k.split("/")[0] for k in store})
  print(f"  {len(parts)} shards -> {target} ({n} architectures)")


if __name__ == "__main__":
  generation = sys.argv[1] if len(sys.argv) > 1 else "SlowGoal"
  print("merging:")
  merge_json("scaling_shard[0-9].json", "scaling.json", ("yaw", "xy"))
  merge_json(f"conditions_{generation}_shard[0-9].json",
             f"conditions_{generation}.json", ("conditions",))
  merge_npz(f"rollout_{generation}_shard[0-9].npz", f"rollout_{generation}.npz")
