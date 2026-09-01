"""Check that the Jetson deployment stack can actually execute GPU kernels.

`torch.cuda.is_available()` is not the test that matters on Thor. A wheel built
without sm_110 still reports an available CUDA device, and only fails when a
kernel is launched -- so this exercises the operations a policy rollout really
performs (matmul, convolution, half precision) and fails loudly if any of them
cannot run.

Run via `bash jetson/setup.sh`, or directly with `uv run --no-sync python
jetson/verify.py`.
"""

from __future__ import annotations

import sys

THOR_CAPABILITY = (11, 0)

failures: list[str] = []


def check(label: str, fn) -> None:
  try:
    print(f"  ok   {label}: {fn()}")
  except Exception as exc:  # noqa: BLE001 - report every failure, never abort early
    failures.append(label)
    print(f"  FAIL {label}: {type(exc).__name__}: {str(exc)[:200]}")


def main() -> int:
  import torch

  print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
  print(f"compiled architectures: {torch.cuda.get_arch_list()}")

  if not torch.cuda.is_available():
    print("FAIL no CUDA device visible", file=sys.stderr)
    return 1

  capability = torch.cuda.get_device_capability(0)
  name = torch.cuda.get_device_name(0)
  memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
  print(f"device: {name}, sm_{capability[0]}{capability[1]}, {memory_gb:.0f} GB")

  # The failure this whole file exists to catch, named explicitly.
  arch = f"sm_{capability[0]}{capability[1]}"
  if arch not in torch.cuda.get_arch_list():
    print(
      f"FAIL torch carries no {arch} kernels, so every launch on this device "
      f"will fail. Install the CUDA 13 aarch64 wheels: bash jetson/setup.sh",
      file=sys.stderr,
    )
    return 1

  print("kernels:")
  a = torch.randn(1024, 1024, device="cuda")
  check("fp32 matmul", lambda: float((a @ a).sum()))

  half = torch.randn(512, 512, device="cuda", dtype=torch.float16)
  check("fp16 matmul", lambda: float((half @ half).sum()))

  def bf16_autocast():
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return (a @ a).dtype

  check("bf16 autocast", bf16_autocast)

  def conv():
    import torch.nn as nn

    layer = nn.Conv2d(3, 32, 3).cuda()
    return tuple(layer(torch.randn(2, 3, 224, 224, device="cuda")).shape)

  check("conv2d", conv)
  check("cudnn", lambda: torch.backends.cudnn.version())
  # NVPL supplies this on CPU; a bad link shows up here rather than at import.
  check("cpu lapack (nvpl)", lambda: tuple(torch.linalg.inv(torch.randn(64, 64)).shape))

  def torchvision_cuda_op():
    import torchvision
    from torchvision.ops import nms

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    return f"{torchvision.__version__}, nms -> {nms(boxes, scores, 0.5).tolist()}"

  check("torchvision cuda op", torchvision_cuda_op)

  print("repo stack:")
  check("mujoco", lambda: __import__("mujoco").__version__)
  check("mjlab", lambda: __import__("mjlab").__name__)

  def task_registry():
    from vbrl.tasks import vbrl_task_ids

    return f"{len(vbrl_task_ids())} task ids"

  check("vbrl task registry", task_registry)

  if failures:
    print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}", file=sys.stderr)
    return 1
  print("\nAll checks passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
