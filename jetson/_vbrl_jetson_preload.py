"""Make the Jetson wheels' external CUDA and math libraries loadable.

`jetson/setup.sh` copies this into the venv's site-packages alongside a
`zzz_vbrl_jetson_preload.pth` that imports it, so it runs at interpreter
startup, before any `import torch`.

The .pth indirection is deliberate. The obvious home for this is
`sitecustomize.py`, but Ubuntu ships `/usr/lib/python3.12/sitecustomize.py`,
which precedes site-packages on sys.path and shadows any copy placed there --
silently, so torch just keeps failing to import. A uniquely named module cannot
be shadowed that way.

The CUDA 13 aarch64 torch wheels link NVPL (the ARM BLAS/LAPACK behind
libtorch_cpu.so), cuDSS, and cuDNN without bundling them. uv installs all three,
but into `nvpl/lib` and `nvidia/*/lib` under site-packages, which the dynamic
loader does not search -- so libtorch fails to load with
"libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file".

Loading them here with RTLD_GLOBAL is what fixes that: once an object is in the
global symbol namespace, the loader satisfies libtorch's DT_NEEDED entries from
it by soname rather than searching the filesystem. This is the same mechanism
torch's own `_preload_cuda_deps` uses for its bundled-CUDA builds. Each library
below has `$ORIGIN` on its RPATH and pulls its own dependencies out of its own
directory, so preloading the top-level ones is sufficient.

Two alternatives were tried and rejected. Symlinking into `torch/lib` breaks
whenever uv reinstalls a package: the payload disappears, the links dangle, and
the loader silently falls through to JetPack's system cuDNN 9.12 -- which torch
2.9.1 then rejects, having been compiled against 9.13. Setting LD_LIBRARY_PATH
only works in shells that remember to set it. This file touches no package
directory, so reinstalls cannot invalidate it.
"""

from __future__ import annotations

import ctypes
import glob
import os
import platform
import sys

# Ordered: NVPL BLAS underpins LAPACK, and cuDNN goes last since torch queries
# its version once loaded. Patterns rather than fixed paths because NVIDIA has
# moved these between per-component (nvidia/cudnn/lib) and shared
# (nvidia/cu13/lib) layouts across releases.
_PRELOAD_PATTERNS = (
  "nvpl/lib/libnvpl_blas_lp64_gomp.so.*",
  "nvpl/lib/libnvpl_lapack_lp64_gomp.so.*",
  "nvidia/*/lib/libcudss.so.*",
  "nvidia/*/lib/libcudnn.so.*",
)


def _preload_jetson_libraries() -> None:
  # x86_64 hosts install the cu128 wheels, which bundle everything they need.
  if platform.machine() != "aarch64" or sys.platform != "linux":
    return

  site_packages = os.path.dirname(os.path.abspath(__file__))
  missing: list[str] = []

  for pattern in _PRELOAD_PATTERNS:
    matches = sorted(glob.glob(os.path.join(site_packages, pattern)))
    if not matches:
      missing.append(pattern)
      continue
    try:
      ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
      missing.append(f"{pattern} ({exc})")

  # Never raise: a broken preload must not make the interpreter unusable, and
  # torch's own error is clearer than anything this file could say. Warn so the
  # cause is visible, and let `jetson/verify.py` be the gate that fails.
  if missing:
    print(
      "vbrl jetson preload: could not load "
      + ", ".join(missing)
      + "\n  torch will likely fail to import or fall back to the system cuDNN."
      + "\n  Run: bash jetson/setup.sh",
      file=sys.stderr,
    )


_preload_jetson_libraries()
