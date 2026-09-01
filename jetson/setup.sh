#!/usr/bin/env bash
# Prepare a Jetson AGX Thor for deploying VBRL policies on real hardware.
#
# Thor is a third deployment target beside the clusters and a development
# workstation, and it is the only one that cannot use the cu128 wheels: its GPU
# is sm_110, which those wheels do not carry, so `torch.cuda.is_available()`
# returns True while every kernel launch fails with "no kernel image is
# available for execution on the device". pyproject.toml routes aarch64 to the
# CUDA 13 wheels on the jetson-ai-lab index, which do carry sm_110.
#
# Those wheels link JetPack's system CUDA instead of bundling it, and expect
# NVPL (the ARM BLAS/LAPACK libtorch_cpu.so links against), cuDSS, and cuDNN to
# be found by the dynamic loader. uv installs all three, but into
# site-packages/nvpl/lib and site-packages/nvidia/*/lib, which the loader does
# not search -- so libtorch cannot load until something puts them in reach.
# jetson/sitecustomize.py does that by preloading them at interpreter startup,
# and installing it is the one step uv cannot express, so it lives here.
#
#   bash jetson/setup.sh              # sync, install the preload, verify
#   bash jetson/setup.sh --no-sync    # install the preload and verify only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_SYNC=1
[[ "${1:-}" == "--no-sync" ]] && RUN_SYNC=0

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is for the aarch64 Jetson deployment host; found $(uname -m)." >&2
  echo "On x86_64 the plain 'uv sync' is the whole setup." >&2
  exit 1
fi

# uv installs to ~/.local/bin, which is not always on a non-login shell's PATH.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "uv not found. Install it with:" >&2
  echo "  wget -qO- https://astral.sh/uv/0.12.5/install.sh | sh" >&2
  exit 1
fi

if [[ "$RUN_SYNC" == "1" ]]; then
  echo "==> uv sync"
  uv sync
fi

SITE_PACKAGES="$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TORCH_LIB="$SITE_PACKAGES/torch/lib"
if [[ ! -d "$TORCH_LIB" ]]; then
  echo "No torch/lib under $SITE_PACKAGES -- run without --no-sync first." >&2
  exit 1
fi

echo "==> installing the library preload into site-packages"
# An earlier version of this script symlinked the libraries into torch/lib
# instead. Those links dangle as soon as uv reinstalls a package, and a dangling
# link sends the loader to JetPack's system cuDNN 9.12, which torch 2.9.1
# rejects -- so clear any left behind by a previous run.
stale="$(find "$TORCH_LIB" -maxdepth 1 -type l \
  \( -name 'libnvpl_*' -o -name 'libcudss*' -o -name 'libcudnn*' \) -delete -print 2>/dev/null | wc -l)"
[[ "$stale" -gt 0 ]] && echo "    removed $stale stale symlink(s) from torch/lib"

# The module does the loading; the .pth is what gets it executed at startup.
# "zzz_" so it runs after the .pth files that set up the venv itself.
cp "$REPO_ROOT/jetson/_vbrl_jetson_preload.py" "$SITE_PACKAGES/_vbrl_jetson_preload.py"
echo "import _vbrl_jetson_preload" > "$SITE_PACKAGES/zzz_vbrl_jetson_preload.pth"
rm -f "$SITE_PACKAGES/sitecustomize.py"  # from an earlier, shadowed approach
echo "    $SITE_PACKAGES/_vbrl_jetson_preload.py + zzz_vbrl_jetson_preload.pth"

echo "==> verifying the GPU stack"
uv run --no-sync python "$REPO_ROOT/jetson/verify.py"
