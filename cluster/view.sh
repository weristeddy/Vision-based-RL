#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${VBRL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTAINER_REPO="/workspace/vision-based-rl"
MODEL_ROOT="${VBRL_MODEL_ROOT:-/opt/vbrl-models}"
PROFILE="testing"
LOCAL_PORT="8080"
REMOTE_PORT="8080"
USE_FALLBACK="0"
ALLOCATED="0"
SSH_TUNNEL="0"

usage() {
  command cat <<'EOF'
Usage:
  bash cluster/view.sh --profile testing -- \
    Mjlab-PushCube-State-Trossen --agent zero

  bash cluster/view.sh --profile testing -- \
    Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen \
    --agent trained \
    --checkpoint-file ckpts/lift_cube/dinov2_vits14_local_grid7_real.pt \
    --scene wood

Launcher options (before --):
  --profile NAME       a100, b200, or testing (default: testing)
  --local-port PORT    laptop port when --ssh-tunnel is used (default: 8080)
  --port PORT          requested compute-node Viser port (default: 8080)
  --fallback           use the testing profile's ber-testing fallback
  --ssh-tunnel         print SSH forwarding instructions instead of a direct URL

All arguments after -- are passed unchanged to vbrl.scripts.play.
EOF
}

VIEWER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    --local-port)
      [[ $# -ge 2 ]] || { echo "--local-port requires a value" >&2; exit 2; }
      LOCAL_PORT="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      REMOTE_PORT="$2"
      shift 2
      ;;
    --fallback)
      USE_FALLBACK="1"
      shift
      ;;
    --ssh-tunnel)
      SSH_TUNNEL="1"
      shift
      ;;
    --allocated)
      ALLOCATED="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      VIEWER_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown launcher option: $1 (put viewer options after --)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for port_value in "${LOCAL_PORT}" "${REMOTE_PORT}"; do
  if [[ ! "${port_value}" =~ ^[0-9]+$ ]]; then
    echo "Ports must be integers between 1 and 65535; got ${port_value}" >&2
    exit 2
  fi
  port_number=$((10#${port_value}))
  if ((port_number < 1 || port_number > 65535)); then
    echo "Ports must be integers between 1 and 65535; got ${port_value}" >&2
    exit 2
  fi
done

PROFILE_FILE="${SCRIPT_DIR}/profiles/${PROFILE}.conf"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unknown cluster profile: ${PROFILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

SELECTED_PARTITION="${PARTITION:?Profile must set PARTITION}"
SELECTED_ACCOUNT="${ACCOUNT:-}"
if [[ "${USE_FALLBACK}" == "1" ]]; then
  : "${FALLBACK_PARTITION:?Profile ${PROFILE} has no fallback partition}"
  SELECTED_PARTITION="${FALLBACK_PARTITION}"
  SELECTED_ACCOUNT="${FALLBACK_ACCOUNT:-}"
fi
: "${VIEW_CPUS_PER_TASK:?Profile must set VIEW_CPUS_PER_TASK}"
: "${VIEW_MEMORY:?Profile must set VIEW_MEMORY}"
: "${VIEW_TIME_LIMIT:?Profile must set VIEW_TIME_LIMIT}"

if [[ "${ALLOCATED}" == "0" ]]; then
  SRUN_ARGS=(
    --job-name=vbrl-view
    --partition="${SELECTED_PARTITION}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${VIEW_CPUS_PER_TASK}"
    --gpus=1
    --mem="${VIEW_MEMORY}"
    --time="${VIEW_TIME_LIMIT}"
  )
  if [[ -n "${SELECTED_ACCOUNT}" ]]; then
    SRUN_ARGS+=(--account="${SELECTED_ACCOUNT}")
  fi
  CHILD_ARGS=(
    --allocated
    --profile "${PROFILE}"
    --local-port "${LOCAL_PORT}"
    --port "${REMOTE_PORT}"
  )
  if [[ "${USE_FALLBACK}" == "1" ]]; then
    CHILD_ARGS+=(--fallback)
  fi
  if [[ "${SSH_TUNNEL}" == "1" ]]; then
    CHILD_ARGS+=(--ssh-tunnel)
  fi
  CHILD_ARGS+=(-- "${VIEWER_ARGS[@]}")
  echo "Requesting one GPU from ${SELECTED_PARTITION} for Viser..."
  exec srun "${SRUN_ARGS[@]}" bash "${SCRIPT_DIR}/view.sh" "${CHILD_ARGS[@]}"
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURMD_NODENAME:-}" ]]; then
  echo "--allocated is internal and requires an active Slurm allocation." >&2
  exit 2
fi

CONTAINER_PATH="${REPO}/${CONTAINER_IMAGE:-rl.sif}"
if [[ ! -f "${CONTAINER_PATH}" ]]; then
  echo "Container not found: ${CONTAINER_PATH}" >&2
  exit 1
fi
if [[ -f /etc/slurm/local_job_dir.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/slurm/local_job_dir.sh
fi
RUNTIME_DIR="${LOCAL_JOB_DIR:-/tmp/vbrl-view-${SLURM_JOB_ID:-manual}}"
WARP_DIR="${RUNTIME_DIR}/warp-viewer"
mkdir -p "${WARP_DIR}"

if [[ "${MODEL_ROOT}" != /* ]]; then
  echo "VBRL_MODEL_ROOT must be absolute; got ${MODEL_ROOT}" >&2
  exit 2
fi
MODEL_BIND_ARGS=()
if [[ "${MODEL_ROOT}" != "/opt/vbrl-models" ]]; then
  if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "VBRL_MODEL_ROOT does not exist: ${MODEL_ROOT}" >&2
    exit 2
  fi
  MODEL_BIND_ARGS=(--bind "${MODEL_ROOT}:${MODEL_ROOT}")
fi

if [[ "${SSH_TUNNEL}" == "1" ]]; then
  : "${SSH_USER:?Profile must set SSH_USER for --ssh-tunnel}"
  : "${LOGIN_HOST:?Profile must set LOGIN_HOST for --ssh-tunnel}"
  echo "Viser will listen on ${SLURMD_NODENAME}:${REMOTE_PORT}."
  echo "Run this on your laptop:"
  echo "ssh -N \\"
  echo "  -o ExitOnForwardFailure=yes \\"
  echo "  -o ServerAliveInterval=60 \\"
  echo "  -L 127.0.0.1:${LOCAL_PORT}:${SLURMD_NODENAME}:${REMOTE_PORT} \\"
  echo "  ${SSH_USER}@${LOGIN_HOST}"
  echo "Then open http://127.0.0.1:${LOCAL_PORT}"
else
  if [[ "${SLURMD_NODENAME}" == *.* ]]; then
    DIRECT_HOST="${SLURMD_NODENAME}"
  else
    DIRECT_HOST="${SLURMD_NODENAME}${COMPUTE_HOST_SUFFIX:-}"
  fi
  echo "Viser will be available at http://${DIRECT_HOST}:${REMOTE_PORT}"
fi

PYTHON_ARGS=(
  "${VIEWER_ARGS[@]}"
  --device cuda:0
  --host 0.0.0.0
  --port "${REMOTE_PORT}"
)

exec apptainer exec --nv \
  --env "MUJOCO_GL=egl" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=${CONTAINER_REPO}/src" \
  --env "VBRL_REPO_ROOT=${CONTAINER_REPO}" \
  --env "VBRL_MODEL_ROOT=${MODEL_ROOT}" \
  --env "WARP_CACHE_PATH=${WARP_DIR}" \
  --bind "${REPO}:${CONTAINER_REPO}" \
  --bind "${RUNTIME_DIR}:${RUNTIME_DIR}" \
  "${MODEL_BIND_ARGS[@]}" \
  "${CONTAINER_PATH}" \
  bash -lc 'cd /workspace/vision-based-rl; exec python -m vbrl.scripts.play "$@"' \
  bash "${PYTHON_ARGS[@]}"
