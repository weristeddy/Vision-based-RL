#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${VBRL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTAINER_REPO="/workspace/vision-based-rl"
MODEL_ROOT="${VBRL_MODEL_ROOT:-/opt/vbrl-models}"
PROFILE="testing"
USE_FALLBACK="0"
ALLOCATED="0"

usage() {
  command cat <<'EOF'
Usage:
  bash cluster/evaluate.sh --profile testing -- \
    configs/evaluation/ood_4096.yaml

Launcher options (before --):
  --profile NAME       a100, b200, or testing (default: testing)
  --fallback           use the selected profile's fallback partition

All arguments after -- are passed unchanged to vbrl.scripts.evaluate.
EOF
}

EVALUATION_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    --fallback)
      USE_FALLBACK="1"
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
      EVALUATION_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown launcher option: $1 (put evaluation options after --)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#EVALUATION_ARGS[@]} -eq 0 ]]; then
  echo "Evaluation arguments are required after --" >&2
  usage >&2
  exit 2
fi

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
    --job-name=vbrl-evaluate
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
  CHILD_ARGS=(--allocated --profile "${PROFILE}")
  if [[ "${USE_FALLBACK}" == "1" ]]; then
    CHILD_ARGS+=(--fallback)
  fi
  CHILD_ARGS+=(-- "${EVALUATION_ARGS[@]}")
  echo "Requesting one GPU from ${SELECTED_PARTITION} for evaluation..."
  exec srun "${SRUN_ARGS[@]}" bash "${SCRIPT_DIR}/evaluate.sh" "${CHILD_ARGS[@]}"
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
RUNTIME_DIR="${LOCAL_JOB_DIR:-/tmp/vbrl-evaluate-${SLURM_JOB_ID:-manual}}"
WARP_DIR="${RUNTIME_DIR}/warp-evaluation"
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
  bash -lc 'cd /workspace/vision-based-rl; exec python -m vbrl.scripts.evaluate "$@"' \
  bash "${EVALUATION_ARGS[@]}" --device cuda:0
