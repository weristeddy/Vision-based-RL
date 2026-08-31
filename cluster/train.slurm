#!/usr/bin/env bash
#SBATCH --job-name=vbrl-train
#SBATCH --output=artifacts/slurm/%x-%j.out

set -euo pipefail

SCRIPT_DIR="${VBRL_CLUSTER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="${VBRL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTAINER_REPO="/workspace/vision-based-rl"
MODEL_ROOT="${VBRL_MODEL_ROOT:-/opt/vbrl-models}"
PROFILE="${1:?First argument must be a cluster profile name}"
shift

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

LAUNCH_MODE="train"
SWEEP_ID=""
SWEEP_COUNT="1"
if [[ "${1:-}" == "--sweep-agent" ]]; then
  LAUNCH_MODE="sweep"
  [[ $# -ge 2 ]] || {
    echo "--sweep-agent requires entity/project/sweep-id" >&2
    exit 2
  }
  SWEEP_ID="$2"
  shift 2
  if [[ "${1:-}" == "--count" ]]; then
    [[ $# -ge 2 ]] || {
      echo "--count requires a positive integer" >&2
      exit 2
    }
    SWEEP_COUNT="$2"
    shift 2
  fi
  if [[ $# -ne 0 ]]; then
    echo "Unexpected arguments after sweep-agent configuration: $*" >&2
    exit 2
  fi
  if [[ ! "${SWEEP_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    echo "--sweep-agent must use the full entity/project/sweep-id path" >&2
    exit 2
  fi
  if [[ ! "${SWEEP_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--count must be a positive integer; got ${SWEEP_COUNT}" >&2
    exit 2
  fi
fi

PROFILE_FILE="${SCRIPT_DIR}/profiles/${PROFILE}.conf"
if [[ ! -f "${PROFILE_FILE}" ]]; then
  echo "Unknown cluster profile: ${PROFILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

case "${GPU_COUNT:-}" in
  1|4) ;;
  *)
    echo "Training profiles must request one or four GPUs; got ${GPU_COUNT:-unset}" >&2
    exit 2
    ;;
esac

CONTAINER_PATH="${REPO}/${CONTAINER_IMAGE:-rl.sif}"
if [[ ! -f "${CONTAINER_PATH}" ]]; then
  echo "Container not found: ${CONTAINER_PATH}" >&2
  exit 1
fi

if [[ -f /etc/slurm/local_job_dir.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/slurm/local_job_dir.sh
fi
RUNTIME_DIR="${LOCAL_JOB_DIR:-/tmp/vbrl-${SLURM_JOB_ID:-manual}}"
TORCHRUNX_DIR="${RUNTIME_DIR}/torchrunx"
WARP_DIR="${RUNTIME_DIR}/warp"
mkdir -p "${TORCHRUNX_DIR}" "${WARP_DIR}"

# Compute nodes may add a cluster CA to the host trust bundle.  The inherited
# SSL_CERT_FILE then points outside the container, so copy that bundle into the
# already-bound job runtime and expose one valid path to Python and W&B Core.
HOST_CA_BUNDLE="${SSL_CERT_FILE:-}"
if [[ -z "${HOST_CA_BUNDLE}" || ! -f "${HOST_CA_BUNDLE}" ]]; then
  for candidate in \
    /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
    /etc/ssl/certs/ca-certificates.crt; do
    if [[ -f "${candidate}" ]]; then
      HOST_CA_BUNDLE="${candidate}"
      break
    fi
  done
fi
CA_ENV_ARGS=()
if [[ -n "${HOST_CA_BUNDLE}" && -f "${HOST_CA_BUNDLE}" ]]; then
  JOB_CA_BUNDLE="${RUNTIME_DIR}/cluster-ca-bundle.pem"
  cp -- "${HOST_CA_BUNDLE}" "${JOB_CA_BUNDLE}"
  chmod 0444 "${JOB_CA_BUNDLE}"
  CA_ENV_ARGS=(
    --env "SSL_CERT_FILE=${JOB_CA_BUNDLE}"
    --env "REQUESTS_CA_BUNDLE=${JOB_CA_BUNDLE}"
    --env "CURL_CA_BUNDLE=${JOB_CA_BUNDLE}"
  )
  echo "[TLS] using host CA bundle inside container: ${JOB_CA_BUNDLE}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  if [[ -n "${REQUIRE_CUDA_DEVICE_SUBSTRING:-}" ]]; then
    for gpu_name in "${GPU_NAMES[@]}"; do
      if [[ "${gpu_name}" != *"${REQUIRE_CUDA_DEVICE_SUBSTRING}"* ]]; then
        echo "Expected ${REQUIRE_CUDA_DEVICE_SUBSTRING} GPU, found ${gpu_name}" >&2
        exit 1
      fi
    done
  fi
fi

# Slurm normally exposes exactly the allocation through CUDA_VISIBLE_DEVICES.
# Do not count nvidia-smi rows: on some eight-GPU nodes NVML can still report
# devices outside the job cgroup.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
  if [[ "${#VISIBLE_GPU_IDS[@]}" -ne "${GPU_COUNT}" ]]; then
    echo "Expected ${GPU_COUNT} CUDA_VISIBLE_DEVICES entries; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
fi


if [[ "${LAUNCH_MODE}" == "sweep" ]]; then
  CONTAINER_COMMAND='cd /workspace/vision-based-rl; exec wandb agent --count "$2" "$1"'
  CONTAINER_ARGS=(bash "${SWEEP_ID}" "${SWEEP_COUNT}")
else
  [[ $# -ge 1 ]] || { echo "Training requires a registered MJLab task ID." >&2; exit 2; }
  # The launcher owns which GPUs the run may touch, in both topologies: the
  # allocation is what Slurm granted and a job must not reach outside it.
  for argument in "$@"; do
    case "${argument}" in
      --gpu-ids|--gpu-ids=*)
        echo "The launcher owns --gpu-ids; it follows the Slurm allocation." >&2
        exit 2
        ;;
    esac
  done
  USER_SET_NUM_ENVS="0"
  for argument in "$@"; do
    case "${argument}" in
      --env.scene.num-envs|--env.scene.num-envs=*) USER_SET_NUM_ENVS="1" ;;
    esac
  done
  if [[ "${GPU_COUNT}" == "4" ]]; then
    # Four ranks share one policy update, so the per-rank environment count is
    # part of the topology rather than a tuning knob: changing it silently
    # changes the effective batch every rank contributes to.
    [[ "${USER_SET_NUM_ENVS}" == "0" ]] || {
      echo "The four-GPU launcher owns --env.scene.num-envs." >&2
      exit 2
    }
    TRAIN_ARGS=("$@" --env.scene.num-envs 1024 --gpu-ids all)
  else
    # One rank owns the whole update, so the environment count *is* the batch
    # and is the thing worth tuning. Default to the four-rank per-rank figure so
    # an unqualified single-GPU run is the familiar one.
    #
    # `--gpu-ids` is deliberately *not* passed. Its type is
    # `list[int] | Literal["all"] | None`, which tyro will not build from a bare
    # `0`, and mjlab's own single-GPU command omits the flag rather than quoting
    # a one-element list. The default is already `[0]`, and `select_gpus` reads
    # it as an index into CUDA_VISIBLE_DEVICES -- which Slurm has already
    # narrowed to the one granted GPU, so index 0 *is* the allocation.
    TRAIN_ARGS=("$@")
    [[ "${USER_SET_NUM_ENVS}" == "1" ]] || TRAIN_ARGS+=(--env.scene.num-envs 1024)
  fi
  CONTAINER_COMMAND='cd /workspace/vision-based-rl; exec python -m vbrl.scripts.train "$@"'
  CONTAINER_ARGS=(bash "${TRAIN_ARGS[@]}")
fi

apptainer exec --nv \
  "${CA_ENV_ARGS[@]}" \
  --env "MUJOCO_GL=egl" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=${CONTAINER_REPO}/src" \
  --env "VBRL_REPO_ROOT=${CONTAINER_REPO}" \
  --env "VBRL_MODEL_ROOT=${MODEL_ROOT}" \
  --env "TORCHRUNX_LOG_DIR=${TORCHRUNX_DIR}" \
  --env "WARP_CACHE_PATH=${WARP_DIR}" \
  --bind "${REPO}:${CONTAINER_REPO}" \
  --bind "${RUNTIME_DIR}:${RUNTIME_DIR}" \
  "${MODEL_BIND_ARGS[@]}" \
  "${CONTAINER_PATH}" \
  bash -lc "${CONTAINER_COMMAND}" "${CONTAINER_ARGS[@]}"
