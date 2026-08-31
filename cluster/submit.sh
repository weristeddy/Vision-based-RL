#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${VBRL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export VBRL_CLUSTER_DIR="${SCRIPT_DIR}"
export VBRL_REPO_ROOT="${REPO}"
PROFILE="a100"
MODE="train"
SWEEP_ID=""
SWEEP_COUNT="1"
COUNT_SET="0"
USE_FALLBACK="0"
DEPENDENCY=""

usage() {
  command cat <<'EOF'
Usage:
  bash cluster/submit.sh --profile a100 -- TASK_ID [native MJLab options ...]
  bash cluster/submit.sh --profile a100 --sweep-agent entity/project/sweep-id [--count N]
  bash cluster/submit.sh --profile a100 --dependency afterany:<job-id> -- TASK_ID

Profiles: a100, b200, testing, a100-single, b200-single

A four-GPU profile receives four GPUs and 1,024 environments per rank, and owns
both --env.scene.num-envs and --gpu-ids. A *-single profile receives one GPU,
defaults to 1,024 environments, and lets --env.scene.num-envs through: with one
rank the environment count is the batch, so it is a tuning knob rather than
topology. mjlab runs a single GPU in-process, with no torchrunx launcher.

The first training argument is one registered MJLab task ID; remaining arguments
pass unchanged to the native MJLab/Tyro CLI.
Sweep agents run one trial unless --count is provided.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    --sweep-agent)
      [[ $# -ge 2 ]] || { echo "--sweep-agent requires entity/project/sweep-id" >&2; exit 2; }
      MODE="sweep"
      SWEEP_ID="$2"
      shift 2
      ;;
    --count)
      [[ $# -ge 2 ]] || { echo "--count requires a positive integer" >&2; exit 2; }
      SWEEP_COUNT="$2"
      COUNT_SET="1"
      shift 2
      ;;
    --fallback)
      USE_FALLBACK="1"
      shift
      ;;
    --dependency)
      [[ $# -ge 2 ]] || { echo "--dependency requires a value" >&2; exit 2; }
      DEPENDENCY="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown launcher option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${DEPENDENCY}" \
  && ! "${DEPENDENCY}" =~ ^(after|afterany|afterok|afternotok):[0-9]+(:[0-9]+)*$ ]]; then
  echo "Unsupported dependency expression: ${DEPENDENCY}" >&2
  exit 2
fi
if [[ "${MODE}" == "sweep" ]]; then
  if [[ ! "${SWEEP_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    echo "--sweep-agent must use entity/project/sweep-id" >&2
    exit 2
  fi
  if [[ ! "${SWEEP_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--count must be a positive integer; got ${SWEEP_COUNT}" >&2
    exit 2
  fi
  [[ $# -eq 0 ]] || { echo "Sweep agents do not accept training overrides." >&2; exit 2; }
else
  [[ "${COUNT_SET}" == "0" ]] || { echo "--count requires --sweep-agent" >&2; exit 2; }
  [[ $# -ge 1 ]] || { echo "Training requires a registered MJLab task ID." >&2; exit 2; }
  [[ "$1" != --* ]] || {
    echo "The first training argument must be a registered task ID." >&2; exit 2;
  }
fi

PROFILE_FILE="${SCRIPT_DIR}/profiles/${PROFILE}.conf"
[[ -f "${PROFILE_FILE}" ]] || { echo "Unknown cluster profile: ${PROFILE}" >&2; exit 2; }
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

: "${PARTITION:?Profile must set PARTITION}"
: "${CPUS_PER_TASK:?Profile must set CPUS_PER_TASK}"
: "${MEMORY:?Profile must set MEMORY}"
: "${TIME_LIMIT:?Profile must set TIME_LIMIT}"
: "${GPU_COUNT:?Profile must set GPU_COUNT}"
case "${GPU_COUNT}" in
  1|4) ;;
  *)
    echo "Training profiles must request one or four GPUs; got ${GPU_COUNT}" >&2
    exit 2
    ;;
esac

SELECTED_PARTITION="${PARTITION}"
SELECTED_ACCOUNT="${ACCOUNT:-}"
if [[ "${USE_FALLBACK}" == "1" ]]; then
  : "${FALLBACK_PARTITION:?Profile ${PROFILE} has no fallback partition}"
  SELECTED_PARTITION="${FALLBACK_PARTITION}"
  SELECTED_ACCOUNT="${FALLBACK_ACCOUNT:-}"
fi

mkdir -p "${REPO}/artifacts/slurm"
SBATCH_ARGS=(
  --nodes=1
  --ntasks=1
  --gpus="${GPU_COUNT}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
  --partition="${SELECTED_PARTITION}"
  --output="${REPO}/artifacts/slurm/%x-%j.out"
  --export=ALL
  --job-name="vbrl-${MODE}"
)
[[ -z "${SELECTED_ACCOUNT}" ]] || SBATCH_ARGS+=(--account="${SELECTED_ACCOUNT}")
[[ -z "${DEPENDENCY}" ]] || SBATCH_ARGS+=(--dependency="${DEPENDENCY}")

if [[ "${MODE}" == "sweep" ]]; then
  exec sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/train.slurm" "${PROFILE}" \
    --sweep-agent "${SWEEP_ID}" --count "${SWEEP_COUNT}"
fi
exec sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/train.slurm" "${PROFILE}" "$@"
