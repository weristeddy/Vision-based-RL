#!/usr/bin/env bash
#
# Submit one four-GPU training job per architecture in a task-ID variant.
#
#   bash cluster/submit_sweep.sh --profile b200
#   bash cluster/submit_sweep.sh --dry-run
#   bash cluster/submit_sweep.sh -- --agent.max-iterations 6000
#
# Each job is one `cluster/submit.sh` call, so the four-GPU topology and the
# 1,024 environments per rank come from there and are not repeated here.
#
# The task IDs are read out of the registration statements themselves. VBRL
# writes one line per task ID precisely so they can be grepped, which keeps this
# script from carrying a second copy of the list that could drift.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${VBRL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROFILE="b200"
VARIANT="SlowGoal"
ARCH_KEEP=""
ARCH_DROP=""
DRY_RUN="0"
ASSUME_YES="0"
USE_FALLBACK="0"

usage() {
  command cat <<'EOF'
Usage:
  bash cluster/submit_sweep.sh [--profile b200] [--variant SlowGoal]
                              [--arch LIST] [--exclude LIST]
                              [--fallback] [--dry-run] [--yes]
                              [-- NATIVE OPTIONS ...]

  --profile NAME   a100, b200, testing, a100-single, b200-single (default: b200)
  --variant TOKEN  the variant token in the task IDs to sweep (default: SlowGoal)
  --arch LIST      comma-separated substrings; keep only task IDs matching one.
                   Filters submission, never registration: an ID has to stay
                   registered for its existing checkpoints to remain loadable,
                   so narrowing a sweep belongs here rather than in the task
                   registry.
  --exclude LIST   comma-separated substrings; drop task IDs matching one.
                   Applied after --arch. Note a trailing hyphen disambiguates a
                   prefix: "R3MResNet50-" excludes layer 4 and leaves
                   "R3MResNet50L3-" alone.
  --fallback       use the profile's fallback partition (required for --profile
                   testing on hpc-ber1, where the primary partition is elsewhere)
  --dry-run        list the jobs that would be submitted and exit
  --yes            skip the confirmation prompt
  --               everything after this is passed to every job unchanged

Each job gets the profile's GPU_COUNT (1 or 4) from cluster/submit.sh. A
four-GPU profile pins 1,024 environments per rank; a *-single profile defaults
to 1,024 and lets --env.scene.num-envs through after --.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
               PROFILE="$2"; shift 2 ;;
    --variant) [[ $# -ge 2 ]] || { echo "--variant requires a value" >&2; exit 2; }
               VARIANT="$2"; shift 2 ;;
    --arch)    [[ $# -ge 2 ]] || { echo "--arch requires a value" >&2; exit 2; }
               ARCH_KEEP="$2"; shift 2 ;;
    --exclude) [[ $# -ge 2 ]] || { echo "--exclude requires a value" >&2; exit 2; }
               ARCH_DROP="$2"; shift 2 ;;
    --fallback) USE_FALLBACK="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --yes|-y)  ASSUME_YES="1"; shift ;;
    --help|-h) usage; exit 0 ;;
    --)        shift; break ;;
    *)         echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${VARIANT}" =~ ^[A-Za-z0-9]+$ ]] || {
  echo "--variant must be a single alphanumeric token; got ${VARIANT}" >&2
  exit 2
}

mapfile -t TASK_IDS < <(
  grep -rhoE "\"Mjlab-[A-Za-z0-9]+-${VARIANT}-[A-Za-z0-9-]+\"" \
    "${REPO}"/src/vbrl/tasks/*/config/*/__init__.py \
    | tr -d '"' | sort -u
)

if [[ ${#TASK_IDS[@]} -eq 0 ]]; then
  echo "No registered task IDs contain the variant '${VARIANT}'." >&2
  echo "Check the token against: vbrl-list tasks" >&2
  exit 1
fi

filter_task_ids() {
  local mode="$1" patterns="$2" kept=() task_id pattern hit
  [[ -n "${patterns}" ]] || return 0
  IFS=',' read -r -a PATTERN_LIST <<<"${patterns}"
  for task_id in "${TASK_IDS[@]}"; do
    hit="0"
    for pattern in "${PATTERN_LIST[@]}"; do
      [[ -n "${pattern}" ]] || continue
      [[ "${task_id}" == *"${pattern}"* ]] && hit="1"
    done
    if [[ "${mode}" == "keep" ]]; then
      [[ "${hit}" == "1" ]] && kept+=("${task_id}")
    else
      [[ "${hit}" == "0" ]] && kept+=("${task_id}")
    fi
  done
  TASK_IDS=("${kept[@]}")
}

filter_task_ids keep "${ARCH_KEEP}"
filter_task_ids drop "${ARCH_DROP}"

if [[ ${#TASK_IDS[@]} -eq 0 ]]; then
  echo "No task IDs survive --arch '${ARCH_KEEP}' / --exclude '${ARCH_DROP}'." >&2
  exit 1
fi

FALLBACK_ARGS=()
[[ "${USE_FALLBACK}" == "0" ]] || FALLBACK_ARGS+=(--fallback)

PARTITION_NOTE=""
[[ "${USE_FALLBACK}" == "0" ]] || PARTITION_NOTE=" (fallback partition)"
SWEEP_PROFILE_FILE="${SCRIPT_DIR}/profiles/${PROFILE}.conf"
[[ -f "${SWEEP_PROFILE_FILE}" ]] || {
  echo "Unknown cluster profile: ${PROFILE}" >&2
  exit 2
}
# shellcheck disable=SC1090
SWEEP_GPU_COUNT="$(source "${SWEEP_PROFILE_FILE}"; echo "${GPU_COUNT:-?}")"
echo "Variant ${VARIANT}: ${#TASK_IDS[@]} jobs on profile ${PROFILE}${PARTITION_NOTE}, ${SWEEP_GPU_COUNT} GPU(s) each"
for task_id in "${TASK_IDS[@]}"; do
  echo "  ${task_id}"
done
[[ $# -eq 0 ]] || echo "Extra options for every job: $*"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "(dry run -- nothing submitted)"
  exit 0
fi

if [[ "${ASSUME_YES}" != "1" ]]; then
  read -r -p "Submit ${#TASK_IDS[@]} jobs ($(( ${#TASK_IDS[@]} * SWEEP_GPU_COUNT )) GPUs)? [y/N] " reply
  [[ "${reply}" == "y" || "${reply}" == "Y" ]] || { echo "Cancelled."; exit 1; }
fi

for task_id in "${TASK_IDS[@]}"; do
  printf '%-62s ' "${task_id}"
  bash "${SCRIPT_DIR}/submit.sh" --profile "${PROFILE}" "${FALLBACK_ARGS[@]}" \
    -- "${task_id}" "$@"
done

echo "Submitted ${#TASK_IDS[@]} jobs. Watch them with: squeue -u \"\${USER}\""
