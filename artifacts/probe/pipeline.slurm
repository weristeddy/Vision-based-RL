#!/usr/bin/env bash
# One stage of the probe pipeline. Args: <script> <shard i/n> [extra]
#SBATCH --job-name=probe
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --partition=ber-gpu1
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=08:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
P="${SLURM_SUBMIT_DIR}/artifacts/probe"
apptainer exec --nv \
  --env "PYTHONPATH=${SLURM_SUBMIT_DIR}/src:${P}" \
  --env "VBRL_REPO_ROOT=${SLURM_SUBMIT_DIR}" \
  --env "VBRL_PROBE_GENERATION=${GENERATION:-SlowGoal}" \
  --env "VBRL_PROBE_SHARD=${2:-0/1}" \
  rl.sif python "${P}/${1}" ${3:-}
