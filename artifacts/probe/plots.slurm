#!/usr/bin/env bash
#SBATCH --job-name=probe-plots
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --partition=ber-gpu1
#SBATCH --account=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
P="${SLURM_SUBMIT_DIR}/artifacts/probe"
G="${GENERATION:-SlowGoal}"
py() { apptainer exec --env "PYTHONPATH=${SLURM_SUBMIT_DIR}/src:${P}" rl.sif python "$@"; }
py "${P}/merge.py" "${G}"
py "${P}/plot_scaling.py"   "${P}/scaling.json"        "${P}/scaling.png"
py "${P}/plot_conditions.py" "${P}/conditions_${G}.json" "${P}/conditions_${G}.png"
py "${P}/plot_rollout.py"   "${P}/rollout_${G}.npz"    "${P}/rollout_traces_${G}.png"
py "${P}/plot_tolerance.py"
echo "=== figures ==="; ls -1 "${P}"/*.png | sed 's/^/  /'
