#!/bin/bash
#SBATCH -J Lindblad_simulation
#SBATCH -o ./job_runs/%x_%j.out
#SBATCH -e ./job_runs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --clusters=serial
#SBATCH --partition=serial_long

#SBATCH --time=00:50:00


source /dss/dsshome1/04/di54zil/thesis/Lindblad-Simulation-Neutral-Atom/.venv/bin/activate
python numerical_simulation/lindbladian_simulation/extract_unitary.py