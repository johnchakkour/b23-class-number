#!/bin/bash
#SBATCH --job-name=b23-certify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=03:00:00
#SBATCH --output=certify_b23_%j.out
# ---------------------------------------------------------------------------
# Step 2 of 2: build the certificate from the hits of step 1
# (search/search_b23.slurm).  Needs only Python 3.8+, no extra packages.
# Submit from the repository root (see search_b23.slurm).
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load StdEnv/2023

RUN=$SCRATCH/b23_run
python3 --version
python3 search/certify_b23.py --selftest
python3 search/certify_b23.py "$RUN/b23" --p 23 --cutoff 50000000 --out "$RUN/b23.certificate.txt"
