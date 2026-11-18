#!/bin/bash
#SBATCH --job-name=b23-search
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --output=search_b23_%j.out
# ---------------------------------------------------------------------------
# Step 1 of 2: search B_{23,1} for every element with coefficients in {-1,0,1}
# (all (3^23-1)/2 = 47,071,589,413 of them, up to sign), recording those whose
# norm is a prime or a usable factorization below 10^15.
#
# Written for the Trillium cluster (192 cores per node); about 20 core-hours.
# Submit from the repository root, together with step 2:
#     jid=$(sbatch --parsable search/search_b23.slurm)
#     sbatch --dependency=afterok:$jid search/certify_b23.slurm
# The hits (about 4 GB) go to $SCRATCH/b23_run.
# ---------------------------------------------------------------------------
set -euo pipefail
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PLACES=cores OMP_PROC_BIND=close
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

RUN=$SCRATCH/b23_run
mkdir -p "$RUN"
rm -f "$RUN"/b23_shard*_rank*.txt               # the program appends: start clean

mpicc -O3 -march=native -fopenmp -o "$RUN/bp1_norm_search" search/bp1_norm_search.c -lgmp -lm
srun --nodes=1 --ntasks=1 "$RUN/bp1_norm_search" --selftest 13

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none \
     "$RUN/bp1_norm_search" --p 23 --coeff 1 \
        --bound 1000000000000000 --comp-bound 1000000000000000 \
        --out "$RUN/b23"
