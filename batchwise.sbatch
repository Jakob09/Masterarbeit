#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00



echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

python -u batch_wise_pipeline.py 1
echo "done"
