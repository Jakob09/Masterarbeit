#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3



echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

python -u targeted_batch_wise_pipeline.py $SLURM_ARRAY_TASK_ID
echo "done"
