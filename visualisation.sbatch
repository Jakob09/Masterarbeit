#!/bin/bash
#SBATCH --partition=study     
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=1         
#SBATCH --time=01:00:00            

echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID runs on $(hostname)"

python -u image_visualisation.py
echo "done"