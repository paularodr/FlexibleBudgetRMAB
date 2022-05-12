#!/bin/bash
#SBATCH -n 24        # number of cores
#SBATCH -N 1         # ensure all cores are on one machine
#SBATCH -t 00-02:00  # runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe,shared   # partition to submit to
#SBATCH --mem=1000   # memory pool for all cores
#SBATCH -o ./logs/dropOutState_%A_%a.o     # output file
#SBATCH -e ./logs/dropOutState_%A_%a.e     # error file

module load Anaconda3/2020.11
source activate frmab
python run_experiments.py ${SLURM_ARRAY_TASK_ID}