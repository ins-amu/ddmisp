#!/bin/bash -l

#SBATCH --job-name="ssinf"
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --output=run/log/ssinf.%j.out
#SBATCH --error=run/log/ssinf.%j.err

module load daint-gpu

BATCH_FILE=$1

srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE   1  12  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  13  24  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  25  36  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  37  48  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  49  60  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  61  72  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  73  84  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  85  96  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE  97 108  &
srun -N1 -n1 -c12 run-tasks.sh $BATCH_FILE 109 120  &

wait
