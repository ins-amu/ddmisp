#!/bin/bash -l
#SBATCH --job-name="simprop"
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH -o run/log/slurm-%j.out
#SBATCH -o run/log/slurm-%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export STAN_NUM_THREADS=$SLURM_CPUS_PER_TASK

f=${1}
nobs=${2}
chain=${3}

mkdir -p run/synth/group/f${f}_nobs${nobs}/output/
mkdir -p run/synth/group/f${f}_nobs${nobs}/log/

echo ${f} ${nobs} ${chain}

srun ./stan/msinf sample                                       \
    num_warmup=500 num_samples=500                             \
    random seed=42 id=${chain}                                 \
    data file=run/synth/group/f${f}_nobs${nobs}/input/input.R  \
    output file=run/synth/group/f${f}_nobs${nobs}/output/chain_${chain}.csv &> run/synth/group/f${f}_nobs${nobs}/log/chain_${chain}.csv
