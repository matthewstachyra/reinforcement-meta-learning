#!/bin/bash
#SBATCH -J initial-bounded-networks-run     # job name
#SBATCH --time=00-05:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu --gres=gpu:p100:1            # partition
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=16g                             # requesting 2GB of RAM total
#SBATCH --output=bounded-networks-run.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=bounded-networks-run.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

module load anaconda/2023.07
conda activate mthesis
python bounded-networks.py
