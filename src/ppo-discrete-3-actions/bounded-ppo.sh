#!/bin/bash
#SBATCH -J bounded-ppo-mlp-policy      # job name
#SBATCH --time=00-08:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH --gres=gpu:a100:1            # partition
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=16g                             # requesting 2GB of RAM total
#SBATCH --output=slurm/bounded-ppo-mlp-policy-run.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=slurm/bounded-ppo-mlp-policy-run.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

source /cluster/tufts/hpc/tools/anaconda/202307/etc/profile.d/conda.sh
module load anaconda/2023.07
source /cluster/home/mstach01/condaenv/mthesis/bin/activate
python bounded-networks.py --sb3_model='PPO' --sb3_polcy='MlpPolicy' --log_dir='slurm'
