#!/bin/bash
#SBATCH -J ppo-1-task-test
#SBATCH --time=00-08:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH --gres=gpu:p100:1            # partition
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=16g                             # requesting 2GB of RAM total
#SBATCH --output=slurm/ppo-1-task-test.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=slurm/ppo-1-task-test.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

module load anaconda/2023.07
source activate mthesis
python bounded-networks.py --epochs 1 --timesteps 60000 --n_tasks 1 --sb3_model='PPO' --sb3_policy='MlpPolicy' --log_dir='slurm'
