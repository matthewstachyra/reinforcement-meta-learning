#!/bin/bash
#SBATCH -J test-wandb
#SBATCH --time=00-10:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=16g                             # requesting 2GB of RAM total
#SBATCH --output=test-wandb.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=test-wandb.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

module load anaconda/2023.07
module load cuda/11.7
module load cudnn/8.9.5-11.x
source activate mthesis
wandb login "eaa09efc618b8de89f1aaf350442d4ee69be3cf5"
python bounded-discrete.py --pretrain --epochs 1 --timesteps 1000 --n_tasks 1 --sb3_model='PPO' --sb3_policy='MlpPolicy' --log_dir='slurm' 
