#!/bin/bash
#SBATCH -J ppo-learning-rates
#SBATCH --time=01-15:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH --gres=gpu:k20xm:1 
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=64g                             # requesting 2GB of RAM total
#SBATCH --output=ppo-learning-rates.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=ppo-learnings-rates.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

source ~/.bashrc
module load anaconda/2023.07
module load cuda/11.7
module load cudnn/8.9.5-11.x
source activate mthesis
wandb login "eaa09efc618b8de89f1aaf350442d4ee69be3cf5"
# learning rates = 0.01, 0.001, 0.0003 (default)
python disc-episode-adding-layers.py --meta_learning_rate=0.01 --n_runs 1 --epochs 10  --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, lr=0.01"
python disc-episode-adding-layers.py --meta_learning_rate=0.001 --n_runs 1 --epochs 10 --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, lr=0.001"
python disc-episode-adding-layers.py --meta_learning_rate=0.0003 --n_runs 1 --epochs 10 --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, lr=0.0003(default)"
