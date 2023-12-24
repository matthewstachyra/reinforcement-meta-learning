#!/bin/bash
#SBATCH -J ppo-nsteps
#SBATCH --time=01-15:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=64g                             # requesting 2GB of RAM total
#SBATCH --output=ppo-nsteps.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=ppo-nsteps.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

source ~/.bashrc
module load anaconda/2023.07
module load cuda/11.7
module load cudnn/8.9.5-11.x
source activate mthesis
wandb login "eaa09efc618b8de89f1aaf350442d4ee69be3cf5"
# nsteps = 5, 512, 1024, 2048(default)
python disc-episode-adding-layers.py --meta_n_steps=5 --n_runs 1 --epochs 10  --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, nsteps=5"
python disc-episode-adding-layers.py --meta_n_steps=512 --n_runs 1 --epochs 10 --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, nsteps=512"
python disc-episode-adding-layers.py --meta_n_steps=1024 --n_runs 1 --epochs 10 --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, nsteps=1024"
python disc-episode-adding-layers.py --meta_n_steps=2048 --n_runs 1 --epochs 10 --timesteps 1000 --n_tasks 7 --sb3_model='PPO' --sb3_policy='MlpPolicy' --data_dir='slurm' --wandb_run="ppo, nsteps=2048"
