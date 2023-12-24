#!/bin/bash
#SBATCH -J rppo-10runs-k20
#SBATCH --time=01-15:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH -gres=gpu:k20xm:1
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=64g                             # requesting 2GB of RAM total
#SBATCH --output=rppo.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=rppo.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

source ~/.bashrc
module load anaconda/2023.07
module load cuda/11.7
module load cudnn/8.9.5-11.x
source activate mthesis
wandb login "eaa09efc618b8de89f1aaf350442d4ee69be3cf5"
python disc-episode-adding-layers.py --meta_n_steps=5 --n_runs 10 --epochs 20 --timesteps 1000 --n_tasks 7 --sb3_model='RecurrentPPO' --sb3_policy='MlpLstmPolicy' --data_dir='slurm' --wandb_run="rppo (nsteps=5), 10, 20, 1000, 7"
