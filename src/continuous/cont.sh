#!/bin/bash
#SBATCH -J test
#SBATCH --time=00-15:00:00                  # requested time (DD-HH:MM:SS)
#SBATCH -p gpu 
#SBATCH -N 1                                # 1 nodes
#SBATCH -n 2                                # 2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=16g                             # requesting 2GB of RAM total
#SBATCH --output=test.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=test.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                     # email
#SBATCH --mail-user=matthew.stachyra@tufts.edu

source ~/.bashrc
module load anaconda/2023.07
module load cuda/11.7
module load cudnn/8.9.5-11.x
source activate mthesis
wandb login "eaa09efc618b8de89f1aaf350442d4ee69be3cf5"
python cont-nudging-layers.py --n_runs 1 --epochs 1 --timesteps 1000 --n_tasks 2 --sb3_model='TD3' --sb3_policy='MlpPolicy' --log_dir='slurm' 
