#!/bin/bash

#SBATCH --time=01-15:00:00          
#SBATCH -p gpu 
#SBATCH -N 1                               
#SBATCH -n 2                                
#SBATCH --mem=64g                            
#SBATCH --output=evaluation/"$1".%j.%N.out
#SBATCH --error=evaluation/"$1".%j.%N.err   
#SBATCH --mail-type=ALL                     
#SBATCH --mail-user=matthew.stachyra@tufts.edu

source ~/.bashrc
module load anaconda/2023.07
module load cuda/11.7
module load cudnn/8.9.5-11.x
source activate mthesis
wandb login "eaa09efc618b8de89f1aaf350442d4ee69be3cf5"
python disc-episode-adding-layers.py --n_runs 6 --timesteps 1200 --n_tasks 7 --sb3_model="RecurrentPPO"

exit 
