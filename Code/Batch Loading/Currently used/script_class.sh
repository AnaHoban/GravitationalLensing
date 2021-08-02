#!/bin/sh
#BATCH --job-name=classification
#SBATCH --account=def-sfabbro
#SBATCH --gres=gpu:v100:1
#SBATCH --time=15:00:00
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
source $HOME/lensing/bin/activate
python create_evaluation_cutouts.py