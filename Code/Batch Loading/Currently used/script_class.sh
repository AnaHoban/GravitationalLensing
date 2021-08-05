#!/bin/sh
#SBATCH --job-name=classification
#SBATCH --account=def-sfabbro
#SBATCH --time=15:00:00
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
source $HOME/lensing/bin/activate
python new_pipeline.py
