#!/bin/sh
#BATCH --job-name=classification
#SBATCH --account=def-sfabbro
#SBATCH --time=1:00:0
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
source $HOME/lensing/bin/activate
python create_evaluation_cutouts.py
python classification_pipeline.py
