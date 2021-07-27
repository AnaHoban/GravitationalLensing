#!/bin/sh
#BATCH --job-name=classification
#SBATCH --account=rrg-kyi
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
source $HOME/umap/bin/activate
python create_evaluation_cutouts.py
python classification_pipeline.py
