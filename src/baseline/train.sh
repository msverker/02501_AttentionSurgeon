#!/bin/bash
#BSUB -J Train_classificationhead          # Job name
#BSUB -q gpuv100             # Queue name (check with bqueues; use gpuv100 or gpua100 typically)
#BSUB -n 4                       # Number of CPU cores
#BSUB -R "span[hosts=1]"         # Run on a single node
#BSUB -R "rusage[mem=16GB]"      # Memory per host (16 GB is usually fine)
#BSUB -W 15:00                # Wall time (HH:MM) - Increased to 5 hours for 30 models
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU
#BSUB -o Train_classificationhead_%J.out  # Standard output
#BSUB -e Train_classificationhead_%J.err  # Standard error

# --- ENVIRONMENT SETUP ---
# Always go to your project folder
cd /zhome/29/b/146867/ADLCV || exit

# Activate your virtual environment
source .venv/bin/activate
cd Proj/02501_AttentionSurgeon/src/baseline || exit
# Optional: make sure CUDA is visible (good habit)
export CUDA_VISIBLE_DEVICES=0

python train_imagenet_classification.py

