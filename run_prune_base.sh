#!/bin/sh
### General options
### –- specify queue --
# BSUB -q gpuv100
### -- set the job Name --
#BSUB -J AttentionSurgeon_Pruning
### -- ask for number of processors (cores) --
#BSUB -n 4
### -- Select node with 1 GPU --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --
#BSUB -W 02:00
### -- request GB memory per process --
#BSUB -R "rusage[mem=8GB]"
### -- Specify the output and error file. %J is the job-id --
### -- NOTE: We remove the logs/ prefix so it doesn't fail if the folder isn't there --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Load modules
module load python3/3.10.12
module load cuda/12.1

cd ADLCV/Proj/02501_AttentionSurgeon

# Activate virtual environment
source .venv/bin/activate

# Ensure logs directory exists for safety
mkdir -p results logs

echo "Starting ADE20K Baseline Pruning..."
# Reduced batch size to 4 to be safe with memory on large tasks
python src/run_baseline_pruning.py --dataset ade20k --batch-size 4

echo "Starting COCO Baseline Pruning..."
python src/run_baseline_pruning.py --dataset coco --batch-size 4

echo "Job finished."