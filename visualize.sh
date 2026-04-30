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
#BSUB -R "rusage[mem=4GB]"
### -- Specify the output and error file. %J is the job-id --
### -- NOTE: We remove the logs/ prefix so it doesn't fail if the folder isn't there --
#BSUB -o visualize_%J.out
#BSUB -e visualize_%J.err
# -- end of LSF options --

# Load modules
module load python3/3.10.12
module load cuda/12.1

cd ADLCV/Proj/02501_AttentionSurgeon

# Activate virtual environment
source .venv/bin/activate

# Ensure logs directory exists for safety
mkdir -p results logs
cd ADLCV/Proj/02501_AttentionSurgeon
source .venv/bin/activate
echo "Starting visualization of pruning results..."
python src/visualize_pruning.py --input results/baseline_pruning_ade20k_results.npz --output figures --dataset-name ADE20K --metric-name "mIoU"
python src/visualize_pruning.py --input results/baseline_pruning_coco_results.npz --output figures --dataset-name COCO --metric-name "mAP"

echo "Job finished."