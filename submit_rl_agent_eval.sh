#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J AttentionSurgeon_RL_EVAL
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

# Load modules
module load python3/3.10.12
module load cuda/12.1

# Activate virtual environment
source .venv/bin/activate

# Ensure logs directory exists for safety
mkdir -p results logs

echo "Starting ADE20K RL Agent Pruning Evaluation..."
python src/run_baseline_pruning.py --dataset ade20k --batch-size 4 --run-agent --only-agent --rl-agent-ckpt checkpoints/rl_agent_ppo_ade20k.pt

echo "Starting COCO RL Agent Pruning Evaluation..."
python src/run_baseline_pruning.py --dataset coco --batch-size 4 --run-agent --only-agent --rl-agent-ckpt checkpoints/rl_agent_ppo_coco.pt

echo "Job finished."