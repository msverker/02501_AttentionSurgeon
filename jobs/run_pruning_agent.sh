#!/bin/bash
#BSUB -J run_pruning_agent
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00

uv run src/pruning_agent.py