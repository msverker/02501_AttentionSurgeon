#!/bin/bash
#BSUB -J train_probe
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 4:00

uv run src/train.py --task cls --epochs 5
