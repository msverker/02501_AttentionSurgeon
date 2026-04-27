#!/bin/bash
#BSUB -J run_census
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00

uv run src/run_census.py