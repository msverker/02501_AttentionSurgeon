#!/bin/bash
#BSUB -J run_census
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00

uv run src/run_census.py


# OR 
# #!/bin/sh
# #BSUB -q gpuv100
# #BSUB -J train_gpt
# #BSUB -gpu "num=1:mode=exclusive_process"
# #BSUB -n 4
# #BSUB -R "rusage[mem=8GB]"
# #BSUB -R "span[hosts=1]"
# #BSUB -W 2:00
# ##BSUB -u s216143@dtu.dk
# #BSUB -B
# #BSUB -N
# #BSUB -o Output_%J.out
# #BSUB -e Output_%J.err

# module load python3/3.11.13
# source /work3/s216143/dlcv/dlcv/bin/activate


# python -u /work3/s216143/02501_AttentionSurgeon/src/run_census.py classification