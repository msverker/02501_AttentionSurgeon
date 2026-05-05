
# bsub << EOF
# #!/bin/bash
# #BSUB -J run_pruning_agent_imagenet
# #BSUB -q gpuv100
# #BSUB -gpu "num=1:mode=exclusive_process"
# #BSUB -n 4
# #BSUB -R "rusage[mem=8GB]"
# #BSUB -W 24:00
# #BSUB -o imagenet_standard_%J.out
# #BSUB -e imagenet_standard_%J.err

# uv run src/pruning_agent.py --dataset imagenet --episodes 500
# EOF

# bsub << EOF
# #!/bin/bash
# #BSUB -J run_pruning_agent_ade20k
# #BSUB -q gpuv100
# #BSUB -gpu "num=1:mode=exclusive_process"
# #BSUB -n 4
# #BSUB -R "rusage[mem=8GB]"
# #BSUB -W 24:00
# #BSUB -o ade20k_standard_%J.out
# #BSUB -e ade20k_standard_%J.err

# uv run src/pruning_agent.py --dataset ade20k --episodes 500

# EOF

bsub << EOF
#!/bin/bash
#BSUB -J run_pruning_agent_coco
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o coco_standard_%J.out
#BSUB -e coco_standard_%J.err

uv run src/pruning_agent.py --dataset coco --episodes 500

EOF


