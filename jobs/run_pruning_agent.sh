
bsub << EOF
#!/bin/bash
#BSUB -J run_pruning_agent_imagenet
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o imagenet_standard.out
#BSUB -e imagenet_standard.err

uv run src/pruning_agent.py --dataset imagenet --episodes 20
EOF

bsub << EOF
#!/bin/bash
#BSUB -J run_pruning_agent_ade20k
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o ade20k_standard.out
#BSUB -e ade20k_standard.err

uv run src/pruning_agent.py --dataset ade20k --episodes 20

EOF

bsub << EOF
#!/bin/bash
#BSUB -J run_pruning_agent_coco
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o coco_standard.out
#BSUB -e coco_standard.err

uv run src/pruning_agent.py --dataset coco --episodes 20

EOF