
bsub << EOF
#!/bin/bash
#BSUB -J run_baseline_pruning_imagenet
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o imagenet_standard_%J.out
#BSUB -e imagenet_standard_%J.err

uv run src/run_baseline_pruning.py --dataset imagenet --rl-agent-ckpt checkpoints/rl_agent_ppo_imagenet.pt --run-agent
EOF



bsub << EOF
#!/bin/bash
#BSUB -J run_baseline_pruning_ade40k
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o imagenet_standard_%J.out
#BSUB -e imagenet_standard_%J.err

uv run src/run_baseline_pruning.py --dataset ade40k --rl-agent-ckpt checkpoints/rl_agent_ppo_ade40k.pt --run-agent
EOF


bsub << EOF
#!/bin/bash
#BSUB -J run_baseline_pruning_coco
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o imagenet_standard_%J.out
#BSUB -e imagenet_standard_%J.err

uv run src/run_baseline_pruning.py --dataset coco --rl-agent-ckpt checkpoints/rl_agent_ppo_coco.pt --run-agent
EOF