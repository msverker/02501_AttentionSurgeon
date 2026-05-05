
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

uv run src/visualize_pruning.py --dataset-name ImageNet --input results/baseline_pruning_imagenet_results.npz
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

uv run src/visualize_pruning.py --dataset-name ADE20K --input results/baseline_pruning_ade40k_results.npz
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

uv run src/visualize_pruning.py --dataset-name COCO --input results/baseline_pruning_coco_results.npz
EOF