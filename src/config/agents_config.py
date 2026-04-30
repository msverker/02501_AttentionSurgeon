DATASET_CONFIGS = {
    "coco": {
        "alpha": 1.5,               # Linear penalty for loss degradation
        "beta": 1.5,                # High reward for sparsity
        "gamma": 2,               # Exponential penalty multiplier
        "epsilon": 0.1,            # 10% loss degradation is "free"
        "safe_prune_bonus": 0.05,   # Salary for staying alive
        "reward_floor": -1.0,       # Max punishment per step
    },
    "ade20k": {
        "alpha": 0.5,               # Softer linear penalty (loss spikes fast here)
        "beta": 5.0,                # Still highly reward sparsity
        "gamma": 0.1,               # VERY soft exponential penalty 
        "epsilon": 0.20,            # Give it a wider 20% safe margin
        "safe_prune_bonus": 0.2,   # Higher salary to encourage bravery
        "reward_floor": -5.0,       # Shield remains the same
    },
    "imagenet": {
        "alpha": 0.5,               # Linear penalty for loss degradation
        "beta": 5.0,                # High reward for sparsity
        "gamma": 0.1,               # Exponential penalty multiplier
        "epsilon": 0.2,            # 20% loss degradation is "free"
        "safe_prune_bonus": 0.2,   # Salary for staying alive
        "reward_floor": -4.0,       # Max punishment per step
    }
}