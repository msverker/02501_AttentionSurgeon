
class PruningEvaluator:
    def __init__(self, backbone, probe, dataloader, task="cls"):
        # stores backbone, probe, eval dataloader
        ...

    def evaluate(self, head_mask):
        # head_mask: (12, 12) binary tensor
        # returns: accuracy with this mask applied
        ...
        
    def run_pruning_strategy(self, strategy, n_steps=72):
        # strategy: function that given current mask + census data
        #           returns which head to prune next
        # returns: list of (n_pruned, accuracy, flops) at each step
        mask = torch.ones(12, 12)
        results = []
        for step in range(n_steps):
            layer, head = strategy(mask, census)
            mask[layer, head] = 0
            acc = self.evaluate(mask)
            flops_ratio = (144 - (step + 1)) / 144
            reward = acc * flops_ratio
            results.append((step + 1, acc, flops_ratio, reward))
        return results

    @staticmethod
    def random_strategy(mask, census):
        # pick a random unpruned head
        ...

    @staticmethod
    def magnitude_strategy(mask, census):
        # pick lowest magnitude unpruned head
        ...

    @staticmethod
    def importance_strategy(mask, census):
        # pick lowest importance unpruned head
        ...

    @staticmethod
    def _apply_pruning_hooks(backbone, mask):
        """
        mask: (12, 12) binary tensor — 0 = prune, 1 = keep
        returns: list of hooks to remove later
        """
        hooks = []
        head_dim = 768 // 12

        def make_hook(layer_idx, mask_row):
            def hook(module, input, output):
                # output is (B, seq_len, 768)
                B, S, _ = output.shape
                ctx = output.view(B, S, 12, head_dim)
                ctx = ctx * mask_row.to(ctx.device).view(1, 1, 12, 1)
                return ctx.view(B, S, 768)
            return hook

        for i, layer in enumerate(backbone.model.encoder.layer):
            h = layer.attention.output.register_forward_hook(make_hook(i, mask[i]))
            hooks.append(h)

        return hooks
