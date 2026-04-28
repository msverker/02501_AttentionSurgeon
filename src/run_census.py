import torch
import torch.nn as nn
import numpy as np
from baseline.backbone import (
    DinoV2Backbone,
    get_imagenet_loaders,
    ClassificationHead,
    ADE20KHead,
    CocoHead,
)
from baseline.loaders import get_ade20k_dataloaders, get_coco_dataloaders
from head_census import AttentionCensus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="cls",
        choices=[
            "cls",
            "segmentaion",
            "object_detection",
        ],
        help="what to do when running the script (default: %(default)s)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = DinoV2Backbone(device=device)
    census = AttentionCensus(backbone)
    if args.mode == "cls":
        _, val_loader = get_imagenet_loaders(batch_size=32)

        # load pretrained probe
        probe = ClassificationHead(in_dim=768, num_classes=1000).to(device)
        probe.load_state_dict(torch.load("checkpoints/probe_imagenet.pt"))
        probe.eval()

        loss_fn = nn.CrossEntropyLoss()

        print("Running task-agnostic metrics...")
        results = census.run(val_loader, num_batches=5, task="cls")

        print("Running importance scoring...")
        del results  # free memory first
        torch.cuda.empty_cache()
        results = census.run(val_loader, num_batches=5, task="cls")

        importance = census.compute_importance(
            val_loader, probe, loss_fn, task="cls", num_batches=50
        )

    elif args.mode == "segmentation":
        _, val_loader = get_ade20k_dataloaders(
            root="/work3/s216143/02501_AttentionSurgeon/data/archive/",
            image_size=(224, 224),
            batch_size=4,
            num_workers=8,
        )
        # load segmentation probe
        probe = ADE20KHead().to(device)
        probe.load_state_dict(
            torch.load(
                "/work3/s216143/02501_AttentionSurgeon/src/checkpoints/ade20k_head.pth"
            )
        )
        probe.eval()

        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        print("Running task-agnostic metrics...")
        results = census.run(val_loader, num_batches=5, task="seg")

        print("Running importance scoring...")
        del results  # free memory first
        torch.cuda.empty_cache()
        results = census.run(val_loader, num_batches=5, task="seg")

        importance = census.compute_importance(
            val_loader, probe, loss_fn, task="seg", num_batches=25
        )
    elif args.mode == "object_detection":
        _, val_loader = get_coco_dataloaders(batch_size=4)

        # load coco probe
        probe = CocoHead().to(device)
        probe.load_state_dict(
            torch.load(
                "/work3/s216143/02501_AttentionSurgeon/src/checkpoints/coco_head.pth"
            )
        )
        probe.eval()

        print("Running task-agnostic metrics...")
        results = census.run(val_loader, num_batches=5, task="det")

        print("Running importance scoring...")
        del results  # free memory first
        torch.cuda.empty_cache()
        results = census.run(val_loader, num_batches=5, task="det")
        importance = census.compute_importance(
            val_loader,
            probe,
            loss_fn=None,  # not used for det
            task="det",
            num_batches=5,
        )

    results["importance_cls"] = importance

    np.savez(
        f"/work3/s216143/02501_AttentionSurgeon/src/npz_weights/head_profiles_{args.mode}.npz",
        entropy=results["entropy"].numpy(),
        distance=results["distance"].numpy(),
        activation_mag=results["magnitude"].numpy(),
        importance=results["importance_cls"].detach().numpy(),
    )
    print("Saved head_profiles.npz")
