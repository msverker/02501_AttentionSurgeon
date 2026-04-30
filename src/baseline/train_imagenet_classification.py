import torch
import torch.nn as nn
from loaders import get_imagenet_loaders
from backbone import DinoV2Backbone, ClassificationHead, train_classification, evaluate

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    backbone = DinoV2Backbone(device=device)

    head = ClassificationHead(in_dim=768, num_classes=1000).to(device)

    train_loader, val_loader = get_imagenet_loaders(batch_size=64)

    print("Starting training...")
    epochs = 5
    train_classification(backbone, head, train_loader, val_loader, epochs=epochs)

    acc = evaluate(backbone, head, val_loader)
    print(f"Final Validation Accuracy: {acc:.4f}")

    print("Saving the classification head...")
    torch.save(head.state_dict(), "imagenet_classification_head.pth")
    print("Training complete and model saved.")

    import matplotlib.pyplot as plt
    import os

    os.makedirs("results", exist_ok=True)
    backbone.eval()
    head.eval()

    # Get a batch of validation images
    imgs, labels = next(iter(val_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        feats = backbone(imgs)
        outputs = head(feats)
        preds = outputs.argmax(dim=1)

    # Plot the first 8 images with their predicted and true labels
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(imgs):
            break
        
        # Unnormalize image for visualization
        # DinoV2 uses ImageNet mean/std
        img = imgs[i].cpu().permute(1, 2, 0).numpy()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = std * img + mean
        img = img.clip(0, 1)

        ax.imshow(img)
        ax.set_title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/imagenet_examples.png")
    print("Example predictions saved to results/imagenet_examples.png")

if __name__ == "__main__":
    main()
