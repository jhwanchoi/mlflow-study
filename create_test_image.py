"""Create test images from CIFAR-10 dataset for BentoML testing."""

import torchvision
from PIL import Image

# Download CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True
)

# Get first 5 images from different classes
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Save one image from each class
saved_count = {}
for i, (img, label) in enumerate(testset):
    class_name = class_names[label]

    if class_name not in saved_count:
        # Save image
        img.save(f"test_{class_name}.png")
        print(f"Saved: test_{class_name}.png (label: {label})")
        saved_count[class_name] = 1

    # Stop after saving 10 images (one per class)
    if len(saved_count) == 10:
        break

print(f"\nCreated {len(saved_count)} test images!")
