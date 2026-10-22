
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import Compose, Resize

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "cats_dogs_mobilenet_v2.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
SEED = 42

CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx",
}


class BinaryPetDataset(torch.utils.data.Dataset):
    """Convert the 37 pet breeds into Cat=0 and Dog=1."""

    def __init__(self, base_dataset, cat_indices):
        self.base_dataset = base_dataset
        self.cat_indices = cat_indices

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, breed_label = self.base_dataset[index]
        label = 0 if int(breed_label) in self.cat_indices else 1
        return image, label


def evaluate_model(model, data_loader, device):
    """Calculate classification accuracy on the validation set."""
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training device: {device}")

    # Load pretrained MobileNetV2 weights.
    weights = MobileNet_V2_Weights.DEFAULT

    # Preprocessing expected by the pretrained model.
    transform = Compose([
        Resize((224, 224)),
        weights.transforms(),
    ])

    # Download/load the Oxford-IIIT Pet dataset.
    dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        download=True,
        transform=transform,
    )

    # Find the 12 cat breed indices by their exact names.
    cat_indices = {
        index
        for index, name in enumerate(dataset.classes)
        if name in CAT_BREEDS
    }

    if len(cat_indices) != 12:
        raise ValueError(
            f"Expected 12 cat breeds, found {len(cat_indices)}."
        )

    print(f"Cat breeds detected: {len(cat_indices)}")
    print(f"Dog breeds detected: {len(dataset.classes) - len(cat_indices)}")

    binary_dataset = BinaryPetDataset(dataset, cat_indices)

    # Create an 80/20 train-validation split.
    train_size = int(0.8 * len(binary_dataset))
    val_size = len(binary_dataset) - train_size

    train_dataset, val_dataset = random_split(
        binary_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Build pretrained MobileNetV2.
    model = models.mobilenet_v2(weights=weights)

    # Freeze the pretrained feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the classifier for two classes: Cat and Dog.
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        validation_accuracy = evaluate_model(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.2%}"
        )

    # Save model weights and class-name mapping.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "class_names": ["Cat", "Dog"],
        "architecture": "MobileNetV2",
        "image_size": 224,
    }

    torch.save(checkpoint, MODEL_PATH)

    print(f"\nTraining complete.")
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
