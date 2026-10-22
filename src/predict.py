
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights

from src.model import build_model


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "cats_dogs_mobilenet_v2.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()

CLASS_NAMES = ["Cat", "Dog"]


def load_model():
    """Load the trained MobileNetV2 checkpoint."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = build_model(num_classes=2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()


def predict_image(image: Image.Image) -> dict:
    """Predict whether an image contains a cat or a dog."""
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return {
        "label": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
    }
