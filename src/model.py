
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torch import nn


def build_model(num_classes: int = 2):
    """
    Build a MobileNetV2 model for image classification.

    Args:
        num_classes: Number of output classes.

    Returns:
        A pretrained MobileNetV2 model with a new classification head.
    """
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze pretrained feature extractor parameters.
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final layer for our two classes.
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)

    return model
