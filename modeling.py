from torch import nn
from torchvision import models

from .config import SUPPORTED_MODELS


def _create_with_weights(model_name: str, pretrained: bool):
    if model_name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            return models.resnet18(weights=weights)
        except AttributeError:
            return models.resnet18(pretrained=pretrained)

    if model_name == "mobilenet_v3_small":
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            return models.mobilenet_v3_small(weights=weights)
        except AttributeError:
            return models.mobilenet_v3_small(pretrained=pretrained)

    if model_name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            return models.efficientnet_b0(weights=weights)
        except AttributeError:
            return models.efficientnet_b0(pretrained=pretrained)

    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")


def _replace_classifier_head(model, model_name: str, num_classes: int):
    if model_name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    if model_name in {"mobilenet_v3_small", "efficientnet_b0"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return

    raise ValueError(f"Unsupported model '{model_name}'.")


def create_transfer_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    model = _create_with_weights(model_name=model_name, pretrained=pretrained)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    _replace_classifier_head(model=model, model_name=model_name, num_classes=num_classes)
    return model
