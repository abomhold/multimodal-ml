import torch.nn as nn
from torchvision import models


def get_pretrained_vgg16(num_classes=2, freeze_features=True):
    model = models.vgg16(pretrained=True)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, num_classes)
    )

    return model


def get_pretrained_resnet50(num_classes=2, freeze_features=True):
    model = models.resnet50(pretrained=True)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_pretrained_efficientnet_b0(num_classes=2, freeze_features=True):
    model = models.efficientnet_b0(pretrained=True)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
