import torch
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
