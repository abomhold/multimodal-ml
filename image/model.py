import torch
import torch.F as F
import numpy
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
