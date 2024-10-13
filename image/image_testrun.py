from image_preprocessing import preprocess_image
from model import VGG16
import os
import torch.optim as optim


def test(input_path, device):
    image_dir = os.join(input_path, "image")
    images = preprocess_image(image_dir)
    classes = get_classes(input_path)

    # This is for gender.
    model = VGG16(2).to(device)
    optimizer = optim.adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    losses = []
    accuracies = []
    step_interval = 50

    model.train()
    for epoch in range(100):
        print("Yaaaay")
