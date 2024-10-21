import os
# import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_image(image_dir):
    images = {}
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        image_name = image_file.removesuffix('.jpg')
        images[image_name] = image
    return images


def get_classes(class_path):
    df = pd.read_csv(class_path)
    ids = df["userid"]
    genders = df["gender"].astype(int)
    assert genders.isin([0, 1]).all(), "Gender column contains invalid values"
    classes = {user_id: gender for user_id, gender in zip(ids, genders)}
    return classes


def create_dataloader(images, classes, batch_size=32):
    image_tensors = []
    class_tensors = []
    for user_id, image_tensor in images.items():
        image_tensors.append(image_tensor)
        class_tensors.append(classes[user_id])

    image_tensors = torch.stack(image_tensors)
    class_tensors = torch.tensor(class_tensors)

    dataset = TensorDataset(image_tensors, class_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
