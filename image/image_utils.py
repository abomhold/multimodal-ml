import os
# import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def get_image(image_path):
    images = []
    for image_file in os.path(image_path):
        image = Image.open(image_file)
        images.append(image)
    return images


def get_classes(class_path):
    df = pd.read_csv(class_path + "profile/profile.csv")
    gender = df["gender"]
    return gender
