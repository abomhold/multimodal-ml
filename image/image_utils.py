from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import v2
from image.image_config import WIDTH, HEIGHT

transforms = v2.Compose([
    v2.Resize([WIDTH, HEIGHT]),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_image(image_dir):
    images = {}
    for image_file in image_dir.iterdir():
        try:
            original_image = Image.open(image_file)
            image_copy = original_image.copy()
            augmented_image = transforms(image_copy)
            images[image_file.stem] = augmented_image
        except IOError:
            print(f"Error opening image: {image_file}")
    return images


def get_classes(class_path):
    df = pd.read_csv(class_path)
    ids = df["userid"]
    genders = df["gender"].astype(int)
    assert genders.isin([0, 1]).all(), "Gender column contains invalid values"
    classes = {user_id: gender for user_id, gender in zip(ids, genders)}
    return classes


def match_userid_image(image_dict, class_dict):
    matched_dict = {}
    for user_id, image_tensor in image_dict.items():
        matched_dict[user_id] = (image_tensor, class_dict[user_id])

    return matched_dict


def create_dataset(images, classes, batch_size=32):
    image_tensors = []
    class_tensors = []
    for user_id, image_tensor in images.items():
        image_tensors.append(image_tensor)
        class_tensors.append(classes[user_id])

    image_tensors = torch.stack(image_tensors)
    class_tensors = torch.tensor(class_tensors)

    dataset = TensorDataset(image_tensors, class_tensors)

    return dataset


def split_train_val_dataset(dataset, train_size=0.8, batch_size=32):
    dataset_size = len(dataset)
    train_len = int(train_size*dataset_size)
    val_len = dataset_size - train_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def process_dataframe(df):
    ids = df["userid"]
    genders = df["gender"].astype(int)
    assert genders.isin([0, 1]).all(), "Gender column contains invalid values"
    classes = {user_id: gender for user_id, gender in zip(ids, genders)}
    return classes
